"""
Bonito Input/Output
"""

import os
import sys
from warnings import warn
from logging import getLogger
from os.path import realpath, splitext, dirname
from multiprocessing import Process, Queue, Lock, cpu_count

import numpy as np
from tqdm import tqdm
from mappy import revcomp

import bonito
from bonito.training import ChunkDataSet
from bonito.convert import filter_chunks
from bonito.util import mean_qscore_from_qstring


logger = getLogger('bonito')


def summary_file():
    """
    Return the filename to use for the summary tsv.
    """
    if sys.stdout.isatty():
        return 'summary.tsv'
    return '%s_summary.tsv' % splitext(realpath('/dev/fd/1'))[0]


def write_summary_header(fd, alignment=None, sep='\t'):
    """
    Write the summary tsv header.
    """
    fields = [
        'filename',
        'read_id',
        'run_id',
        'channel',
        'mux',
        'start_time',
        'duration',
        'template_start',
        'template_duration',
        'sequence_length_template',
        'mean_qscore_template',
    ]
    if alignment:
        fields.extend([
            'alignment_genome',
            'alignment_genome_start',
            'alignment_genome_end',
            'alignment_strand_start',
            'alignment_strand_end',
            'alignment_direction',
            'alignment_length',
            'alignment_num_aligned',
            'alignment_num_correct',
            'alignment_num_insertions',
            'alignment_num_deletions',
            'alignment_num_substitutions',
            'alignment_strand_coverage',
            'alignment_identity',
            'alignment_accuracy',
        ])
    fd.write('%s\n' % sep.join(fields))
    fd.flush()


def write_summary_row(fd, read, seqlen, qscore, alignment=False, sep='\t'):
    """
    Write a summary tsv row.
    """
    fields = [str(field) for field in [
        read.filename,
        read.read_id,
        read.run_id,
        read.channel,
        read.mux,
        read.start,
        read.duration,
        read.template_start,
        read.template_duration,
        seqlen,
        qscore,
    ]]

    if alignment:

        ins = sum(count for count, op in alignment.cigar if op == 1)
        dels = sum(count for count, op in alignment.cigar if op == 2)
        subs = alignment.NM - ins - dels
        length = alignment.blen
        matches = length - ins - dels
        correct = alignment.mlen

        fields.extend([str(field) for field in [
            alignment.ctg,
            alignment.r_st,
            alignment.r_en,
            alignment.q_st if alignment.strand == +1 else seqlen - alignment.q_en,
            alignment.q_en if alignment.strand == +1 else seqlen - alignment.q_st,
            '+' if alignment.strand == +1 else '-',
            length, matches, correct,
            ins, dels, subs,
            (alignment.q_en - alignment.q_st) / seqlen,
            correct / matches,
            correct / length,
        ]])

    elif alignment is None:
        fields.extend([str(field) for field in
            ['*', -1, -1, -1, -1, '*', 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0]
        ])

    fd.write('%s\n' % sep.join(fields))
    fd.flush()


def write_fasta(header, sequence, fd=sys.stdout):
    """
    Write a fasta record to a file descriptor.
    """
    fd.write(">%s\n" % header)
    fd.write("%s\n" % sequence)
    fd.flush()


def write_fastq(header, sequence, qstring, fd=sys.stdout):
    """
    Write a fastq record to a file descriptor.
    """
    fd.write("@%s\n" % header)
    fd.write("%s\n" % sequence)
    fd.write("+\n")
    fd.write("%s\n" % qstring)
    fd.flush()


def write_sam_header(aligner, fd=sys.stdout, sep='\t'):
    """
    Write the SQ & PG sam headers to a file descriptor.
    """
    fd.write('%s\n' % os.linesep.join([
        sep.join([
            '@SQ', 'SN:%s' % name, 'LN:%s' % len(aligner.seq(name))
        ]) for name in aligner.seq_names
     ]))

    fd.write('%s\n' % sep.join([
        '@PG',
        'ID:bonito',
        'PN:bonito',
        'VN:%s' % bonito.__version__,
        'CL:%s' % ' '.join(sys.argv),
    ]))
    fd.flush()


def write_sam(read_id, sequence, qstring, mapping, fd=sys.stdout, unaligned=False, sep='\t'):
    """
    Write a sam record to a file descriptor.
    """
    if unaligned:
        fd.write("%s\n" % sep.join(map(str, [
            read_id, 4, '*', 0, 0, '*', '*', 0, 0, sequence, qstring, 'NM:i:0'
        ])))
    else:
        softclip = [
            '%sS' % mapping.q_st if mapping.q_st else '',
            mapping.cigar_str,
            '%sS' % (len(sequence) - mapping.q_en) if len(sequence) - mapping.q_en else ''
        ]
        fd.write("%s\n" % sep.join(map(str, [
            read_id,
            0 if mapping.strand == +1 else 16,
            mapping.ctg,
            mapping.r_st + 1,
            mapping.mapq,
            ''.join(softclip if mapping.strand == +1 else softclip[::-1]),
            '*', 0, 0,
            sequence if mapping.strand == +1 else revcomp(sequence),
            qstring,
            'NM:i:%s' % mapping.NM,
        ])))
    fd.flush()


class ProcessIterator(Process):
    """
    Runs an iterator in a separate process
    """
    def __init__(self, iterator, maxsize=5, progress=False):
        super().__init__()
        self.progress = progress
        self.iterator = iterator
        self.queue = Queue(maxsize)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        if self.progress:
            self.iterator = tqdm(self.iterator, ascii=True, ncols=100, leave=False)
        for item in self.iterator:
            self.queue.put(item)
        self.queue.put(None)

    def stop(self):
        self.join()


class CTCWriter(Process):
    """
    CTC writer process that writes output numpy training data
    """
    def __init__(self, model, aligner, min_coverage=0.90, min_accuracy=0.90):
        super().__init__()
        self.model = model
        self.queue = Queue()
        self.aligner = aligner
        self.min_coverage = min_coverage
        self.min_accuracy = min_accuracy

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.queue.put(None)
        self.stop()

    def run(self):

        chunks = []
        targets = []
        target_lens = []

        while True:

            job = self.queue.get()
            if job is None: break
            chunks_, predictions = job

            # convert logprobs to probs
            predictions = np.exp(predictions.astype(np.float32))

            for chunk, pred in zip(chunks_, predictions):

                try:
                    sequence = self.model.decode(pred)
                except:
                    continue

                if not sequence:
                    continue

                for mapping in self.aligner.map(sequence):
                    cov = (mapping.q_en - mapping.q_st) / len(sequence)
                    acc = mapping.mlen / mapping.blen
                    refseq = self.aligner.seq(mapping.ctg, mapping.r_st + 1, mapping.r_en)
                    if 'N' in refseq: continue
                    if mapping.strand == -1: refseq = revcomp(refseq)
                    break
                else:
                    continue

                if acc > self.min_accuracy and cov > self.min_accuracy:
                    chunks.append(chunk.squeeze())
                    targets.append([
                        int(x) for x in refseq.translate({65: '1', 67: '2', 71: '3', 84: '4'})
                    ])
                    target_lens.append(len(refseq))

        if len(chunks) == 0: return

        chunks = np.array(chunks, dtype=np.float32)
        chunk_lens = np.full(chunks.shape[0], chunks.shape[1], dtype=np.int16)

        targets_ = np.zeros((chunks.shape[0], max(target_lens)), dtype=np.uint8)
        for idx, target in enumerate(targets): targets_[idx, :len(target)] = target
        target_lens = np.array(target_lens, dtype=np.uint16)

        training = ChunkDataSet(chunks, chunk_lens, targets_, target_lens)
        training = filter_chunks(training)

        output_directory = '.' if sys.stdout.isatty() else dirname(realpath('/dev/fd/1'))
        np.save(os.path.join(output_directory, "chunks.npy"), training.chunks.squeeze(1))
        np.save(os.path.join(output_directory, "chunk_lengths.npy"), training.chunk_lengths)
        np.save(os.path.join(output_directory, "references.npy"), training.targets)
        np.save(os.path.join(output_directory, "reference_lengths.npy"), training.target_lengths)

        sys.stderr.write("> written ctc training data\n")
        sys.stderr.write("  - chunks.npy with shape (%s)\n" % ','.join(map(str, training.chunks.squeeze(1).shape)))
        sys.stderr.write("  - chunk_lengths.npy with shape (%s)\n" % ','.join(map(str, training.chunk_lengths.shape)))
        sys.stderr.write("  - references.npy with shape (%s)\n" % ','.join(map(str, training.targets.shape)))
        sys.stderr.write("  - reference_lengths.npy shape (%s)\n" % ','.join(map(str, training.target_lengths.shape)))

    def stop(self):
        self.join()


class DecoderWriterPool:
   """
   Simple pool of decoder writers
   """
   def __init__(self, model, procs=4, aligner=None, **kwargs):
       self.lock = Lock()
       self.queue = Queue()
       self.procs = procs if procs else cpu_count()
       self.aligner = aligner
       self.decoders = []

       if aligner: write_sam_header(aligner)

       with open(summary_file(), 'w') as summary:
           write_summary_header(summary, alignment=aligner)

       for _ in range(self.procs):
           decoder = DecoderWriter(model, self.queue, self.lock, aligner=aligner, **kwargs)
           decoder.start()
           self.decoders.append(decoder)

   def stop(self):
       for decoder in self.decoders: self.queue.put(None)
       for decoder in self.decoders: decoder.join()

   def __enter__(self):
       return self

   def __exit__(self, exc_type, exc_val, exc_tb):
       self.stop()


class DecoderWriter(Process):
    """
    Decoder Process that writes output records to stdout
    """
    def __init__(self, model, queue, lock, fastq=False, beamsize=5, aligner=None):
        super().__init__()
        self.queue = queue
        self.lock = lock
        self.model = model
        self.fastq = fastq
        self.aligner = aligner
        self.beamsize = beamsize

    def run(self):
        while True:
            job = self.queue.get()
            if job is None: return
            read, predictions = job

            # convert logprobs to probs
            predictions = np.exp(predictions.astype(np.float32))

            sequence, path = self.model.decode(
                predictions, beamsize=self.beamsize, qscores=True, return_path=True
            )
            sequence, qstring = sequence[:len(path)], sequence[len(path):]
            mean_qscore = mean_qscore_from_qstring(qstring)

            if not self.fastq:  # beam search
                qstring = '*'
                try:
                    sequence, path = self.model.decode(
                        predictions, beamsize=self.beamsize, qscores=False, return_path=True
                    )
                except:
                    pass

            if not self.aligner:
                mapping = False

            if sequence:
                with self.lock, open(summary_file(), 'a') as summary:
                    if self.aligner:
                        for mapping in self.aligner.map(sequence):
                            write_sam(read.read_id, sequence, qstring, mapping)
                            break
                        else:
                            mapping = None
                            write_sam(read.read_id, sequence, qstring, mapping, unaligned=True)
                    elif self.fastq:
                        write_fastq(read.read_id, sequence, qstring)
                    else:
                        write_fasta(read.read_id, sequence)
                    write_summary_row(summary, read, len(sequence), mean_qscore, alignment=mapping)
            else:
                logger.warn("> skipping empty sequence %s", read.read_id)
