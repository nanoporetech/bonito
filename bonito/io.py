"""
Bonito Input/Output
"""

import os
import sys
import csv
import pandas as pd
from warnings import warn
from threading import Thread
from logging import getLogger
from contextlib import contextmanager
from os.path import realpath, splitext, dirname

import numpy as np
from mappy import revcomp

import bonito
from bonito.cli.convert import typical_indices


logger = getLogger('bonito')


class CSVLogger:
    def __init__(self, filename, sep=','):
        self.filename = str(filename)
        if os.path.exists(self.filename):
            with open(self.filename) as f:
                self.columns = csv.DictReader(f).fieldnames
        else:
            self.columns = None
        self.fh = open(self.filename, 'a', newline='')
        self.csvwriter = csv.writer(self.fh, delimiter=sep)
        self.count = 0

    def set_columns(self, columns):
        if self.columns:
            raise Exception('Columns already set')
        self.columns = list(columns)
        self.csvwriter.writerow(self.columns)

    def append(self, row):
        if self.columns is None:
            self.set_columns(row.keys())
        self.csvwriter.writerow([row.get(k, '-') for k in self.columns])
        self.count += 1
        if self.count > 100:
            self.count = 0
            self.fh.flush()

    def close(self):
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@contextmanager
def devnull(*args, **kwds):
    """
    A context manager that sends all out stdout & stderr to devnull.
    """
    save_fds = [os.dup(1), os.dup(2)]
    null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
    os.dup2(null_fds[0], 1)
    os.dup2(null_fds[1], 2)
    try:
        yield
    finally:
        os.dup2(save_fds[0], 1)
        os.dup2(save_fds[1], 2)
        for fd in null_fds + save_fds: os.close(fd)


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
            'MD:Z:%s' % mapping.MD,
        ])))
    fd.flush()


def summary_file():
    """
    Return the filename to use for the summary tsv.
    """
    stdout = realpath('/dev/fd/1')
    if sys.stdout.isatty() or stdout.startswith('/proc'):
        return 'summary.tsv'
    return '%s_summary.tsv' % splitext(stdout)[0]


summary_field_names = [
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
    #if alignment
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
    'alignment_mapq',
    'alignment_strand_coverage',
    'alignment_identity',
    'alignment_accuracy',
]


def summary_row(read, seqlen, qscore, alignment=False):
    """
    Summary tsv row.
    """
    fields = [
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
    ]

    if alignment:

        ins = sum(count for count, op in alignment.cigar if op == 1)
        dels = sum(count for count, op in alignment.cigar if op == 2)
        subs = alignment.NM - ins - dels
        length = alignment.blen
        matches = length - ins - dels
        correct = alignment.mlen

        fields.extend([
            alignment.ctg,
            alignment.r_st,
            alignment.r_en,
            alignment.q_st if alignment.strand == +1 else seqlen - alignment.q_en,
            alignment.q_en if alignment.strand == +1 else seqlen - alignment.q_st,
            '+' if alignment.strand == +1 else '-',
            length, matches, correct,
            ins, dels, subs,
            alignment.mapq,
            (alignment.q_en - alignment.q_st) / seqlen,
            correct / matches,
            correct / length,
        ])

    elif alignment is None:
        fields.extend(
            ['*', -1, -1, -1, -1, '*', 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0]
        )

    return dict(zip(summary_field_names, fields))


duplex_summary_field_names = [
    'filename_template',
    'read_id_template',
    'filename_complement',
    'read_id_complement',
    'run_id',
    'channel_template',
    'mux_template',
    'channel_complement',
    'mux_complement',
    'sequence_length_duplex',
    'mean_qscore_duplex',
    #if alignment
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
    'alignment_mapq',
    'alignment_strand_coverage',
    'alignment_identity',
    'alignment_accuracy',
]


def duplex_summary_row(read_temp, comp_read, seqlen, qscore, alignment=False):
    """
    Duplex summary tsv row.
    """
    fields = [
        read_temp.filename,
        read_temp.read_id,
        comp_read.filename,
        comp_read.read_id,
        read_temp.run_id,
        read_temp.channel,
        read_temp.mux,
        comp_read.channel,
        comp_read.mux,
        seqlen,
        qscore,
    ]

    if alignment:

        ins = sum(count for count, op in alignment.cigar if op == 1)
        dels = sum(count for count, op in alignment.cigar if op == 2)
        subs = alignment.NM - ins - dels
        length = alignment.blen
        matches = length - ins - dels
        correct = alignment.mlen

        fields.extend([
            alignment.ctg,
            alignment.r_st,
            alignment.r_en,
            alignment.q_st if alignment.strand == +1 else seqlen - alignment.q_en,
            alignment.q_en if alignment.strand == +1 else seqlen - alignment.q_st,
            '+' if alignment.strand == +1 else '-',
            length, matches, correct,
            ins, dels, subs,
            alignment.mapq,
            (alignment.q_en - alignment.q_st) / seqlen,
            correct / matches,
            correct / length,
        ])

    elif alignment is None:
        fields.extend(
            ['*', -1, -1, -1, -1, '*', 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0]
        )

    return dict(zip(duplex_summary_field_names, fields))



class Writer(Thread):

    def __init__(self, iterator, aligner, fd=sys.stdout, fastq=False, duplex=False):
        super().__init__()
        self.fd = fd
        self.log = []
        self.fastq = fastq
        self.duplex = duplex
        self.aligner = aligner
        self.iterator = iterator
        self.write_headers()

    def write_headers(self):
        if self.aligner:
            write_sam_header(self.aligner, fd=self.fd)

    def run(self):

        with CSVLogger(summary_file(), sep='\t') as summary:
            for read, res in self.iterator:

                seq = res['sequence']
                qstring = res.get('qstring', '*')
                mean_qscore = res.get('mean_qscore', 0.0)
                mapping = res.get('mapping', False)

                if self.duplex:
                    samples = len(read[0].signal) + len(read[1].signal)
                    read_id = '%s;%s' % (read[0].read_id, read[1].read_id)
                else:
                    samples = len(read.signal)
                    read_id = read.read_id

                if len(seq):
                    if self.aligner:
                        write_sam(read_id, seq, qstring, mapping, fd=self.fd, unaligned=mapping is None)
                    else:
                        if self.fastq:
                            write_fastq(read_id, seq, qstring, fd=self.fd)
                        else:
                            write_fasta(read_id, seq, fd=self.fd)

                    if self.duplex:
                        summary.append(duplex_summary_row(read[0], read[1], len(seq), mean_qscore, alignment=mapping))
                    else:
                        summary.append(summary_row(read, len(seq), mean_qscore, alignment=mapping))

                    self.log.append((read_id, samples))

                else:
                    logger.warn("> skipping empty sequence %s", read_id)


class CTCWriter(Thread):
    """
    CTC writer process that writes output numpy training data.
    """
    def __init__(self, iterator, aligner, min_coverage, min_accuracy, fd=sys.stdout):
        super().__init__()
        self.fd = fd
        self.log = []
        self.aligner = aligner
        self.iterator = iterator
        self.min_coverage = min_coverage
        self.min_accuracy = min_accuracy
        self.write_headers()

    def write_headers(self):
        if self.aligner:
            write_sam_header(self.aligner, fd=self.fd)

    def run(self):

        chunks = []
        targets = []
        lengths = []

        with CSVLogger(summary_file(), sep='\t') as summary:
            for read, ctc_data in self.iterator:

                seq = ctc_data['sequence']
                qstring = ctc_data['qstring']
                mean_qscore = ctc_data['mean_qscore']
                mapping = ctc_data.get('mapping', False)

                self.log.append((read.read_id, len(read.signal)))

                if len(seq) == 0 or mapping is None:
                    continue

                cov = (mapping.q_en - mapping.q_st) / len(seq)
                acc = mapping.mlen / mapping.blen
                refseq = self.aligner.seq(mapping.ctg, mapping.r_st, mapping.r_en)

                if acc < self.min_accuracy or cov < self.min_coverage or 'N' in refseq:
                    continue

                write_sam(read.read_id, seq, qstring, mapping, fd=self.fd, unaligned=mapping is None)
                summary.append(summary_row(read, len(seq), mean_qscore, alignment=mapping))

                if mapping.strand == -1:
                    refseq = revcomp(refseq)

                target = [int(x) for x in refseq.translate({65: '1', 67: '2', 71: '3', 84: '4'})]
                targets.append(target)
                chunks.append(read.signal)
                lengths.append(len(target))

        if len(chunks) == 0:
            sys.stderr.write("> no suitable ctc data to write\n")
            return

        chunks = np.array(chunks, dtype=np.float16)
        targets_ = np.zeros((chunks.shape[0], max(lengths)), dtype=np.uint8)
        for idx, target in enumerate(targets): targets_[idx, :len(target)] = target
        lengths = np.array(lengths, dtype=np.uint16)
        indices = np.random.permutation(typical_indices(lengths))

        chunks = chunks[indices]
        targets_ = targets_[indices]
        lengths = lengths[indices]

        summary = pd.read_csv(summary_file(), sep='\t')
        summary.iloc[indices].to_csv(summary_file(), sep='\t', index=False)

        output_directory = '.' if sys.stdout.isatty() else dirname(realpath('/dev/fd/1'))
        np.save(os.path.join(output_directory, "chunks.npy"), chunks)
        np.save(os.path.join(output_directory, "references.npy"), targets_)
        np.save(os.path.join(output_directory, "reference_lengths.npy"), lengths)

        sys.stderr.write("> written ctc training data\n")
        sys.stderr.write("  - chunks.npy with shape (%s)\n" % ','.join(map(str, chunks.shape)))
        sys.stderr.write("  - references.npy with shape (%s)\n" % ','.join(map(str, targets_.shape)))
        sys.stderr.write("  - reference_lengths.npy shape (%s)\n" % ','.join(map(str, lengths.shape)))

    def stop(self):
        self.join()
