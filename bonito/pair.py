#!/usr/bin/env python

"""
Bonito pair consensus decoding.

https://www.biorxiv.org/content/10.1101/2020.02.25.956771v1

$ bonito pair --half read-pairs.csv reads/ > basecalls.fasta
"""

import os
import sys
import time
import json
from glob import glob
from textwrap import wrap
from os.path import basename
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue, Lock, cpu_count
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import parasail
import numpy as np
from tqdm import tqdm

from bonito.util import accuracy, load_model
from bonito.util import get_raw_data_for_read
from fast_ctc_decode import beam_search, beam_search_2d
from ont_fast5_api.fast5_interface import get_fast5_file


def get_read_ids(filename):
    """
    Return a dictionary of read_id -> filename mappings.
    """
    with get_fast5_file(filename, 'r') as f5:
        return {
            read.read_id: basename(filename) for read in f5.get_reads()
        }


def build_index(files, workers=8):
    """
    Build an index of read ids to filename mappings
    """
    index = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in tqdm(pool.map(get_read_ids, files), ascii=True, ncols=100):
            index.update(res)
    return index


def build_envelope(len1, seq1, path1, len2, seq2, path2, padding=15):

    # needleman-wunsch alignment with constant gap penalty.
    aln = parasail.nw_trace_striped_32(seq2, seq1, 2, 2, parasail.dnafull)

    # pair up positions
    alignment = np.column_stack([
        np.cumsum([x != '-' for x in aln.traceback.ref]) - 1,
        np.cumsum([x != '-' for x in aln.traceback.query]) - 1
    ])

    path_range1 = np.column_stack([path1, path1[1:] + [len1]])
    path_range2 = np.column_stack([path2, path2[1:] + [len2]])

    envelope = np.full((len1, 2), -1, dtype=int)

    for idx1, idx2 in alignment.clip(0):

        st_1, en_1 = path_range1[idx1]
        st_2, en_2 = path_range2[idx2]

        for idx in range(st_1, en_1):
            if st_2 < envelope[idx, 0] or envelope[idx, 0] < 0:
                envelope[idx, 0] = st_2
            if en_2 > envelope[idx, 1] or envelope[idx, 1] < 0:
                envelope[idx, 1] = en_2

    # add a little padding to ensure some overlap
    envelope[:, 0] = envelope[:, 0] - padding
    envelope[:, 1] = envelope[:, 1] + padding
    envelope = np.clip(envelope, 0, len2)

    prev_end = 0
    for i in range(envelope.shape[0]):

        if envelope[i, 0] > envelope[i, 1]:
            envelope[i, 0] = 0

        if envelope[i, 0] > prev_end:
            envelope[i, 0] = prev_end

        prev_end = envelope[i, 1]

    return envelope.astype(np.uint64)


class PairDecoderWriterPool:
    """
    Simple pool of `procs` pairwise decoders
    """
    def __init__(self, alphabet, procs=0, **kwargs):
        self.lock = Lock()
        self.queue = Queue()
        self.procs = procs if procs else cpu_count()
        self.decoders = []
        for _ in range(self.procs):
            decoder = PairDecoderWriter(self.queue, self.lock, alphabet, **kwargs)
            decoder.start()
            self.decoders.append(decoder)

    def stop(self):
        for decoder in self.decoders: self.queue.put(None)
        for decoder in self.decoders: decoder.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class PairDecoderWriter(Process):
    """
    Pairwise Decoder Process that writes the consensus fasta records to stdout
    """
    def __init__(self, queue, lock, alphabet, beamsize=128, padding=15, match=60, threshold=0.01):
        super().__init__()
        self.lock = lock
        self.queue = queue
        self.match = match
        self.minseqlen = 10
        self.padding = padding
        self.alphabet = alphabet
        self.beamsize = beamsize
        self.threshold = threshold

    def run(self):
        while True:

            job = self.queue.get()
            if job is None: return

            read_id_1, logits_1, read_id_2, logits_2 = job

            # revcomp decode the second read
            logits_2 = logits_2[::-1, [0, 4, 3, 2, 1]]

            # fast-ctc-decode expects probs (not logprobs)
            probs_1 = np.exp(logits_1)
            probs_2 = np.exp(logits_2)

            temp_seq, temp_path = beam_search(
                probs_1, self.alphabet, beam_size=16, beam_cut_threshold=self.threshold
            )
            comp_seq, comp_path = beam_search(
                probs_2, self.alphabet, beam_size=16, beam_cut_threshold=self.threshold
            )

            # catch any bad reads before attempt to align (parasail will segfault)
            if len(temp_seq) < self.minseqlen or len(comp_seq) < self.minseqlen:
                continue

            # check template/complement agreement
            if accuracy(temp_seq, comp_seq) < self.match:
                continue

            env = build_envelope(probs_1.shape[0], temp_seq, temp_path, probs_2.shape[0], comp_seq, comp_path, padding=self.padding)

            consensus = beam_search_2d(
                probs_1, probs_2, self.alphabet, envelope=env,
                beam_size=self.beamsize, beam_cut_threshold=self.threshold
            )

            with self.lock:
                sys.stdout.write(">%s;%s;\n" % (read_id_1, read_id_2))
                sys.stdout.write("%s\n" % os.linesep.join(wrap(consensus, 100)))
                sys.stdout.flush()


def main(args):

    samples = 0
    num_pairs = 0
    max_read_size = 4e6
    dtype = np.float16 if args.half else np.float32

    if args.index is not None:
        sys.stderr.write("> loading read index\n")
        index = json.load(open(args.index, 'r'))
    else:
        sys.stderr.write("> building read index\n")
        files = list(glob(os.path.join(args.reads_directory, '*.fast5')))
        index = build_index(files)
        if args.save_index:
            with open('bonito-read-id.idx', 'w') as f:
                json.dump(index, f)

    sys.stderr.write("> loading model\n")
    model = load_model('dna_r9.4.1', args.device, half=args.half)
    decoders = PairDecoderWriterPool(model.alphabet, procs=args.num_procs)

    t0 = time.perf_counter()
    sys.stderr.write("> calling\n")

    with torch.no_grad(), open(args.pairs_file) as pairs, decoders:

        for pair in tqdm(pairs, ascii=True, ncols=100):

            read_id_1, read_id_2 = pair.strip().split(args.sep)

            if read_id_1 not in index or read_id_2 not in index: continue

            read_1 = get_raw_data_for_read(os.path.join(args.reads_directory, index[read_id_1]), read_id_1)
            raw_data_1 = read_1.signal

            if len(raw_data_1) > max_read_size:
                sys.stderr.write("> skipping long read %s (%s samples)\n" % (read_id_1, len(raw_data_1)))
                continue

            read_2 = get_raw_data_for_read(os.path.join(args.reads_directory, index[read_id_2]), read_id_2)
            raw_data_2 = read_2.signal

            if len(raw_data_2) > max_read_size:
                sys.stderr.write("> skipping long read %s (%s samples)\n" % (read_id_2, len(raw_data_2)))
                continue

            # call the template strand
            raw_data_1 = raw_data_1[np.newaxis, np.newaxis, :].astype(dtype)
            gpu_data_1 = torch.tensor(raw_data_1).to(args.device)
            logits_1 = model(gpu_data_1).cpu().numpy().squeeze().astype(np.float32)

            # call the complement strand
            raw_data_2 = raw_data_2[np.newaxis, np.newaxis, :].astype(dtype)
            gpu_data_2 = torch.tensor(raw_data_2).to(args.device)
            logits_2 = model(gpu_data_2).cpu().numpy().squeeze().astype(np.float32)

            num_pairs += 1
            samples += raw_data_1.shape[-1] + raw_data_2.shape[-1]

            # pair decode
            decoders.queue.put((read_id_1, logits_1, read_id_2, logits_2))

    duration = time.perf_counter() - t0

    sys.stderr.write("> completed pairs: %s\n" % num_pairs)
    sys.stderr.write("> samples per second %.1E\n" % (samples / duration))
    sys.stderr.write("> done\n")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("pairs_file")
    parser.add_argument("reads_directory")
    parser.add_argument("--sep", default=' ')
    parser.add_argument("--half", action="store_true", default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-procs", default=0, type=int)
    parser.add_argument("--index", default=None)
    parser.add_argument("--save-index", action="store_true", default=False)
    return parser
