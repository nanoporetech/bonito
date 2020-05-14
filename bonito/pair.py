#!/usr/bin/env python

"""
Bonito pair consensus decoding.

https://www.biorxiv.org/content/10.1101/2020.02.25.956771v1

$ bonito pair --half read-pairs.csv reads/ > basecalls.fasta
"""

import os
import sys
import time
from textwrap import wrap
from multiprocessing import Process, Queue, Lock, cpu_count
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import parasail
import numpy as np
from tqdm import tqdm

from bonito.util import get_raw_data
from bonito.util import accuracy, load_model
from fast_ctc_decode import beam_search, beam_search_2d


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
    def __init__(self, queue, lock, alphabet, beamsize=128, padding=20, match=60, threshold=0.05):
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

    sys.stderr.write("> loading model\n")
    model = load_model('dna_r9.4.1', args.device, half=args.half)
    decoders = PairDecoderWriterPool(model.alphabet, procs=args.num_procs)

    t0 = time.perf_counter()
    sys.stderr.write("> calling\n")

    with torch.no_grad(), open(args.pairs_file) as pairs, decoders:

        for pair in tqdm(pairs, ascii=True, ncols=100):

            read_1, read_2 = pair.strip().split(args.sep)
            read_1 = os.path.join(args.reads_directory, read_1)
            read_2 = os.path.join(args.reads_directory, read_2)

            if not (os.path.exists(read_1) and os.path.exists(read_2)): continue

            read_id_1, raw_data_1 = next(get_raw_data(read_1))

            if len(raw_data_1) > max_read_size:
                sys.stderr.write("> skipping long read %s (%s samples)\n" % (read_id_1, len(raw_data_1)))
                continue

            read_id_2, raw_data_2 = next(get_raw_data(read_2))

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
    return parser
