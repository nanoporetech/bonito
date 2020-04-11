#!/usr/bin/env python

"""
Convert a Taiyaki chunkify training file to set of Bonito CTC .npy files
"""

import os
import h5py
import toml
import random
import numpy as np
from bisect import bisect_left
from argparse import ArgumentParser
from itertools import chain, zip_longest, groupby
from argparse import ArgumentDefaultsHelpFormatter


def align(samples, pointers, reference):
    """ align to the start of the mapping """
    squiggle_duration = len(samples)
    mapped_off_the_start = len(pointers[pointers < 0])
    mapped_off_the_end = len(pointers[pointers >= squiggle_duration])
    pointers = pointers[mapped_off_the_start:len(pointers) - mapped_off_the_end]
    reference = reference[mapped_off_the_start:len(reference) - mapped_off_the_end]
    return samples[pointers[0]:pointers[-1]], pointers - pointers[0], reference


def scale(read, samples, normalise=True):
    """ scale and normalise a read """
    scaling = read.attrs['range'] / read.attrs['digitisation']
    scaled = (scaling * (samples + read.attrs['offset'])).astype(np.float32)
    if normalise:
        return (scaled - read.attrs['shift_frompA']) / read.attrs['scale_frompA']
    return scaled


def boundary(sequence, r=5):
    """ check if we are on a homopolymer boundary """
    return len(set(sequence[-r:])) == 1


def num_reads(tfile):
    """ return the sample lengths for each read in the training file """
    with h5py.File(tfile, 'r') as training_file:
        return len(training_file['Reads'])


def get_reads(tfile):
    """ get each dataset per read """
    with h5py.File(tfile, 'r') as training_file:
        for read_id in np.random.permutation(training_file['Reads']):
            read = training_file['Reads/%s' % read_id]
            reference = read['Reference'][:]
            pointers = read['Ref_to_signal'][:]
            samples = read['Dacs'][:]
            samples = scale(read, samples)
            samples, pointers, reference = align(samples, pointers, reference)
            yield read_id, samples, reference, pointers


def main(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_directory, exist_ok=True)

    read_idx = 0
    chunk_idx = 0
    chunk_count = 0

    min_bases = 0
    max_bases = 0
    off_the_end_ref = 0
    off_the_end_sig = 0
    min_run_count = 0
    homopolymer_boundary = 0

    total_reads = num_reads(args.chunkify_file)

    chunks = np.zeros((args.chunks, args.max_seq_len * args.max_samples_per_base), dtype=np.float32)
    chunk_lengths = np.zeros(args.chunks, dtype=np.uint16)

    targets = np.zeros((args.chunks, args.max_seq_len), dtype=np.uint8)
    target_lengths = np.zeros(args.chunks, dtype=np.uint16)

    with open(os.path.join(args.output_directory, 'config.toml'), 'w') as conf:
        toml.dump(dict(chunks=vars(args)), conf)

    for read_id, samples, reference, pointers in get_reads(args.chunkify_file):

        read_idx += 1

        squiggle_duration = len(samples)
        sequence_length = len(reference) - 1

        # first chunk
        seq_starts = 0
        seq_ends = np.random.randint(args.min_seq_len, args.max_seq_len)

        repick = int((args.max_seq_len - args.min_seq_len) / 2)
        while boundary(reference[seq_starts:seq_ends]) and repick:
            seq_ends = seq_starts + np.random.randint(args.min_seq_len, args.max_seq_len)
            seq_ends = min(seq_ends, sequence_length)
            repick -= 1

        chunk_idxs = [(seq_starts, seq_ends)]

        # variable size chunks with overlap
        while seq_ends < sequence_length - args.min_seq_len:

            # overlap chunks with +/- 3% of max seq len
            overlap = np.int32(args.max_seq_len * 0.03)

            seq_starts = seq_ends + np.random.randint(-overlap, overlap)
            seq_ends = seq_starts + np.random.randint(args.min_seq_len, args.max_seq_len)
            seq_ends = min(seq_ends, sequence_length)

            repick = int((args.max_seq_len - args.min_seq_len) / 2)
            while boundary(reference[seq_starts:seq_ends]) and repick:
                seq_ends = seq_starts + np.random.randint(args.min_seq_len, args.max_seq_len)
                seq_ends = min(seq_ends, sequence_length)
                repick -= 1

            chunk_idxs.append((seq_starts, seq_ends))

        for start, end in chunk_idxs:

            chunk_idx += 1

            if end > sequence_length:
                print(read_id, end, sequence_length)
                off_the_end_ref += 1
                continue

            squiggle_start = pointers[start]
            squiggle_end = pointers[end + 1] # fence post mapping
            squiggle_length = squiggle_end - squiggle_start

            reference_length = end - start

            samples_per_base = squiggle_length / reference_length

            if samples_per_base < args.min_samples_per_base:
                min_bases += 1
                continue

            if samples_per_base > args.max_samples_per_base:
                max_bases += 1
                continue

            if squiggle_end > squiggle_duration:
                off_the_end_sig += 1
                continue

            longest_run = max(len(list(run)) for label, run in groupby(reference[start:end]))

            if longest_run < args.min_run:
                min_run_count += 1
                continue

            if boundary(reference[start:end]):
                homopolymer_boundary += 1
                # continue - include the chunk anyway

            chunks[chunk_count, :squiggle_length] = samples[squiggle_start:squiggle_end]
            chunk_lengths[chunk_count] = squiggle_length

            # index alphabet from 1 (ctc blank labels - 0)
            targets[chunk_count, :reference_length] = reference[start:end] + 1
            target_lengths[chunk_count] = reference_length

            chunk_count += 1

            if chunk_count == args.chunks:
                break

        if chunk_count == args.chunks:
            break

    skipped = chunk_idx - chunk_count
    percent = (skipped / chunk_idx * 100) if skipped else 0

    print("Processed %s reads of out %s [%.2f%%]" % (read_idx, total_reads, read_idx / total_reads * 100))
    print("Skipped %s chunks out of %s due to bad chunks [%.2f%%].\n" % (skipped, chunk_idx, percent))
    print("Reason for skipping:")
    print("  - off the end (signal)          ", off_the_end_sig)
    print("  - off the end (sequence)        ", off_the_end_ref)
    print("  - homopolymer chunk boundary    ", homopolymer_boundary)
    print("  - longest run too short         ", min_run_count)
    print("  - minimum number of bases       ", min_bases)
    print("  - maximum number of bases       ", max_bases)

    if chunk_count < args.chunks:
        chunks = np.delete(chunks, np.s_[chunk_count:], axis=0)
        chunk_lengths = chunk_lengths[:chunk_count]
        targets = np.delete(targets, np.s_[chunk_count:], axis=0)
        target_lengths = target_lengths[:chunk_count]

    if args.chunks > args.validation_chunks:
        split = args.validation_chunks
        vdir = os.path.join(args.output_directory, "validation")
        os.makedirs(vdir, exist_ok=True)
        np.save(os.path.join(vdir, "chunks.npy"), chunks[:split])
        np.save(os.path.join(vdir, "chunk_lengths.npy"), chunk_lengths[:split])
        np.save(os.path.join(vdir, "references.npy"), targets[:split])
        np.save(os.path.join(vdir, "reference_lengths.npy"), target_lengths[:split])
    else:
        split = 0

    np.save(os.path.join(args.output_directory, "chunks.npy"), chunks[split:])
    np.save(os.path.join(args.output_directory, "chunk_lengths.npy"), chunk_lengths[split:])
    np.save(os.path.join(args.output_directory, "references.npy"), targets[split:])
    np.save(os.path.join(args.output_directory, "reference_lengths.npy"), target_lengths[split:])

    print()
    print("Training data written to %s:" % args.output_directory)
    print("  - chunks.npy with shape", chunks[split:].shape)
    print("  - chunk_lengths.npy with shape", chunk_lengths[split:].shape)
    print("  - references.npy with shape", targets[split:].shape)
    print("  - reference_lengths.npy shape", target_lengths[split:].shape)


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("chunkify_file")
    parser.add_argument("output_directory")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--chunks", default=10000000, type=int)
    parser.add_argument("--validation-chunks", default=1000, type=int)
    parser.add_argument("--min-run", default=5, type=int)
    parser.add_argument("--min-seq-len", default=200, type=int)
    parser.add_argument("--max-seq-len", default=400, type=int)
    parser.add_argument("--min-samples-per-base", default=8, type=int)
    parser.add_argument("--max-samples-per-base", default=12, type=int)
    return parser
