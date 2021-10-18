#!/usr/bin/env python

"""
Convert a Taiyaki chunkify training file to set of Bonito CTC .npy files
"""

import os
import h5py
import random
import numpy as np
from argparse import ArgumentParser
from collections import OrderedDict
from itertools import islice as take
from argparse import ArgumentDefaultsHelpFormatter

from tqdm import tqdm
from bonito.data import ChunkDataSet


def align(samples, pointers, reference):
    """ align to the start of the mapping """
    squiggle_duration = len(samples)
    mapped_off_the_start = len(pointers[pointers < 0])
    mapped_off_the_end = len(pointers[pointers >= squiggle_duration])
    pointers = pointers[mapped_off_the_start:len(pointers) - mapped_off_the_end]
    reference = reference[mapped_off_the_start:len(reference) - mapped_off_the_end]
    return samples[pointers[0]:pointers[-1]], pointers - pointers[0], reference


def scale(read, normalise=True):
    """ scale and normalise a read """
    samples = read['Dacs'][:]
    scaling = read.attrs['range'] / read.attrs['digitisation']
    scaled = (scaling * (samples + read.attrs['offset'])).astype(np.float32)
    if normalise:
        return (scaled - read.attrs['shift_frompA']) / read.attrs['scale_frompA']
    return scaled


def pad_lengths(ragged_array, max_len=None):
    lengths = np.array([len(x) for x in ragged_array], dtype=np.uint16)
    padded = np.zeros((len(ragged_array), max_len or np.max(lengths)), dtype=ragged_array[0].dtype)
    for x, y in zip(ragged_array, padded):
        y[:len(x)] = x
    return padded, lengths


def regular_break_points(n, chunk_len, overlap=0, align='mid'):
    num_chunks, remainder = divmod(n - overlap, chunk_len - overlap)
    start = {'left': 0, 'mid': remainder // 2, 'right': remainder}[align]
    starts = np.arange(start, start + num_chunks*(chunk_len - overlap), (chunk_len - overlap))
    return np.vstack([starts, starts + chunk_len]).T


def get_chunks(read, break_points):
    sample = scale(read)
    pointers = read['Ref_to_signal'][:]
    target = read['Reference'][:] + 1  # CTC convention
    return (
        (sample[i:j], target[ti:tj]) for (i, j), (ti, tj)
        in zip(break_points, np.searchsorted(pointers, break_points))
    )


def chunk_dataset(reads, chunk_len, num_chunks=None):
    all_chunks = (
        (chunk, target) for read in reads for chunk, target in
        get_chunks(reads[read], regular_break_points(len(reads[read]['Dacs']), chunk_len))
    )
    chunks, targets = zip(*tqdm(take(all_chunks, num_chunks), total=num_chunks))
    targets, target_lens = pad_lengths(targets) # convert refs from ragged arrray
    return ChunkDataSet(chunks, targets, target_lens)


def validation_split(reads, num_valid=1000):
    reads = np.random.permutation(sorted(reads.items()))
    return OrderedDict(reads[:-num_valid]), OrderedDict(reads[-num_valid:])


def typical_indices(x, n=2.5):
    mu, sd = np.mean(x), np.std(x)
    idx, = np.where((mu - n*sd < x) & (x < mu + n*sd))
    return idx


def filter_chunks(ds, idx):
    filtered = ChunkDataSet(ds.chunks.squeeze(1)[idx], ds.targets[idx], ds.lengths[idx])
    filtered.targets = filtered.targets[:, :filtered.lengths.max()]
    return filtered


def save_chunks(chunks, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    np.save(os.path.join(output_directory, "chunks.npy"), chunks.chunks.squeeze(1))
    np.save(os.path.join(output_directory, "references.npy"), chunks.targets)
    np.save(os.path.join(output_directory, "reference_lengths.npy"), chunks.lengths)
    print()
    print("> data written to %s:" % output_directory)
    print("  - chunks.npy with shape", chunks.chunks.squeeze(1).shape)
    print("  - references.npy with shape", chunks.targets.shape)
    print("  - reference_lengths.npy shape", chunks.lengths.shape)


def main(args):

    random.seed(args.seed)
    np.random.seed(args.seed)

    reads = h5py.File(args.chunkify_file, 'r')['Reads']
    training, validation = validation_split(reads, args.validation_reads)

    print("> preparing training chunks\n")
    training_chunks = chunk_dataset(training, args.chunksize)
    training_indices = typical_indices(training_chunks.lengths)
    training_chunks = filter_chunks(training_chunks, np.random.permutation(training_indices))
    save_chunks(training_chunks, args.output_directory)

    print("\n> preparing validation chunks\n")
    validation_chunks = chunk_dataset(validation, args.chunksize)
    validation_indices = typical_indices(validation_chunks.lengths)
    validation_chunks = filter_chunks(validation_chunks, validation_indices)
    save_chunks(validation_chunks, os.path.join(args.output_directory, "validation"))


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("chunkify_file")
    parser.add_argument("output_directory")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--chunksize", default=3600, type=int)
    parser.add_argument("--validation-reads", default=1000, type=int)
    return parser
