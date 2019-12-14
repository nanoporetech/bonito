"""
Bonito utils
"""

import re
import os
from glob import glob
from itertools import groupby
from collections import defaultdict

from bonito.model import Model

import toml
import torch
import random
import parasail
import numpy as np

try:
    from claragenomics.bindings import cuda
    from claragenomics.bindings.cudapoa import CudaPoaBatch
except ImportError:
    pass


split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")


def init(seed, device):
    """
    Initialise random libs and setup cudnn

    https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cpu": return
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert(torch.cuda.is_available())


def decode_ref(encoded, labels):
    """
    Convert a integer encoded reference into a string and remove blanks
    """
    return ''.join(labels[e] for e in encoded if e)


def decode_ctc(predictions, labels):
    """
    Argmax decoder with collapsing repeats
    """
    path = np.argmax(predictions, axis=1)
    return ''.join([labels[b] for b, g in groupby(path) if b])


def load_data(shuffle=False, limit=None, directory=None):
    """
    Load the training data
    """
    if directory is None:
        directory = os.path.join(os.path.dirname(__file__), "data")

    chunks = np.load(os.path.join(directory, "chunks.npy"), mmap_mode='r')
    chunk_lengths = np.load(os.path.join(directory, "chunk_lengths.npy"), mmap_mode='r')
    targets = np.load(os.path.join(directory, "references.npy"), mmap_mode='r')
    target_lengths = np.load(os.path.join(directory, "reference_lengths.npy"), mmap_mode='r')

    if limit:
        chunks = chunks[:limit]
        chunk_lengths = chunk_lengths[:limit]
        targets = targets[:limit]
        target_lengths = target_lengths[:limit]

    if shuffle:
        shuf = np.random.permutation(chunks.shape[0])
        chunks = chunks[shuf]
        chunk_lengths = chunk_lengths[shuf]
        targets = targets[shuf]
        target_lengths = target_lengths[shuf]

    return chunks, chunk_lengths, targets, target_lengths


def load_model(dirname, device, weights=None):
    """
    Load a model from disk
    """
    if not weights: # take the latest checkpoint
        weight_files = glob(os.path.join(dirname, "weights_*.tar"))
        weights = max([int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files])

    device = torch.device(device)
    config = os.path.join(dirname, 'config.toml')
    weights = os.path.join(dirname, 'weights_%s.tar' % weights)
    model = Model(toml.load(config))
    model.to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    return model


def parasail_to_sam(result, seq):
    """
    Extract reference start and sam compatible cigar string.

    :param result: parasail alignment result.
    :param seq: query sequence.

    :returns: reference start coordinate, cigar string.
    """
    cigstr = result.cigar.decode.decode()
    first = re.search(split_cigar, cigstr)

    first_count, first_op = first.groups()
    prefix = first.group()
    rstart = result.cigar.beg_ref
    cliplen = result.cigar.beg_query

    clip = '' if cliplen == 0 else '{}S'.format(cliplen)
    if first_op == 'I':
        pre = '{}S'.format(int(first_count) + cliplen)
    elif first_op == 'D':
        pre = clip
        rstart = int(first_count)
    else:
        pre = '{}{}'.format(clip, prefix)

    mid = cigstr[len(prefix):]
    end_clip = len(seq) - result.end_query - 1
    suf = '{}S'.format(end_clip) if end_clip > 0 else ''
    new_cigstr = ''.join((pre, mid, suf))
    return rstart, new_cigstr


def accuracy(ref, seq, balanced=False):
    """
    Calculate the accuracy between `ref` and `seq`
    """
    alignment = parasail.sw_trace_striped_32(ref, seq, 8, 4, parasail.dnafull)
    counts = defaultdict(int)
    _, cigar = parasail_to_sam(alignment, seq)

    for count, op  in re.findall(split_cigar, cigar):
        counts[op] += int(count)

    if balanced:
        accuracy = (counts['='] - counts['I']) / (counts['='] + counts['X'] + counts['D'])
    else:
        accuracy = counts['='] / (counts['='] + counts['I'] + counts['X'] + counts['D'])
    return accuracy * 100


def print_alignment(ref, seq):
    """
    Print the alignment between `ref` and `seq`
    """
    alignment = parasail.sw_trace_striped_32(ref, seq, 8, 4, parasail.dnafull)
    print(alignment.traceback.query)
    print(alignment.traceback.comp)
    print(alignment.traceback.ref)
    print("  Score=%s" % alignment.score)
    return alignment.score


def poa(groups, max_sequences_per_poa=100, gpu_mem_per_batch=0.9):
    """
    Generate consensus for POA groups.

    Args:
        groups : A list of lists of sequences for which consensus is to be generated.
    """
    free, total = cuda.cuda_get_mem_info(cuda.cuda_get_device())
    gpu_mem_per_batch *= free
    batch = CudaPoaBatch(max_sequences_per_poa, gpu_mem_per_batch, stream=None, output_type="consensus")
    results = []

    for i, group in enumerate(groups, start=1):
        group_status, seq_status = batch.add_poa_group(group)

        # Once batch is full, run POA processing
        if group_status == 1 or i == len(groups):
            batch.generate_poa()

            consensus, coverage, status = batch.get_consensus()
            results.extend(consensus)

            batch.reset()
            group_status, seq_status = batch.add_poa_group(group)

    return results
