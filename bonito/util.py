"""
Bonito utils
"""

import re
import os
from glob import glob
from itertools import groupby
from collections import defaultdict

import torch
import random
import parasail
import numpy as np


__dir__ = os.path.dirname(__file__)
labels = ['N', 'A', 'C', 'G', 'T']


def init(seed):
    """
    Initialise random libs and setup cudnn

    https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert(torch.cuda.is_available())


def decode_ref(encoded):
    """
    Convert a integer encoded reference into a string and remove blanks
    """
    return ''.join(labels[e] for e in encoded if e)


def decode_ctc(predictions, p=0.0):
    """
    Argmax decoder with collapsing repeats
    """
    path = np.argmax(predictions, axis=1)
    return ''.join([labels[b] for b, g in groupby(path) if b])


def load_data(shuffle=False, limit=None):
    """
    Load the training data
    """
    chunks = np.load(os.path.join(__dir__, "data", "chunks.npy"))
    targets = np.load(os.path.join(__dir__, "data", "references.npy"))
    target_lengths = np.load(os.path.join(__dir__, "data", "reference_lengths.npy"))

    if limit:
        chunks = chunks[:limit]
        targets = targets[:limit]
        target_lengths = target_lengths[:limit]

    if shuffle:
        shuf = np.random.permutation(chunks.shape[0])
        chunks = chunks[shuf]
        targets = targets[shuf]
        target_lengths = target_lengths[shuf]

    return chunks, targets, target_lengths


def load_model(dirname, device, weights=None):
    """
    Load a model from disk
    """
    if not weights: # take the latest checkpoint
        weight_files = glob(os.path.join(dirname, "weights_*.tar"))
        weights = max([int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files])

    weights = os.path.join(dirname, 'weights_%s.tar' % weights)
    modelfile = os.path.join(dirname, 'model.py')
    device = torch.device(device)
    model = torch.load(modelfile, map_location=device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    return model


def accuracy(seq1, seq2):
    """
    Calculate the balanced accuracy between `seq` and `seq2`
    """
    alignment = parasail.sg_trace_scan_16(seq1, seq2, 10, 1, parasail.blosum62)

    counts = defaultdict(int)
    cigar = alignment.cigar.decode.decode()

    for c in re.findall("[0-9]+[=XID]", cigar):
        counts[c[-1]] += int(c[:-1])

    accuracy = (counts['='] - counts['I']) / (counts['='] + counts['M'] + counts['D'])
    return accuracy * 100


def print_alignment(seq1, seq2):
    """
    Print the alignment between `seq1` and `seq2`
    """
    alignment = parasail.sg_trace(seq1, seq2, 10, 1, parasail.blosum62)
    print(alignment.traceback.query)
    print(alignment.traceback.comp)
    print(alignment.traceback.ref)
    print("  Score=%s" % alignment.score)
    return alignment.score
