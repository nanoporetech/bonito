"""
Bonito utils
"""

import re
import os
from glob import glob
from itertools import groupby

import torch
import random
import numpy as np
from Bio import pairwise2

__dir__ = os.path.dirname(__file__)
labels = {0: 'N', 1: 'A', 2: 'C', 3: 'G', 4: 'T'}


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


def stitch(encoded):
    """
    Convert a integer encode reference into a string and remove blanks
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
        weights = max([int(re.sub("[^0-9]+", "", w)) for w in weight_files])
        print("loaded cp", weights)
    weights = os.path.join(dirname, 'weights_%s.tar' % weights)
    modelfile = os.path.join(dirname, 'model.py')
    device = torch.device(device)
    model = torch.load(modelfile, map_location=device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    return model


def identity(seq1, seq2, score_only=True, trim_end_gaps=True, alignment=False):
    """
    Align two sequences
    """
    if trim_end_gaps:
        needle = pairwise2.align.globalms(seq1, seq2, 5, -4, -10, -0.5, penalize_end_gaps=False, one_alignment_only=True)

        aref = needle[0][0]
        aseq = needle[0][1]

        i = 0
        j = len(aref)

        # do we have gaps at the start?
        if aref.startswith('-'):
            i += len(aref) - len(aref.lstrip('-'))
        elif aseq.startswith('-'):
            i += len(aseq) - len(aseq.lstrip('-'))

        # do we have gaps at the end?
        if aref.endswith('-'):
            j -= len(aref) - len(aref.rstrip('-'))
        elif aseq.endswith('-'):
            j -= len(aseq) - len(aseq.rstrip('-'))

        seq1 = aref[i:j]
        seq2 = aseq[i:j]

    identical_matches = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True, score_only=True)
    needle = pairwise2.align.globalms(seq1, seq2, 5, -4, -10, -0.5, penalize_end_gaps=False, one_alignment_only=True)

    if not score_only: print(pairwise2.format_alignment(*needle[0]))
    if alignment: return pairwise2.format_alignment(*needle[0])
    return (identical_matches / needle[0][4]) * 100


def palign(seq1, seq2):
    """
    Print the alignment between `seq1` and `seq2`
    """
    return identity(seq1, seq2, score_only=False)
