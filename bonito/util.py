import os
from itertools import groupby

import torch
import random
import numpy as np
from Bio import pairwise2


def init(seed):
    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert(torch.cuda.is_available())


# FIX this
lk = {0: 'N', 1: 'A', 2: 'C', 3: 'G', 4: 'T'}


def stitch(target):
    return ''.join(lk[t] for t in target if t)


def decode_ctc(predictions, p=0.0):
    path = np.argmax(predictions, axis=1)
    return ''.join([lk[b] for b, g in groupby(path) if b])


def load_data(basepth, chunksize=2000, shuffle=True, limit=None):
    basepth = os.path.join(basepth, '..', 'chunks', str(chunksize))
    chunks = np.load(os.path.join(basepth, "chunks.npy"))
    targets = np.load(os.path.join(basepth, "references.npy"))
    target_lengths = np.load(os.path.join(basepth, "reference_lengths.npy"))
    
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


def load_model(dirname, device, weights=0):
    outdir = '/data/training/models/'
    workdir = os.path.join(outdir, dirname)
    weights = os.path.join(workdir, 'weights_%s.tar' % weights if weights else 'weights.tar')
    modelfile = os.path.join(workdir, 'model.py')
    device = torch.device(device)
    model = torch.load(modelfile, map_location=device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    return model


def identity(seq1, seq2, score_only=True, trim_end_gaps=True, alignment=False):

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
    return identity(seq1, seq2, score_only=False)
