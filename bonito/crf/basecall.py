"""
Bonito CRF basecall
"""

import sys
import torch
import numpy as np
from tqdm import tqdm
from time import perf_counter
from datetime import timedelta

from bonito.io import Writer
from bonito.fast5 import get_reads
from bonito.aligner import Aligner, align_map
from bonito.util import batchify, unbatchify, chunk, stitch


def basecall(model, reads, aligner=None, beamsize=1, chunksize=4000, overlap=500, batchsize=64):
    """
    Basecalls at set of reads.
    """
    chunks = (
        (read, chunk(torch.tensor(read.signal), chunksize, overlap, pad_start=True))
        for read in reads
    )
    tracebacks = unbatchify(
        (k, compute_scores(model, v)) for k, v in batchify(chunks, batchsize=batchsize)
    )
    tracebacks = (
        (read, {'traceback': stitch(v, overlap, model.stride)}) for read, v in tracebacks
    )
    basecalls = (
        (read, decode(model.global_norm.seq_dist, trace)) for read, trace in tracebacks
    )
    if aligner: return align_map(aligner, basecalls)
    return basecalls


def compute_scores(model, batch, post=True):
    """
    Compute scores for model.
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        seq_dist = model.global_norm.seq_dist
        scores = model.encoder(batch.to(torch.float16).to(device)).to(torch.float32)
        if post:
            scores = (seq_dist.posteriors(scores) + 1e-8).log()
        tracebacks = seq_dist.viterbi(scores).to(torch.int16).T
    return tracebacks.cpu().numpy()


def decode(seq_dist, traceback):
    """
    Convert the network traceback into a sequence.
    """
    return {
        'sequence': seq_dist.path_to_str(traceback['traceback']),
        'qstring': '*',
        'mean_qscore': 0.0
    }


def ctc_data(model, reads, aligner, chunksize=3600, overlap=900, min_accuracy=0.9, min_coverage=0.9):
    """
    Convert reads into a format suitable for ctc training.
    """    
    raise NotImplemented
