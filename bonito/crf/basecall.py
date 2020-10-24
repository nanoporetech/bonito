"""
Bonito CRF basecall
"""

import torch
import numpy as np
from kbeam import beamsearch
from itertools import groupby
from functools import partial
from operator import itemgetter

from bonito.io import Writer
from bonito.fast5 import get_reads
from bonito.aligner import Aligner, align_map
from bonito.multiprocessing import thread_map
from bonito.util import concat, chunk, batchify, unbatchify


def stitch(chunks, start, end):
    """
    Stitch chunks together with a given overlap
    """
    if chunks.shape[0] == 1: return chunks.squeeze(0)
    return concat([chunks[0, :end], *chunks[1:-1, start:end], chunks[-1, start:]])


def compute_scores(model, batch):
    """
    Compute scores for model.
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        scores = model.encoder(batch.to(device).to(torch.float16))
        betas = model.seqdist.backward_scores(scores.to(torch.float32))
        betas -= (betas.max(2, keepdim=True)[0] - 5.0)
    return {
        'scores': scores.to(torch.float16).transpose(0, 1),
        'betas': betas.to(torch.float16).transpose(0, 1),
    }


def transfer_int8(x, scale=127/5):
    scores = x['scores']
    scores *= scale
    scores = torch.round(scores).to(torch.int8).detach()
    betas = x['betas']
    betas *= scale
    betas = torch.round(torch.clamp(betas, -127., 128.)).to(torch.int8).detach()
    pinned_betas = torch.empty(betas.shape, pin_memory=True, dtype=torch.int8)
    pinned_scores = torch.empty(scores.shape, pin_memory=True, dtype=torch.int8)
    pinned_betas.copy_(betas)
    pinned_scores.copy_(scores)
    return {'scores': pinned_scores.numpy(), 'betas': pinned_betas.numpy()}


def decode_int8(scores, seqdist, scale=127/5, beamsize=40, beamcut=100.0):
    path, _ = beamsearch(
        scores['scores'], scale, seqdist.n_base, 40,
        guide=scores['betas'], beam_cut=beamcut
    )
    return {
        'sequence': seqdist.path_to_str(path % 4 + 1),
        'qstring': '*',
        'mean_qscore': 0.0
    }


def basecall(model, reads, aligner=None, beamsize=40, chunksize=4000, overlap=500, batchsize=64):
    """
    Basecalls at set of reads.
    """
    start, end = overlap // 2 //model.stride, (chunksize - overlap // 2) // model.stride

    decode = partial(decode_int8, seqdist=model.seqdist, beamsize=beamsize)

    chunks = (
        ((read, chunk(torch.tensor(read.signal), chunksize, overlap, pad_start=True)) for read in reads)
    )
    scores = (
        (read, compute_scores(model, batch))
        for (read, batch) in batchify(chunks, batchsize=batchsize)
    )
    scores = ((read, transfer_int8(batch)) for read, batch in scores)

    # FIX: unbatching on main threads is 2x perf hit
    scores = unbatchify(scores)

    scores = thread_map(
        lambda x: {k: stitch(v, start, end) for k, v in x.items()},
        scores, n_thread=4
    )

    basecalls = thread_map(decode, scores, n_thread=4)
    if aligner: return align_map(aligner, basecalls)
    return basecalls


def ctc_data(model, reads, aligner, chunksize=4000, overlap=500, min_accuracy=0.9, min_coverage=0.9):
    """
    Convert reads into a format suitable for ctc training.
    """
    raise NotImplemented
