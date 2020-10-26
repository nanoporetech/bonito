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
from bonito.util import concat, chunk, batchify, unbatchify, half_supported


def stitch(chunks, start, end):
    """
    Stitch chunks together with a given overlap
    """
    if isinstance(chunks, dict):
        return {k: stitch(v, start, end) for k, v in chunks.items()}

    if chunks.shape[0] == 1: return chunks.squeeze(0)
    return concat([chunks[0, :end], *chunks[1:-1, start:end], chunks[-1, start:]])


def compute_scores(model, batch):
    """
    Compute scores for model.
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        dtype = torch.float16 if half_supported() else torch.float32
        scores = model.encoder(batch.to(dtype).to(device))
        betas = model.seqdist.backward_scores(scores.to(torch.float32))
        betas -= (betas.max(2, keepdim=True)[0] - 5.0)
    return {
        'scores': scores.transpose(0, 1),
        'betas': betas.transpose(0, 1),
    }


def transfer_int8(x, pinned_scores, pinned_betas, scale=127/5):
    scores = x['scores']
    scores *= scale
    scores = torch.round(scores).to(torch.int8).detach()
    betas = x['betas']
    betas *= scale
    betas = torch.round(torch.clamp(betas, -127., 128.)).to(torch.int8).detach()
    if betas.shape[0] != pinned_betas.shape[0]:
        pinned_betas.resize_as_(betas)
        pinned_scores.resize_as_(scores)
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
    Basecalls a set of reads.
    """
    _decode = partial(
        decode_int8, seqdist=model.seqdist, beamsize=beamsize
    )
    _stitch = partial(
        stitch,
        start=overlap // 2 // model.stride,
        end=(chunksize - overlap // 2) // model.stride,
    )
    pinned_scores = torch.empty(
        (batchsize, chunksize // model.stride, model.seqdist.n_score()),
        pin_memory=True, dtype=torch.int8
    )
    pinned_betas = torch.empty(
        (batchsize, chunksize // model.stride + 1, model.seqdist.n_score() // len(model.alphabet)),
        pin_memory=True, dtype=torch.int8
    )
    chunks = (
        ((read, chunk(torch.from_numpy(read.signal), chunksize, overlap, pad_start=True)) for read in reads)
    )
    batches = (
        (read, compute_scores(model, batch))
        for read, batch in batchify(chunks, batchsize=batchsize)
    )
    scores = (
        (read, transfer_int8(batch, pinned_scores, pinned_betas)) for read, batch in batches
    )
    scores = thread_map(_stitch, unbatchify(scores), n_thread=4)
    basecalls = thread_map(_decode, scores, n_thread=4)
    if aligner: return align_map(aligner, basecalls)
    return basecalls


def ctc_data(model, reads, aligner, chunksize=4000, overlap=500, min_accuracy=0.9, min_coverage=0.9):
    """
    Convert reads into a format suitable for ctc training.
    """
    raise NotImplemented
