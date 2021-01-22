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
from bonito.multiprocessing import thread_map, thread_iter
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


def quantise_int8(x, scale=127/5):
    """
    Quantise scores to int8.
    """
    scores = x['scores']
    scores *= scale
    scores = torch.round(scores).to(torch.int8).detach()
    betas = x['betas']
    betas *= scale
    betas = torch.round(torch.clamp(betas, -127., 128.)).to(torch.int8).detach()
    return {'scores': scores, 'betas': betas}


def transfer(x):
    """
    Device to host transfer using pinned memory.
    """
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        return {
            k: torch.empty(v.shape, pin_memory=True, dtype=v.dtype).copy_(v).numpy()
            for k, v in x.items()
        }


def decode_int8(scores, seqdist, scale=127/5, beamsize=40, beamcut=100.0):
    """
    Beamsearch decode.
    """
    path, _ = beamsearch(
        scores['scores'], scale, seqdist.n_base, beamsize,
        guide=scores['betas'], beam_cut=beamcut
    )
    try:
        return seqdist.path_to_str(path % 4 + 1)
    except IndexError:
        return ""


def basecall(model, reads, aligner=None, beamsize=40, chunksize=4000, overlap=500, batchsize=32, qscores=False):
    """
    Basecalls a set of reads.
    """
    split_read_length=400000
    _stitch = partial(
        stitch,
        start=overlap // 2 // model.stride,
        end=(chunksize - overlap // 2) // model.stride,
    )
    _decode = partial(decode_int8, seqdist=model.seqdist, beamsize=beamsize)
    reads = (
        ((read, i), x) for read in reads
        for (i, x) in enumerate(torch.split(torch.from_numpy(read.signal), split_read_length))
    )
    chunks = (
        ((read, chunk(signal, chunksize, overlap, pad_start=True)) for (read, signal) in reads)
    )
    batches = (
        (read, quantise_int8(compute_scores(model, batch)))
        for read, batch in thread_iter(batchify(chunks, batchsize=batchsize))
    )
    stitched = ((read, _stitch(x)) for (read, x) in unbatchify(batches))
    transferred = thread_map(transfer, stitched, n_thread=1)
    basecalls = thread_map(_decode, transferred, n_thread=8)

    basecalls = (
        (read, ''.join(seq for k, seq in parts)) for read, parts in groupby(basecalls, lambda x: x[0][0])
    )
    basecalls = (
        (read, {'sequence': seq, 'qstring': '?' * len(seq) if qscores else '*', 'mean_qscore': 0.0})
        for read, seq in basecalls
    )

    if aligner: return align_map(aligner, basecalls)
    return basecalls
