"""
Bonito CRF basecall
"""

import torch
import numpy as np
from crf_beam import beam_search

from itertools import groupby
from functools import partial

import bonito
from bonito.multiprocessing import thread_starmap, thread_map, thread_iter
from bonito.util import concat, chunk, batchify, unbatchify, half_supported, stitch


def stitch_scores(scores, length, size, overlap, stride, reverse=False):
    """
    Stitch scores together with a given overlap.
    """
    if isinstance(scores, dict):
        return {
            k: stitch_scores(v, length, size, overlap, stride, reverse=reverse)
            for k, v in scores.items()
        }
    return stitch(scores, size, overlap, length, stride, reverse=reverse)


def compute_scores(model, batch, reverse=False):
    """
    Compute scores for model.
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        dtype = torch.float16 if half_supported() else torch.float32
        scores = model(batch.to(dtype).to(device))
        if reverse: scores = model.seqdist.reverse_complement(scores)
        scores = scores.to(torch.float32)
        betas = model.seqdist.backward_scores(scores)
        fwd = model.seqdist.forward_scores(scores)
        posts = torch.softmax(fwd + betas, dim=-1)
    return {
        'scores': scores.transpose(0, 1),
        'betas': betas.transpose(0, 1),
        'posts': posts.transpose(0, 1),
    }


def quantise_int8(x, scale=127/5):
    """
    Quantise scores to int8.
    """
    scores = x['scores'] * scale
    betas = x['betas']
    betas -= betas.max(2, keepdim=True)[0] - 5.0
    betas *= scale
    posts = x['posts'] * 255 - 128
    return {
        'scores': torch.round(scores).to(torch.int8).detach(),
        'betas': torch.round(torch.clamp(betas, -127., 128.)).to(torch.int8).detach(),
        'posts': torch.round(posts).to(torch.int8).detach(),
    }


def transfer(x):
    """
    Device to host transfer using pinned memory.
    """
    torch.cuda.synchronize()

    with torch.cuda.stream(torch.cuda.Stream()):
        return {
            k: torch.empty(v.shape, pin_memory=True, dtype=v.dtype).copy_(v)
            for k, v in x.items()
        }


def decode(model, scores, beam_size=40):
    """
    Decode sequence and qstring from model scores.
    """
    try:
        qshift = model.config['qscore']['bias']
        qscale = model.config['qscore']['scale']
    except:
        qshift = 0.0
        qscale = 1.0
    sequence, qstring, moves = beam_search(
        scores['scores'], scores['betas'], scores['posts'],
        beam_size=beam_size, q_shift=qshift, q_scale=qscale,
        temperature=127/5,
    )
    return {'sequence': sequence, 'qstring': qstring}


def split_read(read, max_samples=400000):
    """
    Split large reads into manageable pieces.
    """
    if len(read.signal) <= max_samples:
        return [(read, 0, len(read.signal))]
    breaks = np.arange(0, len(read.signal) + max_samples, max_samples)
    return [
        (read, start, min(end, len(read.signal)))
        for start, end in zip(breaks[:-1], breaks[1:])
    ]


def basecall(model, reads, chunksize=4000, overlap=500, batchsize=32, reverse=False):
    """
    Basecalls a set of reads.
    """
    beam_size = 5 if model.seqdist.state_len < 5 else 40
    stitch_on_gpu = model.config['encoder']['features'] >= 768

    reads = (
        read_chunk for read in reads for read_chunk
        in split_read(read)[::-1 if reverse else 1]
    )

    chunks = (
        ((read, start, end), chunk(torch.from_numpy(read.signal[start:end]), chunksize, overlap))
        for (read, start, end) in reads
    )

    with torch.cuda.stream(torch.cuda.Stream()):

        scores = (
            (read, quantise_int8(compute_scores(model, batch, reverse=reverse)))
            for read, batch in thread_iter(batchify(chunks, batchsize=batchsize))
        )

        if stitch_on_gpu:
            unbatched = thread_iter(
                (read, (scores, end - start, chunksize, overlap, model.stride, reverse))
                for ((read, start, end), scores) in unbatchify(scores)
            )
            scores = thread_starmap(stitch_scores, unbatched, n_thread=1)

        scores = thread_map(transfer, scores, n_thread=1)

        if not stitch_on_gpu:
            unbatched = thread_iter(
                (read, (scores, end - start, chunksize, overlap, model.stride, reverse))
                for ((read, start, end), scores) in thread_iter(unbatchify(scores))
            )
            scores = thread_starmap(stitch_scores, unbatched, n_thread=1)

    basecalls = thread_map(partial(decode, model, beam_size=beam_size), scores, n_thread=12)

    return (
        (read, concat([v for k, v in parts])) for read, parts in
        groupby(basecalls, lambda x: (x[0].parent if hasattr(x[0], 'parent') else x[0]))
    )
