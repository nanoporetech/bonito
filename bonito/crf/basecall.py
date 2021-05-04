"""
Bonito CRF basecall
"""

import torch
import numpy as np
from itertools import groupby
from functools import partial
from operator import itemgetter
from fast_ctc_decode import crf_beam_search

import bonito
from bonito.io import Writer
from bonito.fast5 import get_reads
from bonito.aligner import align_map
from bonito.multiprocessing import thread_map, thread_iter
from bonito.util import concat, chunk, batchify, unbatchify, half_supported


def stitch(chunks, chunksize, overlap, length, stride, reverse=False):
    """
    Stitch chunks together with a given overlap
    """
    if isinstance(chunks, dict):
        return {
            k: stitch(v, chunksize, overlap, length, stride, reverse=reverse)
            for k, v in chunks.items()
        }
    return bonito.util.stitch(chunks, chunksize, overlap, length, stride, reverse=reverse)


def compute_scores(model, batch, reverse=False):
    """
    Compute scores for model.
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        dtype = torch.float16 if half_supported() else torch.float32
        scores = model(batch.to(dtype).to(device))
        if reverse: scores = model.seqdist.reverse_complement(scores)
        betas = model.seqdist.backward_scores(scores.to(torch.float32))
        trans, init = model.seqdist.compute_transition_probs(scores, betas)
    return {
        'trans': trans.to(dtype).transpose(0, 1),
        'init': init.to(dtype).unsqueeze(1),
    }


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


def decode(scores, beam_size=10, beam_cut_threshold=0.1, alphabet="NACGT"):
    """
    Beam search over the transition posterior probs.
    """
    return crf_beam_search(
        scores['trans'].astype(np.float32),
        scores['init'][0].astype(np.float32),
        alphabet, beam_size=beam_size,
        beam_cut_threshold=beam_cut_threshold,
    )[0]


def split_read(read, split_read_length=400000):
    """
    Split large reads into manageable pieces.
    """
    if len(read.signal) <= split_read_length:
        return [(read, 0, len(read.signal))]
    breaks = np.arange(0, len(read.signal) + split_read_length, split_read_length)
    return [(read, start, min(end, len(read.signal))) for (start, end) in zip(breaks[:-1], breaks[1:])]


def basecall(model, reads, aligner=None, beamsize=40, chunksize=4000, overlap=500, batchsize=32, qscores=False, reverse=False):
    """
    Basecalls a set of reads.
    """
    reads = (read_chunk for read in reads for read_chunk in split_read(read)[::-1 if reverse else 1])
    chunks = (
        ((read, start, end), chunk(torch.from_numpy(read.signal[start:end]), chunksize, overlap))
        for (read, start, end) in reads
    )
    batches = (
        (k, compute_scores(model, batch, reverse=reverse))
        for k, batch in thread_iter(batchify(chunks, batchsize=batchsize))
    )
    stitched = (
        (read, stitch(x, chunksize, overlap, end - start, model.stride, reverse=reverse))
        for ((read, start, end), x) in unbatchify(batches)
    )

    transferred = thread_map(transfer, stitched, n_thread=1)
    basecalls = thread_map(decode, transferred, n_thread=8)

    basecalls = (
        (read, ''.join(seq for k, seq in parts))
        for read, parts in groupby(basecalls, lambda x: (x[0].parent if hasattr(x[0], 'parent') else x[0]))
    )
    basecalls = (
        (read, {'sequence': seq, 'qstring': '?' * len(seq) if qscores else '*', 'mean_qscore': 0.0})
        for read, seq in basecalls
    )

    if aligner: return align_map(aligner, basecalls)
    return basecalls
