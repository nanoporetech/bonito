"""
Bonito CRF basecalling
"""

import torch
from koi.decode import beam_search, to_str

from bonito.multiprocessing import thread_map, thread_iter
from bonito.util import chunk, batchify, unbatchify, half_supported, stitch


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
    with torch.inference_mode():
        device = next(model.parameters()).device
        dtype = torch.float16 if half_supported() else torch.float32
        scores = model(batch.to(dtype).to(device))
        if reverse: scores = model.seqdist.reverse_complement(scores)
        sequence, qstring, moves = beam_search(scores, beam_width=32)
        return {
            'qstring': qstring,
            'sequence': sequence,
        }


def basecall(model, reads, chunksize=4000, overlap=100, batchsize=32, reverse=False):
    """
    Basecalls a set of reads.
    """
    chunks = thread_iter(
        ((read, 0, len(read.signal)), chunk(torch.from_numpy(read.signal), chunksize, overlap))
        for read in reads
    )

    batches = thread_iter(batchify(chunks, batchsize=batchsize))

    scores = thread_iter(
        (read, compute_scores(model, batch, reverse=reverse)) for read, batch in batches
    )

    stitched = thread_iter(
        (read, stitch_scores(scores, end - start, chunksize, overlap, model.stride, reverse))
        for ((read, start, end), scores) in unbatchify(scores)
    )

    basecalls = thread_iter(
        (read, {k: to_str(v) for k, v in attrs.items()}) for read, attrs in stitched
    )

    return basecalls
