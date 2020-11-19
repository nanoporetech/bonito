"""
Bonito basecall
"""

import torch
import numpy as np
from functools import partial
from bonito.fast5 import ReadChunk
from bonito.aligner import align_map
from bonito.multiprocessing import process_map, thread_map
from bonito.util import mean_qscore_from_qstring, half_supported
from bonito.util import chunk, stitch, batchify, unbatchify, permute, concat


def basecall(model, reads, aligner=None, beamsize=5, chunksize=0, overlap=0, batchsize=1, qscores=False):
    """
    Basecalls a set of reads.
    """
    chunks = (
        (read, chunk(torch.tensor(read.signal), chunksize, overlap)) for read in reads
    )
    scores = unbatchify(
        (k, compute_scores(model, v)) for k, v in batchify(chunks, batchsize)
    )
    scores = (
        (read, {'scores': stitch(v, overlap, model.stride)}) for read, v in scores
    )
    decoder = partial(decode, decode=model.decode, beamsize=beamsize, qscores=qscores)
    basecalls = process_map(decoder, scores, n_proc=4)
    if aligner: return align_map(aligner, basecalls)
    return basecalls


def compute_scores(model, batch):
    """
    Compute scores for model.
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        chunks = batch.to(torch.half).to(device)
        probs = permute(model(chunks), 'TNC', 'NTC')
    return probs.cpu().to(torch.float32)


def decode(scores, decode, beamsize=5, qscores=False):
    """
    Convert the network scores into a sequence.
    """
    # do a greedy decode to get a sensible qstring to compute the mean qscore from
    seq, path = decode(scores['scores'], beamsize=1, qscores=True, return_path=True)
    seq, qstring = seq[:len(path)], seq[len(path):]
    mean_qscore = mean_qscore_from_qstring(qstring)

    # beam search will produce a better sequence but doesn't produce a sensible qstring/path
    if not (qscores or beamsize == 1):
        try:
            seq = decode(scores['scores'], beamsize=beamsize)
            path = None
            qstring = '*'
        except:
            pass
    return {'sequence': seq, 'qstring': qstring, 'mean_qscore': mean_qscore, 'path': path}
