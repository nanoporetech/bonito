"""
Bonito basecall
"""

import torch
import numpy as np
from functools import partial

from bonito.multiprocessing import process_map
from bonito.util import mean_qscore_from_qstring
from bonito.util import chunk, stitch, batchify, unbatchify, permute


def basecall(model, reads, beamsize=5, chunksize=0, overlap=0, batchsize=1, qscores=False, reverse=None):
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
        (read, {'scores': stitch(v, chunksize, overlap, len(read.signal), model.stride)}) for read, v in scores
    )
    decoder = partial(decode, decode=model.decode, beamsize=beamsize, qscores=qscores, stride=model.stride)
    basecalls = process_map(decoder, scores, n_proc=4)
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


def decode(scores, decode, beamsize=5, qscores=False, stride=1):
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
    sig_move = None
    if path is not None:
        sig_move = np.full(path.size * stride, False)
        sig_move[np.where(path)[0] * stride] = True
    return {'sequence': seq, 'qstring': qstring, 'mean_qscore': mean_qscore, 'path': path, 'sig_move': sig_move}
