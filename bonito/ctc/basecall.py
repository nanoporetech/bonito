"""
Bonito basecall
"""

import torch
import numpy as np
from mappy import revcomp
from functools import partial
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


def ctc_data(model, reads, aligner, chunksize=3600, overlap=900, min_accuracy=0.9, min_coverage=0.9):
    """
    Convert reads into a format suitable for ctc training.
    """
    scores = ((read, ctc_compute_scores(read, model, chunksize, overlap)) for read in reads)
    decoder = partial(ctc_decoder, model, aligner, min_accuracy=min_accuracy, min_coverage=min_coverage)
    ctc_data = thread_map(decoder, scores, n_thread=1)
    return align_map(aligner, ctc_data)


def ctc_compute_scores(read, model, chunksize=0, overlap=0):
    """
    Compute score for model.
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        dtype = np.float16 if half_supported() else np.float32
        raw_data = torch.tensor(read.signal.astype(dtype))
        chunks = chunk(raw_data, chunksize, overlap)
        logprobs = permute(model(chunks.to(device)), 'TNC', 'NTC').cpu().to(torch.float32)
        scores = stitch(logprobs, overlap, model.stride)[:raw_data.shape[0] // model.stride]

    if len(raw_data) > chunksize:
        ctc_chunks = chunks.numpy().squeeze()
        ctc_scores = logprobs
    else:
        ctc_chunks = None
        ctc_scores = None

    return {'scores': scores, 'ctc_scores': ctc_scores, 'ctc_chunks': ctc_chunks}


def ctc_decoder(model, aligner, scores, min_accuracy=0.9, min_coverage=0.9):
    """
    Get target sequences by aligning ctc chunks.
    """
    chunks = []
    targets = []

    if scores['ctc_chunks'] is None:
        return {'chunks': chunks, 'targets': targets, **decode(scores, model.decode)}

    for chunk, score in zip(scores['ctc_chunks'], scores['ctc_scores']):

        try:
            sequence = model.decode(score)
        except:
            continue

        if not sequence:
            continue

        for mapping in aligner.map(sequence):
            cov = (mapping.q_en - mapping.q_st) / len(sequence)
            acc = mapping.mlen / mapping.blen
            refseq = aligner.seq(mapping.ctg, mapping.r_st, mapping.r_en)
            if 'N' in refseq: continue
            if mapping.strand == -1: refseq = revcomp(refseq)
            break
        else:
            continue

        if acc > min_accuracy and cov > min_coverage:
            chunks.append(chunk.squeeze())
            targets.append([
                int(x) for x in refseq.translate({65: '1', 67: '2', 71: '3', 84: '4'})
            ])

    return {'chunks': chunks, 'targets': targets, **decode(scores, model.decode)}
