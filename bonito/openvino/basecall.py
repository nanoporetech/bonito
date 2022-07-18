import torch
from crf_beam import beam_search
from bonito.crf.basecall import stitch_results
from bonito.multiprocessing import thread_iter, thread_map
from bonito.util import chunk, stitch, batchify, unbatchify


def compute_scores(model, batch):
    scores = model(batch)
    fwd = model.seqdist.forward_scores(scores)
    bwd = model.seqdist.backward_scores(scores)
    posts = torch.softmax(fwd + bwd, dim=-1)
    return {
        'scores': scores.transpose(0, 1),
        'bwd': bwd.transpose(0, 1),
        'posts': posts.transpose(0, 1),
    }


def decode(x, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0):
    sequence, qstring, moves = beam_search(x['scores'], x['bwd'], x['posts'])
    return {
        'sequence': sequence,
        'qstring': qstring,
        'moves': moves,
    }


def basecall(model, reads, chunksize=4000, overlap=100, batchsize=32, reverse=False):

    chunks = thread_iter(
        ((read, 0, len(read.signal)), chunk(torch.from_numpy(read.signal), chunksize, overlap))
        for read in reads
    )

    batches = thread_iter(batchify(chunks, batchsize=batchsize))

    scores = thread_iter(
        (read, compute_scores(model, batch)) for read, batch in batches
    )

    results = thread_iter(
        (read, stitch_results(scores, end - start, chunksize, overlap, model.stride))
        for ((read, start, end), scores) in unbatchify(scores)
    )

    return thread_map(decode, results, n_thread=48)
