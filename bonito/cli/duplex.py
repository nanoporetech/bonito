"""
Bonito Duplex consensus decoding.

https://www.biorxiv.org/content/10.1101/2020.02.25.956771v1
"""

import os
import sys
import json
from glob import glob
from pathlib import Path
from functools import partial
from time import perf_counter
from datetime import timedelta
from multiprocessing import Pool
from itertools import islice, groupby
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue, Lock, cpu_count
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import spoa
import torch
import parasail
import numpy as np
import pandas as pd
from tqdm import tqdm
from fast_ctc_decode import crf_beam_search, crf_beam_search_duplex

import bonito
from bonito.io import Writer
from bonito.aligner import Aligner, align_map
from bonito.util import half_supported, concat, accuracy
from bonito.crf.basecall import transfer, split_read, stitch
from bonito.fast5 import get_raw_data_for_read, get_read_ids
from bonito.multiprocessing import thread_map, process_map, process_cancel


def get_read(readdir, summary, idx):
    """
    Get a single read from row `idx` in the `summary` dataframe.
    """
    return get_raw_data_for_read(
        (readdir / summary.iloc[idx].filename_fast5, summary.iloc[idx].read_id)
    )


def read_gen(directory, summary, n_proc=1, cancel=None):
    """
    Generate reads from the given `directory` listed in the `summary` dataframe.
    """
    with Pool(n_proc) as pool:
        for read in pool.imap(partial(get_read, Path(directory), summary), range(len(summary))):
            yield read

            if cancel is not None and cancel.is_set():
                return


def build_index(files, n_proc=1):
    """
    Build an index of read ids to filename mappings
    """
    index = {}
    with ProcessPoolExecutor(max_workers=n_proc) as pool:
        for res in tqdm(pool.map(get_read_ids, files), leave=False):
            index.update(res)
    return index


def build_envelope(len1, seq1, path1, len2, seq2, path2, padding=15):

    # needleman-wunsch alignment with constant gap penalty.
    aln = parasail.nw_trace_striped_32(seq2, seq1, 2, 2, parasail.dnafull)

    # pair up positions
    alignment = np.column_stack([
        np.cumsum([x != '-' for x in aln.traceback.ref]) - 1,
        np.cumsum([x != '-' for x in aln.traceback.query]) - 1
    ])

    path_range1 = np.column_stack([path1, path1[1:] + [len1]])
    path_range2 = np.column_stack([path2, path2[1:] + [len2]])

    envelope = np.full((len1, 2), -1, dtype=int)

    for idx1, idx2 in alignment.clip(0):

        st_1, en_1 = path_range1[idx1]
        st_2, en_2 = path_range2[idx2]

        for idx in range(st_1, en_1):
            if st_2 < envelope[idx, 0] or envelope[idx, 0] < 0:
                envelope[idx, 0] = st_2
            if en_2 > envelope[idx, 1] or envelope[idx, 1] < 0:
                envelope[idx, 1] = en_2

    # add a little padding to ensure some overlap
    envelope[:, 0] = envelope[:, 0] - padding
    envelope[:, 1] = envelope[:, 1] + padding
    envelope = np.clip(envelope, 0, len2)

    prev_end = 0
    for i in range(envelope.shape[0]):

        if envelope[i, 0] > envelope[i, 1]:
            envelope[i, 0] = 0

        if envelope[i, 0] > prev_end:
            envelope[i, 0] = prev_end

        prev_end = envelope[i, 1]

    return envelope.astype(np.uint64)


def find_follow_on(df, gap=5, distance=51, cov=0.85, min_len=100):
    """
    Find follow on reads from a sequencing summary file.
    """
    df = df[
        df.alignment_coverage.astype('float32').gt(cov) &
        df.sequence_length_template.astype('int32').gt(min_len)
    ]
    df = df.sort_values(['run_id', 'channel', 'mux', 'start_time'])

    genome_start = np.array(df.alignment_genome_start, dtype=np.int32)
    genome_end = np.array(df.alignment_genome_end, dtype=np.int32)
    direction = np.array(df.alignment_direction)
    start_time = np.array(df.start_time, dtype=np.float32)
    end_time = np.array(df.start_time + df.duration, dtype=np.float32)
    channel = np.array(df.channel, dtype=np.int32)
    mux = np.array(df.mux, dtype=np.int32)

    filt = (
        (channel[1:] == channel[:-1]) &
        (mux[1:] == mux[:-1]) &
        (np.abs(genome_start[1:] - genome_start[:-1]) < distance) &
        (np.abs(genome_end[1:] - genome_end[:-1]) < distance) &
        (direction[1:] != direction[:-1]) &
        (start_time[1:] - end_time[:-1] < gap)
    )
    mask = np.full(len(filt) + 1, False)
    mask[:-1] = mask[:-1] | filt
    mask[1:] = mask[1:] | filt

    return df[mask]


def compute_scores(model, batch, reverse=False):
    with torch.no_grad():
        device = next(model.parameters()).device
        dtype = torch.float16 if half_supported() else torch.float32
        scores = model.encoder(batch.to(dtype).to(device))
        if reverse: scores = model.seqdist.reverse_complement(scores)
        betas = model.seqdist.backward_scores(scores.to(torch.float32))
        trans, init = model.seqdist.compute_transition_probs(scores, betas)
    return {
        'trans': trans.to(dtype).transpose(0, 1),
        'init': init.to(dtype).unsqueeze(1),
    }


def basecall(model, reads, chunksize=4000, overlap=500, batchsize=32, reverse=False):
    reads = (
        read_chunk for read in reads
        for read_chunk in split_read(read, chunksize * batchsize)[::-1 if reverse else 1]
    )
    chunks = (
        ((read, start, end),
        bonito.util.chunk(torch.from_numpy(read.signal[start:end]), chunksize, overlap))
        for (read, start, end) in reads
    )
    batches = (
        (k, compute_scores(model, batch, reverse=reverse))
        for k, batch in bonito.util.batchify(chunks, batchsize=batchsize)
    )
    stitched = (
        (read, stitch(x, chunksize, overlap, end - start, model.stride, reverse=reverse))
        for ((read, start, end), x) in bonito.util.unbatchify(batches)
    )
    transferred = thread_map(transfer, stitched, n_thread=1)

    return (
        (read, concat([part for k, part in parts]))
        for read, parts in groupby(transferred, lambda x: x[0])
    )


def beam_search_duplex(seq1, path1, t1, b1, seq2, path2, t2, b2, alphabet='NACGT', beamsize=5, pad=40, T=0.01):
    env = build_envelope(t1.shape[0], seq1, path1, t2.shape[0], seq2, path2, padding=pad)
    return crf_beam_search_duplex(
        t1, b1, t2, b2,
        alphabet=alphabet,
        beam_size=beamsize,
        beam_cut_threshold=T,
        envelope=env,
    )


def decode(res, beamsize_1=5, pad_1=40, cut_1=0.01, beamsize_2=5, pad_2=40, cut_2=0.01, match=80, alphabet="NACGT"):

    temp_probs, init1 = res[0]['trans'].astype(np.float32), res[0]['init'][0].astype(np.float32)
    comp_probs, init2 = res[1]['trans'].astype(np.float32), res[1]['init'][0].astype(np.float32)

    simplex1, path1 = crf_beam_search(temp_probs, init1, alphabet, beam_size=5, beam_cut_threshold=0.01)
    simplex2, path2 = crf_beam_search(comp_probs, init2, alphabet, beam_size=5, beam_cut_threshold=0.01)

    if len(simplex1) < 10 or len(simplex2) < 10:
        return [simplex1, simplex2]

    if accuracy(simplex1, simplex2) < match:
        return [simplex1, simplex2]

    duplex1 = beam_search_duplex(
        simplex1, path1, temp_probs, init1, simplex2, path2, comp_probs, init2, pad=pad_1, beamsize=5, T=cut_1
    )
    duplex2 = beam_search_duplex(
        simplex2, path2, comp_probs, init2, simplex1, path1, temp_probs, init1, pad=pad_2, beamsize=5, T=cut_2
    )
    return [duplex1, duplex2, simplex1, simplex2]


def poa(seqs, allseq=False):
    con, msa = spoa.poa(seqs, genmsa=False)
    if allseq: return (*seqs, con)
    return (con, )


def call(model, reads_directory, summary_temp, summary_comp, aligner=None, qscore=False):

    temp_reads = read_gen(reads_directory, summary_temp, n_proc=8, cancel=process_cancel())
    comp_reads = read_gen(reads_directory, summary_comp, n_proc=8, cancel=process_cancel())

    temp_scores = basecall(model, temp_reads, reverse=False)
    comp_scores = basecall(model, comp_reads, reverse=True)

    scores = (((r1, r2), (s1, s2)) for (r1, s1), (r2, s2) in zip(temp_scores, comp_scores))

    calls = thread_map(decode, scores, n_thread=16)
    sequences = ((reads, seqs) for reads, seqs in calls if len(seqs) > 2)

    consensus = process_map(poa, sequences, n_proc=8)

    res = ((reads, {'sequence': seq}) for reads, seqs in consensus for seq in seqs)

    if aligner is None: return res
    return align_map(aligner, res)


def main(args):

    sys.stderr.write("> loading model\n")
    model = bonito.util.load_model(args.model, args.device)

    if args.reference:
        sys.stderr.write("> loading reference\n")
        aligner = Aligner(args.reference, preset='ont-map')
        if not aligner:
            sys.stderr.write("> failed to load/build index\n")
            exit(1)
    else:
        aligner = None

    if args.summary:
        sys.stderr.write("> finding follow on strands\n")
        summary = pd.read_csv(args.summary, '\t', low_memory=False)
        summary = summary[summary.sequence_length_template.gt(0)]
        valid_fast5s = [
            f for f in summary.filename_fast5.unique()
            if ((args.reads_directory / Path(f)).exists())
        ]
        summary = summary[summary.filename_fast5.isin(valid_fast5s)]
        summary = find_follow_on(summary)
        sys.stderr.write(f"> found {len(summary) // 2} follow strands\n")

        pairs = summary.head(args.max_reads)
        temp_reads = pairs.iloc[0::2]
        comp_reads = pairs.iloc[1::2]
    else:
        if args.index is not None:
            sys.stderr.write("> loading read index\n")
            index = json.load(open(args.index, 'r'))
        else:
            sys.stderr.write("> building read index\n")
            files = list(glob(os.path.join(args.reads_directory, '*.fast5')))
            index = build_index(files, n_proc=8)
            if args.save_index:
                with open('bonito-read-id.idx', 'w') as f:
                    json.dump(index, f)

        pairs = pd.read_csv(args.pairs, sep=args.sep, names=['read_1', 'read_2']).head(args.max_reads)
        pairs['file_1'] = pairs['read_1'].apply(index.get)
        pairs['file_2'] = pairs['read_2'].apply(index.get)
        pairs = pairs.dropna().reset_index()

        temp_reads = pairs[['read_1', 'file_1']].rename(columns={'read_1': 'read_id', 'file_1': 'filename_fast5'})
        comp_reads = pairs[['read_2', 'file_2']].rename(columns={'read_2': 'read_id', 'file_2': 'filename_fast5'})

    if len(pairs) == 0:
        print("> no valid pairs found", file=sys.stderr)
        exit(1)

    basecalls = call(model, args.reads_directory, temp_reads, comp_reads, aligner=aligner)
    writer = Writer(tqdm(basecalls, desc="> calling", unit=" reads", leave=False), aligner, duplex=True)

    t0 = perf_counter()
    writer.start()
    writer.join()
    duration = perf_counter() - t0
    num_samples = sum(num_samples for read_id, num_samples in writer.log)

    print("> duration: %s" % timedelta(seconds=np.round(duration)), file=sys.stderr)
    print("> samples per second %.1E" % (num_samples / duration), file=sys.stderr)


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model")
    parser.add_argument("reads_directory")

    parser.add_argument("--summary", default=None)

    parser.add_argument("--pairs")
    parser.add_argument("--sep", default=' ')
    parser.add_argument("--index", default=None)
    parser.add_argument("--save-index", action="store_true", default=False)

    parser.add_argument("--reference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-reads", default=100, type=int)
    return parser
