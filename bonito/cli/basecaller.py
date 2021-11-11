"""
Bonito Basecaller
"""

import sys
import numpy as np
from tqdm import tqdm
from time import perf_counter
from datetime import timedelta
from itertools import islice as take
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.mod_util import ModsModel
from bonito.io import CTCWriter, Writer
from bonito.aligner import Aligner, align_map
from bonito.fast5 import get_reads, read_chunks
from bonito.multiprocessing import process_cancel, thread_iter
from bonito.util import column_to_set, load_symbol, load_model


def main(args):

    if args.save_ctc and not args.reference:
        sys.stderr.write("> a reference is needed to output ctc training data\n")
        exit(1)

    sys.stderr.write("> loading model\n")
    model = load_model(args.model_directory, args.device, weights=int(args.weights))
    basecall = load_symbol(args.model_directory, "basecall")

    mods_model = None
    if args.modified_base_model is not None:
        sys.stderr.write("> loading modified base model\n")
        mods_model = ModsModel(args.modified_base_model)
        sys.stderr.write(f"> {mods_model.alphabet_str}\n")

    if args.reference:
        sys.stderr.write("> loading reference\n")
        aligner = Aligner(args.reference, preset='ont-map', best_n=1)
        if not aligner:
            sys.stderr.write("> failed to load/build index\n")
            exit(1)
    else:
        aligner = None

    reads = get_reads(
        args.reads_directory, n_proc=8, recursive=args.recursive,
        read_ids=column_to_set(args.read_ids), skip=args.skip,
        cancel=process_cancel()
    )

    if args.max_reads:
        reads = take(reads, args.max_reads)

    if args.save_ctc:
        reads = (
            chunk for read in reads
            for chunk in read_chunks(read, chunksize=args.chunksize)
        )
        ResultsWriter = CTCWriter
    else:
        ResultsWriter = Writer

    results = basecall(
        model, reads, reverse=args.revcomp,
        batchsize=args.batchsize, chunksize=args.chunksize,
    )

    if mods_model is not None:
        results = thread_iter(
            (
                read,
                mods_model.call_mods_from_model(model.stride, read, read_attrs)
            )
            for read, read_attrs in results
        )
    if aligner:
        results = align_map(aligner, results)

    writer = ResultsWriter(
        tqdm(results, desc="> calling", unit=" reads", leave=False),
        aligner=aligner
    )

    t0 = perf_counter()
    writer.start()
    writer.join()
    duration = perf_counter() - t0
    num_samples = sum(num_samples for read_id, num_samples in writer.log)

    sys.stderr.write("> completed reads: %s\n" % len(writer.log))
    sys.stderr.write("> duration: %s\n" % timedelta(seconds=np.round(duration)))
    sys.stderr.write("> samples per second %.1E\n" % (num_samples / duration))
    sys.stderr.write("> done\n")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("reads_directory")
    parser.add_argument("--reference")
    parser.add_argument("--modified-base-model")
    parser.add_argument("--read-ids")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--skip", action="store_true", default=False)
    parser.add_argument("--save-ctc", action="store_true", default=False)
    parser.add_argument("--revcomp", action="store_true", default=False)
    parser.add_argument("--recursive", action="store_true", default=False)
    parser.add_argument("--batchsize", default=32, type=int)
    parser.add_argument("--chunksize", default=4000, type=int)
    parser.add_argument("--max-reads", default=0, type=int)
    return parser
