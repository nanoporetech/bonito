"""
Bonito Basecaller
"""

import sys
import time
from math import ceil
from glob import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.decode import DecoderWriter
from bonito.util import load_model, chunk_data, stitch, get_raw_data

import torch
import numpy as np
from tqdm import tqdm


def main(args):

    sys.stderr.write("> loading model\n")
    model = load_model(args.model_directory, args.device, weights=int(args.weights))

    num_reads = 0
    num_chunks = 0
    t0 = time.perf_counter()

    sys.stderr.write("> calling\n")

    with DecoderWriter(model.alphabet, args.beamsize) as decoder:
        for fast5 in tqdm(glob("%s/*fast5" % args.reads_directory), ascii=True, ncols=100):
            for read_id, raw_data in get_raw_data(fast5):

                predictions = []
                chunks = chunk_data(raw_data, args.chunksize, args.overlap)
                num_reads += 1
                num_chunks += chunks.shape[0]

                with torch.no_grad():
                    for i in range(ceil(len(chunks) / args.batchsize)):
                        batch = chunks[i*args.batchsize: (i+1)*args.batchsize]
                        tchunks = torch.tensor(batch).to(args.device)
                        probs = torch.exp(model(tchunks))
                        predictions.append(probs.cpu())

                predictions = np.concatenate(predictions)
                predictions = stitch(predictions, int(args.overlap / model.stride / 2))

                decoder.queue.put((read_id, predictions))

    samples = num_chunks * args.chunksize
    duration = time.perf_counter() - t0

    sys.stderr.write("> completed reads: %s\n" % num_reads)
    sys.stderr.write("> samples per second %.1E\n" % (samples  / duration))
    sys.stderr.write("> done\n")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("reads_directory")
    parser.add_argument("model_directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--beamsize", default=5, type=int)
    parser.add_argument("--batchsize", default=64, type=int)
    parser.add_argument("--overlap", default=100, type=int)
    parser.add_argument("--chunksize", default=10000, type=int)
    return parser
