"""
Bonito Basecaller
"""

import sys
import time
from math import ceil
from glob import glob
from textwrap import wrap
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.util import load_model, decode_ctc

import torch
import numpy as np
from tqdm import tqdm
from ont_fast5_api.fast5_interface import get_fast5_file


def med_mad(x, factor=1.4826):
    """
    Calculate signal median and median absolute deviation
    """
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor
    return med, mad


def trim(signal, window_size=40, threshold_factor=3.0, min_elements=3):

    med, mad = med_mad(signal[-(window_size*25):])
    threshold = med + mad * threshold_factor
    num_windows = len(signal) // window_size

    for pos in range(num_windows):

        start = pos * window_size
        end = start + window_size

        window = signal[start:end]

        if len(window[window > threshold]) > min_elements:
            if window[-1] > threshold:
                continue
            return end, len(signal)

    return 0, len(signal)


def preprocess(x, min_samples=1000):
    start, end = trim(x)
    # REVISIT: we can potentially trim all the signal if this goes wrong
    if end - start < min_samples:
        start = 0
        end = len(x)
        #sys.stderr.write("badly trimmed read\n")

    med, mad = med_mad(x[start:end])
    norm_signal = (x[start:end] - med) / mad
    return norm_signal


def get_raw_data(fast5_filepath):
    """
    Get the raw signal and read id from the fast5 files
    """
    with get_fast5_file(fast5_filepath, mode="r") as f5:
        for read_id in f5.get_read_ids():
            read = f5.get_read(read_id)
            raw_data = read.get_raw_data(scale=True)
            raw_data = preprocess(raw_data)
            yield read_id, raw_data


def window(data, size, stepsize=1, padded=False, axis=-1):
    """
    Segment data in `size` chunks with overlap
    """
    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def stitch(predictions, overlap):
    stitched = [predictions[0, 0:-overlap]]
    for i in range(1, predictions.shape[0] - 1): stitched.append(predictions[i][overlap:-overlap])
    stitched.append(predictions[-1][overlap:])
    return np.concatenate(stitched)


def main(args):

    sys.stderr.write("> loading model\n")
    model = load_model(args.model_directory, args.device, weights=int(args.weights))

    num_reads = 0
    num_chunks = 0

    t0 = time.perf_counter()
    sys.stderr.write("> calling\n")

    for fast5 in tqdm(glob("%s/*fast5" % args.reads_directory), ascii=True):

        for read_id, raw_data in get_raw_data(fast5):

            if len(raw_data) <= args.chunksize:
                chunks = np.expand_dims(raw_data, axis=0)
            else:
                chunks = window(raw_data, args.chunksize, stepsize=args.chunksize - args.overlap)

            chunks = np.expand_dims(chunks, axis=1)

            num_reads += 1
            num_chunks += chunks.shape[0]
            predictions = []

            with torch.no_grad():

                for i in range(ceil(len(chunks) / args.batchsize)):
                    batch = chunks[i*args.batchsize: (i+1)*args.batchsize]
                    tchunks = torch.tensor(batch).to(args.device)
                    probs = torch.exp(model(tchunks))
                    predictions.append(probs.cpu())

                predictions = np.concatenate(predictions)

                if len(predictions) > 1:
                    predictions = stitch(predictions, int(args.overlap / model.stride / 2))
                else:
                    predictions = np.squeeze(predictions, axis=0)

                sequence = decode_ctc(predictions, model.alphabet)

                print(">%s" % read_id)
                print('\n'.join(wrap(sequence, 100)))

    t1 = time.perf_counter()
    sys.stderr.write("> completed reads: %s\n" % num_reads)
    sys.stderr.write("> samples per second %.1E\n" % (num_chunks * args.chunksize / (t1 - t0)))
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
    parser.add_argument("--batchsize", default=64, type=int)
    parser.add_argument("--chunks", default=500, type=int)
    parser.add_argument("--overlap", default=600, type=int)
    parser.add_argument("--chunksize", default=2000, type=int)
    return parser
