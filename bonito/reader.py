"""
Bonito Read Utils
"""

from glob import iglob
from collections import OrderedDict
from importlib import import_module

import torch
import numpy as np
from scipy.signal import find_peaks


__formats__ = ["fast5", "pod5"]


class Reader:

    def __init__(self, directory, recursive=False):
        self.fmt = None
        for fmt in __formats__:
            pattern = f"**/*.{fmt}" if recursive else f"*.{fmt}"
            match = next(iglob(directory + "/" + pattern, recursive=True), None)
            if match is not None:
                self.fmt = fmt
                break
        else:
            raise FileNotFoundError()

        _reader = import_module(f"bonito.{self.fmt}")
        self._get_reads = getattr(_reader, "get_reads")
        self._get_read_groups = getattr(_reader, "get_read_groups")

    def get_reads(self, *args, **kwargs):
        return self._get_reads(*args, **kwargs)

    def get_read_groups(self, *args, **kwargs):
        return self._get_read_groups(*args, **kwargs)


class Read:

    def __init__(self, read, filename, meta=False):
        ...

    def __repr__(self):
        return "Read('%s')" % self.read_id

    def readgroup(self, model):
        self._groupdict = OrderedDict([
            ('ID', f"{self.run_id}_{model}"),
            ('PL', f"ONT"),
            ('DT', f"{self.exp_start_time}"),
            ('PU', f"{self.flow_cell_id}"),
            ('PM', f"{self.device_id}"),
            ('LB', f"{self.sample_id}"),
            ('SM', f"{self.sample_id}"),
            ('DS', f"%s" % ' '.join([
                f"run_id={self.run_id}",
                f"basecall_model={model}",
            ]))
        ])
        return '\t'.join(["@RG", *[f"{k}:{v}" for k, v in self._groupdict.items()]])

    def tagdata(self):
        return [
            f"mx:i:{self.mux}",
            f"ch:i:{self.channel}",
            f"st:Z:{self.start_time}",
            f"rn:i:{self.read_number}",
            f"f5:Z:{self.filename}",
            f"sm:f:{self.shift}",
            f"sd:f:{self.scale}",
            f"sv:Z:quantile",
        ]


class ReadChunk:

    def __init__(self, read, chunk, i, n):
        self.read_id = "%s:%i:%i" % (read.read_id, i, n)
        self.run_id = read.run_id
        self.filename = read.filename
        self.mux = read.mux
        self.channel = read.channel
        self.start = read.start
        self.duration = read.duration
        self.template_start = self.start
        self.template_duration = self.duration
        self.signal = chunk

    def __repr__(self):
        return "ReadChunk('%s')" % self.read_id


def read_chunks(read, chunksize=4000, overlap=400):
    """
    Split a Read in fixed sized ReadChunks
    """
    if len(read.signal) < chunksize:
        return

    _, offset = divmod(len(read.signal) - chunksize, chunksize - overlap)
    signal = torch.from_numpy(read.signal[offset:])
    blocks = signal.unfold(0, chunksize, chunksize - overlap)

    for i, block in enumerate(blocks):
        yield ReadChunk(read, block.numpy(), i+1, blocks.shape[0])


def trim(signal, shift, scale, window_size=40, threshold_factor=2.4, min_elements=3):

    min_trim = 10
    signal = signal[min_trim:]

    threshold = shift + scale * threshold_factor
    num_windows = len(signal) // window_size

    seen_peak = False

    for pos in range(num_windows):
        start = pos * window_size
        end = start + window_size
        window = signal[start:end]
        if len(window[window > threshold]) > min_elements or seen_peak:
            seen_peak = True
            if window[-1] > threshold:
                continue
            return min(end + min_trim, len(signal)), len(signal)

    return min_trim, len(signal)


def normalisation(sig):
    """
    Calculate signal shift and scale factors for normalisation..
    """
    q20, q90 = np.quantile(sig, [0.2, 0.9])
    shift = max(10, 0.51 * (q20 + q90))
    scale = max(1.0, 0.53 * (q90 - q20))
    return shift, scale
