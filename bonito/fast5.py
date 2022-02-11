"""
Bonito Fast5 Utils
"""

import sys
from glob import glob
from pathlib import Path
from itertools import chain
from functools import partial
from multiprocessing import Pool
from collections import OrderedDict
from datetime import datetime, timedelta

import torch
import numpy as np
from tqdm import tqdm
from dateutil import parser
from scipy.signal import find_peaks
from ont_fast5_api.fast5_interface import get_fast5_file


class Read:

    def __init__(self, read, filename, meta=False):

        self.meta = meta
        self.read_id = read.read_id
        self.filename = filename.name
        self.run_id = read.get_run_id()
        if type(self.run_id) in (bytes, np.bytes_):
            self.run_id = self.run_id.decode('ascii')

        tracking_id = read.handle[read.global_key + 'tracking_id'].attrs

        self.sample_id = tracking_id['sample_id']
        if type(self.sample_id) in (bytes, np.bytes_):
            self.sample_id = self.sample_id.decode()

        self.exp_start_time = tracking_id['exp_start_time']
        if type(self.exp_start_time) in (bytes, np.bytes_):
            self.exp_start_time = self.exp_start_time.decode('ascii')
        self.exp_start_time = self.exp_start_time.replace('Z', '')

        self.flow_cell_id = tracking_id['flow_cell_id']
        if type(self.flow_cell_id) in (bytes, np.bytes_):
            self.flow_cell_id = self.flow_cell_id.decode('ascii')

        self.device_id = tracking_id['device_id']
        if type(self.device_id) in (bytes, np.bytes_):
            self.device_id = self.device_id.decode('ascii')

        if self.meta:
            return

        read_attrs = read.handle[read.raw_dataset_group_name].attrs
        channel_info = read.handle[read.global_key + 'channel_id'].attrs

        self.offset = int(channel_info['offset'])
        self.sampling_rate = channel_info['sampling_rate']
        self.scaling = channel_info['range'] / channel_info['digitisation']

        self.mux = read_attrs['start_mux']
        self.read_number = read_attrs['read_number']
        self.channel = channel_info['channel_number']
        if type(self.channel) in (bytes, np.bytes_):
            self.channel = self.channel.decode()

        self.start = read_attrs['start_time'] / self.sampling_rate
        self.duration = read_attrs['duration'] / self.sampling_rate

        exp_start_dt = parser.parse(self.exp_start_time)
        start_time = exp_start_dt + timedelta(seconds=self.start)
        self.start_time = start_time.replace(microsecond=0).isoformat()

        raw = read.handle[read.raw_dataset_name][:]
        scaled = np.array(self.scaling * (raw + self.offset), dtype=np.float32)
        self.num_samples = len(scaled)

        trim_start, _ = trim(scaled[:8000])
        scaled = scaled[trim_start:]
        self.trimmed_samples = trim_start
        self.template_start = self.start + (1 / self.sampling_rate) * trim_start
        self.template_duration = self.duration - (1 / self.sampling_rate) * trim_start

        if len(scaled) > 8000:
            med, mad = med_mad(scaled)
            self.signal = (scaled - med) / max(1.0, mad)
        else:
            self.signal = norm_by_noisiest_section(scaled)

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


def trim(signal, window_size=40, threshold_factor=2.4, min_elements=3):

    min_trim = 10
    signal = signal[min_trim:]

    med, mad = med_mad(signal[-(window_size*100):])

    threshold = med + mad * threshold_factor
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


def med_mad(x, factor=1.4826):
    """
    Calculate signal median and median absolute deviation
    """
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor + np.finfo(np.float32).eps
    return med, mad


def norm_by_noisiest_section(signal, samples=100, threshold=6.0):
    """
    Normalise using the medmad from the longest continuous region where the
    noise is above some threshold relative to the std of the full signal.
    """
    threshold = signal.std() / threshold
    noise = np.ones(signal.shape)

    for idx in np.arange(signal.shape[0] // samples):
        window = slice(idx * samples, (idx + 1) * samples)
        noise[window] = np.where(signal[window].std() > threshold, 1, 0)

    # start and end low for peak finding
    noise[0] = 0; noise[-1] = 0
    peaks, info = find_peaks(noise, width=(None, None))

    if len(peaks):
        widest = np.argmax(info['widths'])
        med, mad = med_mad(signal[info['left_bases'][widest]: info['right_bases'][widest]])
    else:
        med, mad = med_mad(signal)
    return (signal - med) / mad


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


def get_meta_data(filename, read_ids=None, skip=False):
    """
    Get the meta data from the fast5 file for a given `filename`.
    """
    meta_reads = []
    with get_fast5_file(filename, 'r') as f5_fh:
        for read_id in f5_fh.get_read_ids():
            if read_ids is None or (read_id in read_ids) ^ skip:
                meta_reads.append(
                    Read(f5_fh.get_read(read_id), filename, meta=True)
                )
        return meta_reads


def get_read_groups(directory, model, read_ids=None, skip=False, n_proc=1, recursive=False, cancel=None):
    """
    Get all the read meta data for a given `directory`.
    """
    groups = set()
    pattern = "**/*.fast5" if recursive else "*.fast5"
    fast5s = [Path(x) for x in glob(directory + "/" + pattern, recursive=True)]
    get_filtered_meta_data = partial(get_meta_data, read_ids=read_ids, skip=skip)

    with Pool(n_proc) as pool:
        for reads in tqdm(
                pool.imap(get_filtered_meta_data, fast5s), total=len(fast5s), leave=False,
                desc="> preprocessing reads", unit=" fast5s", ascii=True, ncols=100
        ):
            groups.update({read.readgroup(model) for read in reads})
        return groups


def get_read_ids(filename, read_ids=None, skip=False):
    """
    Get all the read_ids from the file `filename`.
    """
    with get_fast5_file(filename, 'r') as f5_fh:
        ids = [(filename, rid) for rid in f5_fh.get_read_ids()]
        if read_ids is None:
            return ids
        return [rid for rid in ids if (rid[1] in read_ids) ^ skip]


def get_raw_data_for_read(info):
    """
    Get the raw signal from the fast5 file for a given filename, read_id pair
    """
    filename, read_id = info
    with get_fast5_file(filename, 'r') as f5_fh:
        return Read(f5_fh.get_read(read_id), filename)


def get_raw_data(filename, read_ids=None, skip=False):
    """
    Get the raw signal and read id from the fast5 files
    """
    with get_fast5_file(filename, 'r') as f5_fh:
        for read_id in f5_fh.get_read_ids():
            if read_ids is None or (read_id in read_ids) ^ skip:
                yield Read(f5_fh.get_read(read_id), filename)


def get_reads(directory, read_ids=None, skip=False, n_proc=1, recursive=False, cancel=None):
    """
    Get all reads in a given `directory`.
    """
    pattern = "**/*.fast5" if recursive else "*.fast5"
    get_filtered_reads = partial(get_read_ids, read_ids=read_ids, skip=skip)
    reads = (Path(x) for x in glob(directory + "/" + pattern, recursive=True))
    with Pool(n_proc) as pool:
        for job in chain(pool.imap(get_filtered_reads, reads)):
            for read in pool.imap(get_raw_data_for_read, job):
                yield read
                if cancel is not None and cancel.is_set():
                    return
