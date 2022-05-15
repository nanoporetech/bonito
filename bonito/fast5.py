"""
Bonito Fast5 Utils
"""

from glob import glob
from pathlib import Path
from itertools import chain
from functools import partial
from multiprocessing import Pool
from datetime import datetime, timedelta

import numpy as np
import bonito.reader
from tqdm import tqdm
from dateutil import parser
from ont_fast5_api.fast5_interface import get_fast5_file


class Read(bonito.reader.Read):

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

        trim_start, _ = bonito.reader.trim(scaled[:8000])
        scaled = scaled[trim_start:]
        self.trimmed_samples = trim_start
        self.template_start = self.start + (1 / self.sampling_rate) * trim_start
        self.template_duration = self.duration - (1 / self.sampling_rate) * trim_start

        if len(scaled) > 8000:
            med, mad = bonito.reader.med_mad(scaled)
            self.signal = (scaled - med) / max(1.0, mad)
        else:
            self.signal = bonito.reader.norm_by_noisiest_section(scaled)


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
