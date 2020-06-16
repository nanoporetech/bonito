"""
Bonito Input/Output
"""

import os
import sys
from glob import glob
from textwrap import wrap
from warnings import warn
from logging import getLogger
from multiprocessing import Process, Queue, Lock, cpu_count

import numpy as np
from tqdm import tqdm

from bonito.util import get_raw_data


logger = getLogger('bonito')


def write_fasta(header, sequence, fd=sys.stdout, maxlen=100):
    """
    Write a fasta record to a file descriptor.
    """
    fd.write(">%s\n" % header)
    fd.write("%s\n" % os.linesep.join(wrap(sequence, maxlen)))
    fd.flush()


def write_fastq(header, sequence, qstring, fd=sys.stdout):
    """
    Write a fastq record to a file descriptor.
    """
    fd.write("@%s\n" % header)
    fd.write("%s\n" % sequence)
    fd.write("+\n")
    fd.write("%s\n" % qstring)
    fd.flush()


class PreprocessReader(Process):
    """
    Reader Processor that reads and processes fast5 files
    """
    def __init__(self, directory, maxsize=5):
        super().__init__()
        self.directory = directory
        self.queue = Queue(maxsize)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        for fast5 in tqdm(glob("%s/*fast5" % self.directory), ascii=True, ncols=100):
            for read_id, raw_data in get_raw_data(fast5):
                self.queue.put((read_id, raw_data))
        self.queue.put(None)

    def stop(self):
        self.join()


class DecoderWriterPool:
   """
   Simple pool of decoder writers
   """
   def __init__(self, model, procs=4, **kwargs):
       self.lock = Lock()
       self.queue = Queue()
       self.procs = procs if procs else cpu_count()
       self.decoders = []
       for _ in range(self.procs):
           decoder = DecoderWriter(model, self.queue, self.lock, **kwargs)
           decoder.start()
           self.decoders.append(decoder)

   def stop(self):
       for decoder in self.decoders: self.queue.put(None)
       for decoder in self.decoders: decoder.join()

   def __enter__(self):
       return self

   def __exit__(self, exc_type, exc_val, exc_tb):
       self.stop()


class DecoderWriter(Process):
    """
    Decoder Process that writes fasta records to stdout
    """
    def __init__(self, model, queue, lock, fastq=False, beamsize=5, wrap=100):
        super().__init__()
        self.queue = queue
        self.lock = lock
        self.model = model
        self.wrap = wrap
        self.fastq = fastq
        self.beamsize = beamsize

    def run(self):
        while True:
            job = self.queue.get()
            if job is None: return
            read_id, predictions = job

            # convert logprobs to probs
            predictions = np.exp(predictions.astype(np.float32))

            sequence, path = self.model.decode(
                predictions, beamsize=self.beamsize, qscores=self.fastq, return_path=True
            )
            if sequence:
                with self.lock:
                    if self.fastq: write_fastq(read_id, sequence[:len(path)], sequence[len(path):])
                    else: write_fasta(read_id, sequence, maxlen=self.wrap)
            else:
                logger.warn("> skipping empty sequence %s", read_id)
