"""
Bonito Decoding functions
"""

import os
import sys
from textwrap import wrap
from itertools import groupby
from multiprocessing import Process, Queue

import numpy as np

from fast_ctc_decode import beam_search


def decode_ref(encoded, labels):
    """
    Convert a integer encoded reference into a string and remove blanks
    """
    return ''.join(labels[e] for e in encoded if e)


def greedy_ctc_decode(predictions, labels):
    """
    Greedy argmax decoder with collapsing repeats
    """
    path = np.argmax(predictions, axis=1)
    return ''.join([labels[b] for b, g in groupby(path) if b])


def decode(predictions, alphabet, beam_size=5, threshold=0.1):
    """
    Decode model posteriors to sequence
    """
    alphabet = ''.join(alphabet)
    if beam_size == 1:
        return greedy_ctc_decode(predictions, alphabet)
    return beam_search(predictions.astype(np.float32), alphabet, beam_size, threshold)


class DecoderWriter(Process):
    """
    Decoder Process that writes fasta records to stdout
    """
    def __init__(self, alphabet, beamsize=5):
        super().__init__()
        self.queue = Queue()
        self.beamsize = beamsize
        self.alphabet = ''.join(alphabet)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        while True:
            job = self.queue.get()
            if job is None: return
            read_id, predictions = job
            sequence = decode(predictions, self.alphabet, self.beamsize)
            sys.stdout.write(">%s\n" % read_id)
            sys.stdout.write("%s\n" % os.linesep.join(wrap(sequence, 100)))
            sys.stdout.flush()

    def stop(self):
        self.queue.put(None)
        self.join()
