"""
Bonito Aligner
"""

from threading import Thread
from functools import partial
from mappy import Aligner, ThreadBuffer

from bonito.multiprocessing import ThreadMap


def align_map(aligner, sequences, n_thread=4):
    """
    Align `sequences` with minimap using `n_thread` threads.
    """
    return ThreadMap(partial(MappyWorker, aligner), sequences, n_thread)


class MappyWorker(Thread):
    """
    Process that reads items from an input_queue, applies a func to them and puts them on an output_queue
    """
    def __init__(self, aligner, input_queue=None, output_queue=None):
        super().__init__()
        self.aligner = aligner
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        thrbuf = ThreadBuffer()
        while True:
            item = self.input_queue.get()
            if item is StopIteration:
                self.output_queue.put(item)
                break
            k, v = item
            mapping = next(self.aligner.map(v['sequence'], buf=thrbuf, MD=True), None)
            self.output_queue.put((k, {**v, 'mapping': mapping}))
