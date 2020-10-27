"""
Bonito Multiprocesing
"""

import queue
from threading import Thread
from functools import partial
from collections import deque
from multiprocessing import Process, Queue, Lock, cpu_count


def process_iter(iterator, maxsize=10):
    """
    Take an iterator and run it on another process.
    """
    return iter(ProcessIterator(iterator, maxsize=maxsize))


def process_map(func, iterator, n_proc=4, maxsize=0):
    """
    Take an `iterator` of key, value pairs and apply `func` to all values using `n_proc` processes.
    """
    if n_proc == 0: return ((k, func(v)) for k, v in iterator)
    return iter(ProcessMap(func, iterator, n_proc, output_queue=Queue(maxsize)))


def thread_map(func, iterator, n_thread=4, maxsize=0, preserve_order=False):
    """
    Take an `iterator` of key, value pairs and apply `func` to all values using `n_thread` threads.
    """
    if n_thread == 0: return ((k, func(v)) for k, v in iterator)
    return iter(
        ThreadMap(
            partial(MapWorkerThread, func),
            iterator, n_thread,
            output_queue=queue.Queue(maxsize),
            preserve_order=preserve_order
        )
    )


class ProcessIterator(Process):
    """
    Runs an iterator in a separate process
    """
    def __init__(self, iterator, maxsize=10):
        super().__init__()
        self.iterator = iterator
        self.queue = Queue(maxsize)

    def __iter__(self):
        self.start()
        while True:
            item = self.queue.get()
            if item is StopIteration:
                break
            yield item

    def run(self):
        for item in self.iterator:
            self.queue.put(item)
        self.queue.put(StopIteration)

    def stop(self):
        self.join()


class MapWorker(Process):
    """
    Process that reads items from an input_queue, applies a func to them and puts them on an output_queue
    """
    def __init__(self, func, input_queue, output_queue):
        super().__init__()
        self.func = func
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        while True:
            item = self.input_queue.get()
            if item is StopIteration:
                break
            k, v = item
            self.output_queue.put((k, self.func(v)))


class ProcessMap(Thread):

    def __init__(self, func, iterator, n_proc, output_queue=None):
        super().__init__()
        self.key_map = {}
        self.iterator = iterator
        self.work_queue = Queue(n_proc * 2)
        self.output_queue = output_queue or Queue()
        self.processes = [MapWorker(func, self.work_queue, self.output_queue) for _ in range(n_proc)]

    def start(self):
        for process in self.processes:
            process.start()
        super().start()

    def run(self):
        for (k, v) in self.iterator:
            self.work_queue.put((id(k), v))
            self.key_map[id(k)] = k
        for _ in self.processes:
            self.work_queue.put(StopIteration)
        for process in self.processes:
            process.join()
        self.output_queue.put(StopIteration)

    def __iter__(self):
        self.start()
        while True:
            item = self.output_queue.get()
            if item is StopIteration:
                break
            k, v = item
            yield self.key_map.pop(k), v


class MapWorkerThread(Thread):
    """
    Process that reads items from an input_queue, applies a func to them and puts them on an output_queue
    """
    def __init__(self, func, input_queue=None, output_queue=None):
        super().__init__()
        self.func = func
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        while True:
            item = self.input_queue.get()
            if item is StopIteration:
                break
            k, v = item
            self.output_queue.put((k, self.func(v)))


class ThreadMap(Thread):
    def __init__(self, worker_type, iterator, n_thread, output_queue=None, preserve_order=False):
        super().__init__()
        self.iterator = iterator
        self.work_queue = queue.Queue(n_thread*2)
        self.output_queue = output_queue or queue.Queue()
        self.workers = [worker_type(input_queue=self.work_queue, output_queue=self.output_queue) for _ in range(n_thread)]
        if preserve_order:
            self.keys = deque()
            self.results = {}
        else:
            self.keys = None
            self.results = None

    def start(self):
        for worker in self.workers:
            worker.start()
        super().start()

    def __iter__(self):
        self.start()
        while True:
            while self.keys:
                key = self.keys.popleft()
                if key in self.results:
                    yield (key, self.results.pop(key))
                else:
                    self.keys.appendleft(key)
                    break
            item = self.output_queue.get()
            if item is StopIteration:
                break

            if self.results is None:
                yield item
            else:
                k, v = item
                self.results[k] = v

    def run(self):
        for (k, v) in self.iterator:
            self.work_queue.put((k, v))
            if self.keys is not None: self.keys.append(k)
        for _ in self.workers:
            self.work_queue.put(StopIteration)
        for worker in self.workers:
            worker.join()
        self.output_queue.put(StopIteration)
