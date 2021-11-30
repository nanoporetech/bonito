"""
Bonito Multiprocesing
"""

import queue
from itertools import count
from threading import Thread
from functools import partial
from signal import signal, SIGINT
from multiprocessing import Process, Queue, Event


def process_iter(iterator, maxsize=1):
    """
    Take an iterator and run it on another process.
    """
    return iter(ProcessIterator(iterator, maxsize=maxsize))


def thread_iter(iterator, maxsize=1):
    """
    Take an iterator and run it on another thread.
    """
    return iter(ThreadIterator(iterator, maxsize=maxsize))


def process_cancel():
    """
    Register an cancel event on sigint
    """
    event = Event()
    signal(SIGINT, lambda *a: event.set())
    return event


def process_map(func, iterator, n_proc=4, maxsize=2):
    """
    Take an `iterator` of key, value pairs and apply `func` to all values using `n_proc` processes.
    """
    if n_proc == 0: return ((k, func(v)) for k, v in iterator)
    return iter(ProcessMap(func, iterator, n_proc, output_queue=Queue(maxsize)))


def process_itemmap(func, iterator, n_proc=4, maxsize=2):
    """
    Take an `iterator` of key, value pairs and apply `func` to all (key, values) using `n_proc` processes.
    """
    if n_proc == 0: return ((k, func(k, v)) for k, v in iterator)
    yield from ProcessMap(
        func,
        iterator,
        n_proc,
        output_queue=Queue(maxsize),
        starmap=False,
        send_key=True
    )


def thread_map(func, iterator, n_thread=4, maxsize=2):
    """
    Take an `iterator` of key, value pairs and apply `func` to all values using `n_thread` threads.
    """
    if n_thread == 0: return ((k, func(v)) for k, v in iterator)
    return iter(
        ThreadMap(partial(MapWorkerThread, func, starmap=False), iterator, n_thread, maxsize=maxsize)
    )


def thread_itemmap(func, iterator, n_thread=4, maxsize=2):
    """
    Take an `iterator` of key, value pairs and apply `func` to all (key, values) using `n_thread` threads.
    """
    if n_thread == 0: return ((k, func(k, v)) for k, v in iterator)
    yield from ThreadMap(
        partial(MapWorkerThread, func, starmap=False, send_key=True),
        iterator,
        n_thread,
        maxsize=maxsize,
    )


def thread_starmap(func, iterator, n_thread=4, maxsize=2):
    """
    Take an `iterator` of key, value pairs and apply `func` to all values using `n_thread` threads.
    """
    if n_thread == 0: return ((k, func(*v)) for k, v in iterator)
    return iter(
        ThreadMap(partial(MapWorkerThread, func, starmap=True), iterator, n_thread, maxsize=maxsize)
    )


class BackgroundIterator:
    """
    Runs an iterator in the background.
    """
    def __init__(self, iterator, maxsize=10):
        super().__init__()
        self.iterator = iterator
        self.queue = self.QueueClass(maxsize)

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


class ThreadIterator(BackgroundIterator, Thread):
    """
    Runs an iterator in a separate process.
    """
    QueueClass = queue.Queue


class ProcessIterator(BackgroundIterator, Process):
    """
    Runs an iterator in a separate process.
    """
    QueueClass = Queue


class MapWorker(Process):
    """
    Process that reads items from an input_queue, applies a
    func to them and puts them on an output_queue.
    """
    def __init__(self, func, input_queue, output_queue, starmap=False, send_key=False):
        super().__init__()
        self.func = func
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.starmap = starmap
        self.send_key = send_key

    def run(self):
        while True:
            item = self.input_queue.get()
            if item is StopIteration:
                break
            k, v = item
            if self.starmap:
                if self.send_key:
                    self.output_queue.put((k, self.func(k, *v)))
                else:
                    self.output_queue.put((k, self.func(*v)))
            else:
                if self.send_key:
                    self.output_queue.put((k, self.func(k, v)))
                else:
                    self.output_queue.put((k, self.func(v)))


class ProcessMap(Thread):

    def __init__(self, func, iterator, n_proc, output_queue=None, starmap=False, send_key=False):
        super().__init__()
        self.iterator = iterator
        self.starmap = starmap
        self.send_key = send_key
        self.work_queue = Queue(n_proc * 2)
        self.output_queue = output_queue or Queue()
        self.processes = [
            MapWorker(func, self.work_queue, self.output_queue, self.starmap, self.send_key)
            for _ in range(n_proc)
        ]

    def start(self):
        for process in self.processes:
            process.start()
        super().start()

    def run(self):
        for k, v in self.iterator:
            self.work_queue.put((k, v))
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
            yield item


class MapWorkerThread(Thread):
    """
    Process that reads items from an input_queue, applies a func
    to them and puts them on an output_queue.
    """
    def __init__(self, func, input_queue=None, output_queue=None, starmap=False, send_key=False):
        super().__init__()
        self.func = func
        self.starmap = starmap
        self.send_key = send_key
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        while True:
            item = self.input_queue.get()
            if item is StopIteration:
                self.output_queue.put(item)
                break
            k, v = item
            if self.starmap:
                if self.send_key:
                    self.output_queue.put((k, self.func(k, *v)))
                else:
                    self.output_queue.put((k, self.func(*v)))
            else:
                if self.send_key:
                    self.output_queue.put((k, self.func(k, v)))
                else:
                    self.output_queue.put((k, self.func(v)))

class ThreadMap(Thread):

    def __init__(self, worker_type, iterator, n_thread, maxsize=2):
        super().__init__()
        self.iterator = iterator
        self.n_thread = n_thread
        self.work_queues = [queue.Queue(maxsize) for _ in range(n_thread)]
        self.output_queues = [queue.Queue(maxsize) for _ in range(n_thread)]
        self.workers = [
            worker_type(input_queue=in_q, output_queue=out_q)
            for (in_q, out_q) in zip(self.work_queues, self.output_queues)
        ]

    def start(self):
        for worker in self.workers:
            worker.start()
        super().start()

    def __iter__(self):
        self.start()
        for i in count():
            item = self.output_queues[i % self.n_thread].get()
            if item is StopIteration:
                #do we need to empty output_queues in order to join worker threads?
                for j in range(i + 1, i + self.n_thread):
                    self.output_queues[j % self.n_thread].get()
                break
            yield item

    def run(self):
        for i, (k, v) in enumerate(self.iterator):
            self.work_queues[i % self.n_thread].put((k, v))
        for q in self.work_queues:
            q.put(StopIteration)
        for worker in self.workers:
            worker.join()
