import array
from threading import Thread
from functools import partial
from collections import defaultdict

import numpy as np

from bonito.multiprocessing import ThreadMap

try:
    from remora import RemoraError
    from remora.model_util import load_onnx_model
    from remora.inference import call_read_mods
    from remora.data_chunks import RemoraRead
    REMORA_INSTALLED = True
except ImportError:
    REMORA_INSTALLED = False

SEQ_MIN = np.array(["A"], dtype="S1").view(np.uint8)[0]
REMORA_SEQ_TO_INT_ARR = np.full(26, -1, dtype=np.int)
REMORA_SEQ_TO_INT_ARR[0] = 0
REMORA_SEQ_TO_INT_ARR[2] = 1
REMORA_SEQ_TO_INT_ARR[6] = 2
REMORA_SEQ_TO_INT_ARR[19] = 3


def seq_to_int_remora(seq):
    return REMORA_SEQ_TO_INT_ARR[
        np.array(list(seq), dtype="c").view(np.uint8) - SEQ_MIN
    ]


def log_softmax_axis1(x):
    """Compute log softmax over axis=1"""
    e_x = np.exp((x.T - np.max(x, axis=1)).T)
    with np.errstate(divide="ignore"):
        return np.log((e_x.T / e_x.sum(axis=1)).T)


def format_mm_ml_tags(seq, mod_scores):
    """Format MM and ML tags for BAM output. See
    https://samtools.github.io/hts-specs/SAMtags.pdf for format details.

    Args:
        seq (str): Read-centric read sequence. For reference-anchored calls this
            should be the reverse complement sequence.
        mod_scores (list): List of 3-tuples containing:
            1. Position relative to seq
            2. np.array or log probabilties for modified bases
            3. Modified base single letter codes (str)

    Returns:
        MM string tag and ML array tag
    """

    # initialize dict with all called mods to make sure all called mods are
    # shown in resulting tags
    per_mod_probs = defaultdict(list)
    for pos, mod_lps, mod_bases, can_base in sorted(mod_scores):
        # mod_lps is set to None if invalid sequence is encountered or too
        # few events are found around a mod
        if mod_lps is None:
            continue
        for mod_lp, mod_base in zip(mod_lps, mod_bases):
            mod_prob = np.exp(mod_lp)
            per_mod_probs[mod_base].append((pos, mod_prob))

    mm_tag, ml_tag = "", array.array("B")
    for mod_base, pos_probs in per_mod_probs.items():
        if len(pos_probs) == 0:
            continue
        mod_poss, probs = zip(*sorted(pos_probs))
        # compute modified base positions relative to the running total of the
        # associated canonical base
        can_base_mod_poss = (
            np.cumsum([1 if b == can_base else 0 for b in seq])[
                np.array(mod_poss)
            ]
            - 1
        )
        mm_tag += "{}+{}{};".format(
            can_base,
            mod_base,
            "".join(
                ",{}".format(d)
                for d in np.diff(np.insert(can_base_mod_poss, 0, -1)) - 1
            ),
        )
        # extract mod scores and scale to 0-255 range
        scaled_probs = np.floor(np.array(probs) * 256)
        # last interval includes prob=1
        scaled_probs[scaled_probs == 256] = 255
        ml_tag.extend(scaled_probs.astype(np.uint8))

    return mm_tag, ml_tag


def mods_tags_to_str(mods_tags):
    return [
        f"MM:Z:{mods_tags[0]}",
        f"ML:B:C,{','.join(map(str, mods_tags[1]))}"
    ]


class RemoraMods:
    def __init__(self, remora_model_filename):
        if not REMORA_INSTALLED:
            raise RuntimeError("Remora must be installed.")
        self.remora_model_filename = remora_model_filename
        self.remora_model, self.remora_metadata = load_onnx_model(
            self.remora_model_filename
        )
        self.can_base = self.remora_metadata["motif"][0][
            self.remora_metadata["motif"][1]
        ]

    @property
    def alphabet_str(self):
        mod_str = "; ".join(
            f"{mod_b}={mln}"
            for mod_b, mln in zip(
                self.remora_metadata["mod_bases"],
                self.remora_metadata["mod_long_names"]
            )
        )
        return (
            "Loaded Remora model calls modified bases (alt to "
            f"{self.can_base}): {mod_str}"
        )

    def call_mods(self, sig, seq, seq_to_sig_map):
        int_seq = seq_to_int_remora(seq.upper())
        remora_read = RemoraRead(
            sig,
            int_seq,
            seq_to_sig_map,
        )
        try:
            remora_read.check()
        except RemoraError as e:
            raise RuntimeError(f"Remora read prep error: {e}")
        mod_calls, _, pos = call_read_mods(
            remora_read,
            self.remora_model,
            self.remora_metadata,
        )
        log_probs = log_softmax_axis1(mod_calls)[:, 1:].astype(np.float64)
        return zip(
            map(int, pos),
            log_probs,
            self.remora_metadata["mod_bases"],
            self.can_base,
        )

    def call_mods_map(self, basecalls, stride, n_thread=1):
        """
        Call modified bases using Remora model using `n_thread` threads.
        """
        return ThreadMap(
            partial(RemoraWorker, self, stride), basecalls, n_thread
        )


class RemoraWorker(Thread):
    """Process that reads items from an input_queue, applies a func to them and
    puts them on an output_queue
    """
    def __init__(
        self, remora_model, stride, input_queue=None, output_queue=None
    ):
        super().__init__()
        self.remora_model = remora_model
        self.stride = stride
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        while True:
            item = self.input_queue.get()
            if item is StopIteration:
                self.output_queue.put(item)
                break
            read, read_attrs = item
            seq_to_sig_map = np.where(read_attrs['move'])[0] * self.stride
            mod_scores = self.remora_model.call_mods(
                read.signal, read_attrs['sequence'], seq_to_sig_map
            )
            mod_tags = format_mm_ml_tags(read_attrs['sequence'], mod_scores)
            self.output_queue.put(
                (read, {**read_attrs, 'mods': mod_tags})
            )
