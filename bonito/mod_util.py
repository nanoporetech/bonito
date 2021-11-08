import array

import numpy as np

try:
    from remora import RemoraError
    from remora.util import seq_to_int
    from remora.data_chunks import RemoraRead
    from remora.inference import call_read_mods
    from remora.model_util import load_onnx_model
    REMORA_INSTALLED = True
except ImportError:
    REMORA_INSTALLED = False


def log_softmax_axis1(x):
    """Compute log softmax over axis=1"""
    e_x = np.exp((x.T - np.max(x, axis=1)).T)
    with np.errstate(divide="ignore"):
        return np.log((e_x.T / e_x.sum(axis=1)).T)


def format_mm_ml_tags(seq, poss, log_probs, mod_bases, can_base):
    """Format MM and ML tags for BAM output. See
    https://samtools.github.io/hts-specs/SAMtags.pdf for format details.

    Args:
        seq (str): read-centric read sequence. For reference-anchored calls
            this should be the reverse complement sequence.
        poss (list): positions relative to seq
        log_probs (np.array): log probabilties for modified bases
        mod_bases (str): modified base single letter codes
        can_base (str): canonical base

    Returns:
        MM string tag and ML array tag
    """

    # initialize dict with all called mods to make sure all called mods are
    # shown in resulting tags
    per_mod_probs = dict((mod_base, []) for mod_base in mod_bases)
    for pos, mod_lps in sorted(zip(poss, log_probs)):
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


class ModsModel:
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
            "loaded modified base model to call (alt to "
            f"{self.can_base}): {mod_str}"
        )

    def call_mods(self, sig, seq, seq_to_sig_map):
        int_seq = seq_to_int(seq.upper())
        remora_read = RemoraRead(
            sig,
            int_seq,
            seq_to_sig_map,
        )
        try:
            remora_read.check()
        except RemoraError as e:
            raise RuntimeError(f"Remora read prep error: {e}")
        if len(seq) == 0:
            return []
        mod_calls, _, pos = call_read_mods(
            remora_read,
            self.remora_model,
            self.remora_metadata,
        )
        log_probs = log_softmax_axis1(mod_calls)[:, 1:].astype(np.float64)
        return pos, log_probs, self.remora_metadata["mod_bases"], self.can_base

    def call_mods_from_model(self, stride, read, read_attrs):
        seq_to_sig_map = np.empty(
            len(read_attrs['sequence']) + 1, dtype=np.int32
        )
        seq_to_sig_map[-1] = read.signal.shape[0]
        seq_to_sig_map[:-1] = np.where(read_attrs['moves'])[0] * stride
        mod_scores = self.call_mods(
            read.signal, read_attrs['sequence'], seq_to_sig_map
        )
        mod_tags = format_mm_ml_tags(read_attrs['sequence'], *mod_scores)
        return {**read_attrs, 'mods': mod_tags}
