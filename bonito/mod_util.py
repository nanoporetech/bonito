import array

import numpy as np

from remora.util import seq_to_int
from remora.data_chunks import RemoraRead
from remora.inference import call_read_mods


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


def alphabet_str(mods_model):
    remora_model, remora_metadata = mods_model
    can_base = remora_metadata["motif"][0][remora_metadata["motif"][1]]
    mod_str = "; ".join(
        f"{mod_b}={mln}"
        for mod_b, mln in zip(
            remora_metadata["mod_bases"],
            remora_metadata["mod_long_names"]
        )
    )
    return f"loaded modified base model to call (alt to {can_base}): {mod_str}"


def call_mods(mods_model, read, read_attrs):
    if len(read_attrs['sequence']) == 0:
        return read_attrs
    remora_model, remora_metadata = mods_model
    can_base = remora_metadata["motif"][0][remora_metadata["motif"][1]]
    int_seq = seq_to_int(read_attrs['sequence'].upper())
    seq_to_sig_map = np.empty(
        len(read_attrs['sequence']) + 1, dtype=np.int32
    )
    seq_to_sig_map[-1] = read.signal.shape[0]
    seq_to_sig_map[:-1] = read_attrs['seq_to_sig_map']
    remora_read = RemoraRead(
        read.signal,
        int_seq,
        seq_to_sig_map,
    )
    mod_calls, _, pos = call_read_mods(
        remora_read,
        remora_model,
        remora_metadata,
    )
    log_probs = log_softmax_axis1(mod_calls)[:, 1:].astype(np.float64)
    read_attrs['mods'] = format_mm_ml_tags(
        read_attrs['sequence'],
        pos,
        log_probs,
        remora_metadata["mod_bases"],
        can_base
    )
    return read_attrs
