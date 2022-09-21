import re
from itertools import takewhile
from collections import OrderedDict

import pysam
import numpy as np
from mappy import revcomp
from edlib import align as edlib_align
from parasail import dnafull, sg_trace_scan_32


# Cigar int code ops are: MIDNSHP=X
CODE_TO_OP = OrderedDict((
    ("M", pysam.CMATCH),
    ("I", pysam.CINS),
    ("D", pysam.CDEL),
    ("N", pysam.CREF_SKIP),
    ("S", pysam.CSOFT_CLIP),
    ("H", pysam.CHARD_CLIP),
    ("P", pysam.CPAD),
    ("=", pysam.CEQUAL),
    ("X", pysam.CDIFF),
))
cigar_is_query = [True, True, False, False, True, False, False, True, True]
cigar_is_ref = [True, False, True, True, False, False, False, True, True]


def cigartuples_from_string(cigarstring):
    """ Returns pysam-style list of (op, count) tuples from a cigarstring.
    """
    pattern = re.compile(fr"(\d+)([{''.join(CODE_TO_OP.keys())}])")
    return [
        (CODE_TO_OP[m.group(2)], int(m.group(1)))
        for m in re.finditer(pattern, cigarstring)
    ]


def seq_lens(cigartuples):
    """ Length of query and reference sequences from cigar tuples.  """
    if not len(cigartuples):
        return 0, 0
    ops, counts = np.array(cigartuples).T
    q_len = counts[cigar_is_query[ops]].sum()
    r_len = counts[cigar_is_ref[ops]].sum()
    return q_len, r_len


def trim_while(cigar, from_end=False):
    """ Trim cigartuples until predicate is not satisfied.
    """
    def trim_func(c_op_len, num_match=11):
        return (c_op_len[1] < num_match) or (c_op_len[0] != pysam.CEQUAL)

    cigar_trim = (
        list(takewhile(trim_func, reversed(cigar)))[::-1]
        if from_end else
        list(takewhile(trim_func, cigar))
    )
    if len(cigar_trim):
        cigar = (
            cigar[:-len(cigar_trim)]
            if from_end else
            cigar[len(cigar_trim):]
        )
    q_trim, r_trim = seq_lens(cigar_trim)
    return cigar, q_trim, r_trim


def edlib_adj_align(query, ref, num_match=11):
    def find_first(predicate, seq):
        return next((i for i, x in enumerate(seq) if predicate(x)), None)

    def long_match(c_op_len, num_match=11):
        return (c_op_len[0] == pysam.CEQUAL) and (c_op_len[1] >= num_match)

    def concat(*cigars):
        cigars = [c for c in cigars if len(c)]
        for c1, c2 in zip(cigars[:-1], cigars[1:]):
            (o1, n1), (o2, n2) = c1[-1], c2[0]
            if o1 == o2:
                c1[-1] = (o1, 0)
                c2[0] = (o2, n1 + n2)
        return [(o, n) for c in cigars for (o, n) in c if n]

    def parasail_align(query, ref):
        return cigartuples_from_string(
            sg_trace_scan_32(query, ref, 10, 2, dnafull).cigar.decode.decode()
        )

    # compute full read cigar with edlib
    cigar = cigartuples_from_string(
        edlib_align(query, ref, task="path")["cigar"]
    )

    # find first and last long matches and fix up alignments with parasail
    flm_idx = find_first(long_match, cigar)
    if flm_idx is None:
        return parasail_align(query, ref)
    if flm_idx > 0:
        q_start, r_start = seq_lens(cigar[:flm_idx + 1])
        cigar = concat(
            parasail_align(query[:q_start], ref[:r_start]),
            cigar[flm_idx + 1:]
        )
    llm_idx = find_first(long_match, reversed(cigar))
    if llm_idx is None:
        return parasail_align(query, ref)
    if llm_idx > 0:
        q_end, r_end = seq_lens(cigar[-(llm_idx + 1):])
        cigar = concat(
            cigar[:-(llm_idx + 1)],
            parasail_align(query[-q_end:], ref[-r_end:])
        )

    return cigar


def trimmed_alignment(seq1, seq2):
    cigar = edlib_adj_align(seq1, seq2, num_match=11)
    cigar, s1, s2 = trim_while(cigar)
    cigar, e1, e2 = trim_while(cigar, from_end=True)

    if len(cigar) == 0:
        return None
    cigar_ops, cigar_counts = zip(*cigar)
    return {
        "cigar_ops": cigar_ops,
        "cigar_counts": cigar_counts,
        "trimmed_bases_query": np.array([s1, e1], dtype=np.uint32),
        "trimmed_bases_ref": np.array([s2, e2], dtype=np.uint32),
    }


def call_basespace_duplex(seq1, seq2, qscores1, qscores2):
    seq2 = revcomp(seq2)
    res = trimmed_alignment(seq1, seq2)
    res
