"""
Bonito utils
"""

import os
import re
import sys
import random
from glob import glob
from itertools import groupby
from operator import itemgetter
from importlib import import_module
from collections import deque, defaultdict, OrderedDict
from torch.utils.data import DataLoader

import toml
import torch
import koi.lstm
import parasail
import numpy as np
from torch.cuda import get_device_capability

try:
    from claragenomics.bindings import cuda
    from claragenomics.bindings.cudapoa import CudaPoaBatch
except ImportError:
    pass


__dir__ = os.path.dirname(os.path.realpath(__file__))
__data__ = os.path.join(__dir__, "data")
__models__ = os.path.join(__dir__, "models")
__configs__ = os.path.join(__dir__, "models/configs")

split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")
default_data = os.path.join(__data__, "dna_r9.4.1")
default_config = os.path.join(__configs__, "dna_r9.4.1@v3.1.toml")


def init(seed, device, deterministic=True):
    """
    Initialise random libs and setup cudnn

    https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cpu": return
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = (not deterministic)
    assert(torch.cuda.is_available())


def permute(x, input_layout, output_layout):
    """
    Permute `x` from `input_layout` to `output_layout`

    >>> permute(x, 'TNC', 'NTC')
    """
    if input_layout == output_layout: return x
    return x.permute(*[input_layout.index(x) for x in output_layout])


def concat(xs, dim=0):
    """
    Type agnostic concat.
    """
    if isinstance(xs[0], torch.Tensor):
        return torch.cat(xs, dim=dim)
    elif isinstance(xs[0], np.ndarray):
        return np.concatenate(xs, axis=dim)
    elif isinstance(xs[0], list):
        return [x for l in xs for x in l]
    elif isinstance(xs[0], str):
        return ''.join(xs)
    elif isinstance(xs[0], dict):
        return {k: concat([x[k] for x in xs], dim) for k in xs[0].keys()}
    else:
        raise TypeError


def select_range(x, start, end, dim=0):
    """
    Type agnostic range select.
    """
    if isinstance(x, dict):
        return {k: select_range(v, start, end, dim) for (k, v) in x.items()}
    if dim == 0 or isinstance(x, list): return x[start:end]
    return x[(*(slice(None),)*dim, slice(start, end))]


def size(x, dim=0):
    """
    Type agnostic size.
    """
    if hasattr(x, 'shape'):
        return x.shape[dim]
    elif dim == 0:
        return len(x)
    raise TypeError


def half_supported():
    """
    Returns whether FP16 is support on the GPU
    """
    try:
        return get_device_capability()[0] >= 7
    except:
        return False


def phred(prob, scale=1.0, bias=0.0):
    """
    Converts `prob` into a ascii encoded phred quality score between 0 and 40.
    """
    p = max(1 - prob, 1e-4)
    q = -10 * np.log10(p) * scale + bias
    return chr(int(np.round(q) + 33))


def mean_qscore_from_qstring(qstring):
    """
    Convert qstring into a mean qscore
    """
    if len(qstring) == 0: return 0.0
    qs = (np.array(qstring, 'c').view(np.uint8) - 33)
    mean_err = np.exp(qs * (-np.log(10) / 10.)).mean()
    return -10 * np.log10(max(mean_err, 1e-4))


def decode_ref(encoded, labels):
    """
    Convert a integer encoded reference into a string and remove blanks
    """
    return ''.join(labels[e] for e in encoded.tolist() if e)


def column_to_set(filename, idx=0, skip_header=False):
    """
    Pull a column from a file and return a set of the values.
    """
    if filename and os.path.isfile(filename):
        with open(filename, 'r') as tsv:
            if skip_header:
                next(tsv)
            return {line.strip().split()[idx] for line in tsv.readlines()}


def chunk(signal, chunksize, overlap):
    """
    Convert a read into overlapping chunks before calling
    """
    T = signal.shape[0]
    if chunksize == 0:
        chunks = signal[None, :]
    elif T < chunksize:
        n, overhang = divmod(chunksize, T)
        chunks = torch.cat((signal.repeat(n), signal[:overhang]))[None, :]
    else:
        stub = (T - overlap) % (chunksize - overlap)
        chunks = signal[stub:].unfold(0, chunksize, chunksize - overlap)
        if stub > 0:
            chunks = torch.cat([signal[None, :chunksize], chunks], dim=0)
    return chunks.unsqueeze(1)


def stitch(chunks, chunksize, overlap, length, stride, reverse=False):
    """
    Stitch chunks together with a given overlap
    """
    if chunks.shape[0] == 1: return chunks.squeeze(0)

    semi_overlap = overlap // 2
    start, end = semi_overlap // stride, (chunksize - semi_overlap) // stride
    stub = (length - overlap) % (chunksize - overlap)
    first_chunk_end = (stub + semi_overlap) // stride if (stub > 0) else end

    if reverse:
        chunks = list(chunks)
        return concat([
            chunks[-1][:-start], *(x[-end:-start] for x in reversed(chunks[1:-1])), chunks[0][-first_chunk_end:]
        ])
    else:
        return concat([
            chunks[0, :first_chunk_end], *chunks[1:-1, start:end], chunks[-1, start:]
        ])


def batchify(items, batchsize, dim=0):
    """
    Batch up items up to `batch_size`.
    """
    stack, pos = [], 0
    for k, v in items:
        breaks = range(batchsize - pos, size(v, dim), batchsize)
        for start, end in zip([0, *breaks], [*breaks, size(v, dim)]):
            sub_batch = select_range(v, start, end, dim)
            stack.append(((k, (pos, pos + end - start)), sub_batch))
            if pos + end - start == batchsize:
                ks, vs = zip(*stack)
                yield ks, concat(vs, dim)
                stack, pos = [], 0
            else:
                pos += end - start

    if len(stack):
        ks, vs = zip(*stack)
        yield ks, concat(vs, dim)


def unbatchify(batches, dim=0):
    """
    Reconstruct batches.
    """
    batches = (
        (k, select_range(v, start, end, dim))
        for sub_batches, v in batches
        for k, (start, end) in sub_batches
    )
    return (
        (k, concat([v for (k, v) in group], dim))
        for k, group in groupby(batches, itemgetter(0))
    )


def load_symbol(config, symbol):
    """
    Dynamic load a symbol from module specified in model config.
    """
    if not isinstance(config, dict):
        if not os.path.isdir(config) and os.path.isdir(os.path.join(__models__, config)):
            dirname = os.path.join(__models__, config)
        else:
            dirname = config
        config = toml.load(os.path.join(dirname, 'config.toml'))
    imported = import_module(config['model']['package'])
    return getattr(imported, symbol)


def match_names(state_dict, model):
    keys_and_shapes = lambda state_dict: zip(*[
        (k, s) for s, i, k in sorted([(v.shape, i, k)
        for i, (k, v) in enumerate(state_dict.items())])
    ])
    k1, s1 = keys_and_shapes(state_dict)
    k2, s2 = keys_and_shapes(model.state_dict())
    assert s1 == s2
    remap = dict(zip(k1, k2))
    return OrderedDict([(k, remap[k]) for k in state_dict.keys()])


def get_last_checkpoint(dirname):
    weight_files = glob(os.path.join(dirname, "weights_*.tar"))
    if not weight_files:
        raise FileNotFoundError("no model weights found in '%s'" % dirname)
    weights = max([int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files])
    return os.path.join(dirname, 'weights_%s.tar' % weights)


def set_config_defaults(config, chunksize=None, batchsize=None, overlap=None, quantize=False):
    basecall_params = config.get("basecaller", {})
    # use `value or dict.get(key)` rather than `dict.get(key, value)` to make
    # flags override values in config
    basecall_params["chunksize"] = chunksize or basecall_params.get("chunksize", 4000)
    basecall_params["overlap"] = overlap or basecall_params.get("overlap", 500)
    basecall_params["batchsize"] = batchsize or basecall_params.get("batchsize", 64)
    basecall_params["quantize"] = basecall_params.get("quantize") if quantize is None else quantize
    config["basecaller"] = basecall_params
    return config


def load_model(dirname, device, weights=None, half=None, chunksize=None, batchsize=None, overlap=None, quantize=False, use_koi=False):
    """
    Load a model config and weights off disk from `dirname`.
    """
    if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
        dirname = os.path.join(__models__, dirname)
    weights = get_last_checkpoint(dirname) if weights is None else os.path.join(dirname, 'weights_%s.tar' % weights)
    config = toml.load(os.path.join(dirname, 'config.toml'))
    config = set_config_defaults(config, chunksize, batchsize, overlap, quantize)
    return _load_model(weights, config, device, half, use_koi)


def _load_model(model_file, config, device, half=None, use_koi=False):
    device = torch.device(device)
    Model = load_symbol(config, "Model")
    model = Model(config)

    if config["model"]["package"] == "bonito.crf" and use_koi:
        model.encoder = koi.lstm.update_graph(
            model.encoder,
            batchsize=config["basecaller"]["batchsize"],
            chunksize=config["basecaller"]["chunksize"] // model.stride,
            quantize=config["basecaller"]["quantize"],
        )

    state_dict = torch.load(model_file, map_location=device)
    state_dict = {k2: state_dict[k1] for k1, k2 in match_names(state_dict, model).items()}
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    if half is None:
        half = half_supported()

    if half: model = model.half()
    model.eval()
    model.to(device)
    return model


def parasail_to_sam(result, seq):
    """
    Extract reference start and sam compatible cigar string.

    :param result: parasail alignment result.
    :param seq: query sequence.

    :returns: reference start coordinate, cigar string.
    """
    cigstr = result.cigar.decode.decode()
    first = re.search(split_cigar, cigstr)

    first_count, first_op = first.groups()
    prefix = first.group()
    rstart = result.cigar.beg_ref
    cliplen = result.cigar.beg_query

    clip = '' if cliplen == 0 else '{}S'.format(cliplen)
    if first_op == 'I':
        pre = '{}S'.format(int(first_count) + cliplen)
    elif first_op == 'D':
        pre = clip
        rstart = int(first_count)
    else:
        pre = '{}{}'.format(clip, prefix)

    mid = cigstr[len(prefix):]
    end_clip = len(seq) - result.end_query - 1
    suf = '{}S'.format(end_clip) if end_clip > 0 else ''
    new_cigstr = ''.join((pre, mid, suf))
    return rstart, new_cigstr


def accuracy(ref, seq, balanced=False, min_coverage=0.0):
    """
    Calculate the accuracy between `ref` and `seq`
    """
    alignment = parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)
    counts = defaultdict(int)

    q_coverage = len(alignment.traceback.query) / len(seq)
    r_coverage = len(alignment.traceback.ref) / len(ref)

    if r_coverage < min_coverage:
        return 0.0

    _, cigar = parasail_to_sam(alignment, seq)

    for count, op  in re.findall(split_cigar, cigar):
        counts[op] += int(count)

    if balanced:
        accuracy = (counts['='] - counts['I']) / (counts['='] + counts['X'] + counts['D'])
    else:
        accuracy = counts['='] / (counts['='] + counts['I'] + counts['X'] + counts['D'])
    return accuracy * 100


def print_alignment(ref, seq):
    """
    Print the alignment between `ref` and `seq`
    """
    alignment = parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)
    print(alignment.traceback.ref)
    print(alignment.traceback.comp)
    print(alignment.traceback.query)

    print("  Score=%s" % alignment.score)
    return alignment.score


def poa(groups, max_poa_sequences=100, gpu_mem_per_batch=0.9):
    """
    Generate consensus for POA groups.

    Args:
        groups : A list of lists of sequences for which consensus is to be generated.
    """
    free, total = cuda.cuda_get_mem_info(cuda.cuda_get_device())
    gpu_mem_per_batch *= free
    batch = CudaPoaBatch(max_poa_sequences, gpu_mem_per_batch, stream=None, output_type="consensus")
    results = []

    for i, group in enumerate(groups, start=1):
        group_status, seq_status = batch.add_poa_group(group)

        # Once batch is full, run POA processing
        if group_status == 1 or i == len(groups):
            batch.generate_poa()

            consensus, coverage, status = batch.get_consensus()
            results.extend(consensus)

            batch.reset()
            group_status, seq_status = batch.add_poa_group(group)

    return results
