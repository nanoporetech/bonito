"""
Bonito utils
"""

import os
import re
import sys
import random
from glob import glob
from collections import deque, defaultdict, OrderedDict

from bonito.model import Model
from bonito_cuda_runtime import CuModel

import toml
import torch
import parasail
import numpy as np
from scipy.signal import find_peaks
from torch.cuda import get_device_capability
from ont_fast5_api.fast5_interface import get_fast5_file

try:
    from claragenomics.bindings import cuda
    from claragenomics.bindings.cudapoa import CudaPoaBatch
except ImportError:
    pass


__dir__ = os.path.dirname(os.path.realpath(__file__))
__data__ = os.path.join(__dir__, "data")
__models__ = os.path.join(__dir__, "models")
__configs__ = os.path.join(__models__, "configs")
__url__ = "https://nanoporetech.box.com/shared/static/"

split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")
default_data = os.path.join(__data__, "dna_r9.4.1")
default_config = os.path.join(__configs__, "dna_r9.4.1.toml")


class Read:

    def __init__(self, read, filename):

        self.read_id = read.read_id
        self.run_id = read.get_run_id().decode()
        self.filename = os.path.basename(read.filename)

        read_attrs = read.handle[read.raw_dataset_group_name].attrs
        channel_info = read.handle[read.global_key + 'channel_id'].attrs

        self.offset = int(channel_info['offset'])
        self.sampling_rate = channel_info['sampling_rate']
        self.scaling = channel_info['range'] / channel_info['digitisation']

        self.mux = read_attrs['start_mux']
        self.channel = channel_info['channel_number'].decode()
        self.start = read_attrs['start_time'] / self.sampling_rate
        self.duration = read_attrs['duration'] / self.sampling_rate

        # no trimming
        self.template_start = self.start
        self.template_duration = self.duration

        raw = read.handle[read.raw_dataset_name][:]
        scaled = np.array(self.scaling * (raw + self.offset), dtype=np.float32)
        self.signal = norm_by_noisiest_section(scaled)


def init(seed, device):
    """
    Initialise random libs and setup cudnn

    https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cpu": return
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert(torch.cuda.is_available())


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
    err_probs = [10**((ord(c) - 33) / -10) for c in qstring]
    mean_err = np.mean(err_probs)
    return -10 * np.log10(max(mean_err, 1e-4))


def decode_ref(encoded, labels):
    """
    Convert a integer encoded reference into a string and remove blanks
    """
    return ''.join(labels[e] for e in encoded if e)


def med_mad(x, factor=1.4826):
    """
    Calculate signal median and median absolute deviation
    """
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor
    return med, mad


def norm_by_noisiest_section(signal, samples=100, threshold=6.0):
    """
    Normalise using the medmad from the longest continuous region where the
    noise is above some threshold relative to the std of the full signal.
    """
    threshold = signal.std() / threshold
    noise = np.ones(signal.shape)

    for idx in np.arange(signal.shape[0] // samples):
        window = slice(idx * samples, (idx + 1) * samples)
        noise[window] = np.where(signal[window].std() > threshold, 1, 0)

    # start and end low for peak finding
    noise[0] = 0; noise[-1] = 0
    peaks, info = find_peaks(noise, width=(None, None))

    if len(peaks):
        widest = np.argmax(info['widths'])
        med, mad = med_mad(signal[info['left_bases'][widest]: info['right_bases'][widest]])
    else:
        med, mad = med_mad(signal)
    return (signal - med) / mad


def get_raw_data(filename, read_ids=None, skip=False):
    """
    Get the raw signal and read id from the fast5 files
    """
    with get_fast5_file(filename, 'r') as f5_fh:
        for read_id in f5_fh.get_read_ids():
            if read_ids is None or (read_id in read_ids) ^ skip:
                yield Read(f5_fh.get_read(read_id), filename)


def get_raw_data_for_read(filename, read_id):
    """
    Get the raw signal from the fast5 file for a given read_id
    """
    with get_fast5_file(filename, 'r') as f5_fh:
        return Read(f5_fh.get_read(read_id), filename)


def get_reads(directory, read_ids=None, skip=False, max_read_size=4e6):
    """
    Get all reads in a given `directory`.
    """
    for fast5 in glob("%s/*fast5" % directory):
        for read in get_raw_data(fast5, read_ids=read_ids, skip=skip):
            if len(read.signal) > max_read_size:
                sys.stderr.write("> skipping long read %s (%s samples)\n" % (read.read_id, len(read.signal)))
                continue
            yield read


def chunk(signal, chunksize, overlap):
    """
    Convert a read into overlapping chunks before calling
    """
    T = signal.shape[0]
    if chunksize > 0:
        padded_size = T + (overlap - T) % (chunksize - overlap)
        padded = torch.nn.functional.pad(signal, (0, padded_size - T))
        return padded.unfold(0, chunksize, chunksize - overlap).unsqueeze(1)
    return signal.unsqueeze(0).unsqueeze(0)


def stitch(predictions, overlap, stride=1):
    """
    Stitch predictions together with a given overlap
    """
    overlap = overlap // stride // 2
    if predictions.shape[0] == 1:
        return predictions.squeeze(0)
    stitched = [predictions[0, 0:-overlap]]
    for i in range(1, predictions.shape[0] - 1): stitched.append(predictions[i][overlap:-overlap])
    stitched.append(predictions[-1][overlap:])
    return np.concatenate(stitched)


def batch_reads(reads, chunksize=0, overlap=0, batchsize=1):
    """
    Convert reads into stream in fixed size chunks.
    """
    stack = []; index = defaultdict(list)

    for read in reads:

        chunks = chunk(torch.tensor(read.signal), chunksize, overlap)
        index['reads'].append(read)
        index['chunks'].append(chunks.shape[0])

        stack = torch.cat((stack, chunks)) if len(stack) else chunks
        *batches, stack = torch.split(stack, batchsize)

        if len(batches) or chunksize == 0:
            yield batches, index
            index = defaultdict(list)

    if len(stack):
        yield (stack, ), index


def unbatch_reads(batches, overlap=0, stride=1):
    """
    Split batched reads
    """
    stack = []
    reads = deque()
    num_chunks = deque()

    for batch, index in batches:

        reads.extend(index['reads'])
        num_chunks.extend(index['chunks'])
        stack = torch.cat((stack, batch)) if len(stack) else batch

        while num_chunks and stack.shape[0] >= num_chunks[0]:

            n = num_chunks.popleft()
            read = reads.popleft()
            chunks, stack = torch.split(stack, [n, stack.shape[0] - n])

            stitched = stitch(chunks.numpy(), overlap, stride)
            yield read, stitched[:read.signal.shape[0] // stride].astype(np.float32)


def column_to_set(filename, idx=0, skip_header=False):
    """
    Pull a column from a file and return a set of the values.
    """
    if filename and os.path.isfile(filename):
        with open(filename, 'r') as tsv:
            if skip_header:
                next(tsv)
            return {line.strip().split()[idx] for line in tsv.readlines()}


def load_data(shuffle=False, limit=None, directory=None, validation=False):
    """
    Load the training data
    """
    if directory is None:
        directory = default_data

    if validation and os.path.exists(os.path.join(directory, 'validation')):
        directory = os.path.join(directory, 'validation')

    chunks = np.load(os.path.join(directory, "chunks.npy"))
    chunk_lengths = np.load(os.path.join(directory, "chunk_lengths.npy"))
    targets = np.load(os.path.join(directory, "references.npy"))
    target_lengths = np.load(os.path.join(directory, "reference_lengths.npy"))

    if shuffle:
        state = np.random.get_state()
        for array in (chunks, chunk_lengths, targets, target_lengths):
            np.random.set_state(state)
            np.random.shuffle(array)
    if limit:
        chunks = chunks[:limit]
        chunk_lengths = chunk_lengths[:limit]
        targets = targets[:limit]
        target_lengths = target_lengths[:limit]

    return chunks, chunk_lengths, targets, target_lengths


def load_model(dirname, device, weights=None, half=None, chunksize=0, use_rt=False):
    """
    Load a model from disk
    """
    if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
        dirname = os.path.join(__models__, dirname)

    if not weights: # take the latest checkpoint
        weight_files = glob(os.path.join(dirname, "weights_*.tar"))
        if not weight_files:
            raise FileNotFoundError("no model weights found in '%s'" % dirname)
        weights = max([int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files])

    device = torch.device(device)
    config = os.path.join(dirname, 'config.toml')
    weights = os.path.join(dirname, 'weights_%s.tar' % weights)
    model = Model(toml.load(config))

    state_dict = torch.load(weights, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    if use_rt:
        model = CuModel(model.config, chunksize, new_state_dict)

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
    _, cigar = parasail_to_sam(alignment, seq)

    q_coverage = len(alignment.traceback.query) / len(seq)
    r_coverage = len(alignment.traceback.ref) / len(ref)

    if r_coverage < min_coverage:
        return 0.0

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


def poa(groups, max_sequences_per_poa=100, gpu_mem_per_batch=0.9):
    """
    Generate consensus for POA groups.

    Args:
        groups : A list of lists of sequences for which consensus is to be generated.
    """
    free, total = cuda.cuda_get_mem_info(cuda.cuda_get_device())
    gpu_mem_per_batch *= free
    batch = CudaPoaBatch(max_sequences_per_poa, gpu_mem_per_batch, stream=None, output_type="consensus")
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
