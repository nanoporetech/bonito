"""
Bonito dataset evaluator
Produces detailed statistics of evaluation of a bonito model on a DataLoader
"""

import re
import textwrap
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
import parasail
import pandas as pd
from tqdm import tqdm

from bonito.util import decode_ref, init, load_model, permute
from bonito.data import load_data, ComputeSettings, DataSettings, ModelSetup


@dataclass
class AlignResult:
    accuracy: float = 0
    num_correct: int = 0
    num_mismatches: int = 0
    num_insertions: int = 0
    num_deletions: int = 0
    ref_len: int = 0
    seq_len: int = 0
    align_ref_start: int = 0
    align_ref_end: int = 0
    align_seq_start: int = 0
    align_seq_end: int = 0


def align(*, ref, seq):
    if not seq:
        return AlignResult()

    res = parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)

    cigar = res.cigar.decode.decode()
    counts = defaultdict(int)
    for cnt, op in re.findall(r"(\d+)([A-Z\W])", cigar):
        counts[op] += int(cnt)

    # Sometimes parasail.SW will start with xD and we need to handle this explicitly
    del_start = int(match[0]) if (match := re.findall(r"^(\d+)D", cigar)) else 0
    counts['D'] -= del_start

    ref_start = res.end_ref - counts['='] - counts['X'] - counts['D'] + 1
    seq_start = res.end_query - counts['='] - counts['X'] - counts['I'] + 1

    return AlignResult(
        accuracy=counts["="] / sum(counts.values()),
        num_correct=counts["="],
        num_mismatches=counts["X"],
        num_insertions=counts["I"],
        num_deletions=counts["D"],
        ref_len=len(ref),
        seq_len=len(seq),
        align_ref_start=ref_start,
        align_ref_end=res.end_ref,
        align_seq_start=seq_start,
        align_seq_end=res.end_query,
    )


def main(args):
    init(args.seed, args.device)

    print(f"* loading model from: {args.model_directory}/weights_{args.weights}.tar")
    model = load_model(args.model_directory, args.device, weights=args.weights)
    standardisation = model.config.get("standardisation", {}) if args.standardise else {}
    model_setup = ModelSetup(
        n_pre_context_bases=getattr(model, "n_pre_context_bases", 0),
        n_post_context_bases=getattr(model, "n_post_context_bases", 0),
        standardisation=standardisation,
    )
    mean = model_setup.standardisation.get("mean", 0.0)
    stdev = model_setup.standardisation.get("stdev", 1.0)
    print(f"* * applying standardisation params: mean={mean}, stdev={stdev}")

    print("* loading data")
    compute_settings = ComputeSettings(batch_size=args.batchsize, num_workers=4, seed=args.seed)
    if args.dataset == "valid":
        # Valid-data is often a subset of train data, so we need to provide enough
        # train-chunks to subset from. We will never actually load these.
        data = DataSettings(args.directory, args.chunks * 100, args.chunks, None)
        _, dataloader = load_data(data, model_setup, compute_settings)
    else:

        data = DataSettings(args.directory, args.chunks, args.chunks, None)
        dataloader, _ = load_data(data, model_setup, compute_settings)

    print("* calling")
    seqs = []
    targets = []

    with torch.no_grad():
        for data, target, *_ in tqdm(dataloader, total=args.chunks // args.batchsize):
            targets.extend(torch.unbind(target, 0))
            data = data.type(torch.float16).to(args.device)
            log_probs = model(data)

            if hasattr(model, 'decode_batch'):
                seqs.extend(model.decode_batch(log_probs))
            else:
                seqs.extend([model.decode(p) for p in permute(log_probs, 'TNC', 'NTC')])

    refs = [decode_ref(target, model.alphabet) for target in targets]
    alignments = pd.DataFrame([align(ref=ref, seq=seq) for ref, seq in zip(refs, seqs)])

    print("* aligning")

    print(textwrap.dedent(f"""
        * num_chunks      {len(alignments)}
        * accuracy        {alignments.accuracy.mean():.2%}
        * sub-rate        {(alignments.num_mismatches / alignments.num_correct).mean():.2%}
        * ins-rate        {(alignments.num_insertions / alignments.num_correct).mean():.2%}
        * del-rate        {(alignments.num_deletions / alignments.num_correct).mean():.2%}
        * seq_len         {alignments.seq_len.mean():.1f}
        * seq_lclip       {alignments.align_seq_start.mean():.1f}
        * seq_rclip       {(alignments.seq_len - alignments.align_seq_end - 1).mean():.1f}
        * ref_len         {alignments.ref_len.mean():.1f}
        * ref_lclip       {alignments.align_ref_start.mean():.1f}
        * ref_rclip       {(alignments.ref_len - alignments.align_ref_end - 1).mean():.1f}
        """))

    if args.output_dir:
        args.output_dir.mkdir(exist_ok=True, parents=True)
        with (args.output_dir / "seqs.fasta").open("w") as fh:
            fh.write("".join([f">chunk_{i}\n{s}\n" for i, s in enumerate(seqs)]))
        with (args.output_dir / "refs.fasta").open("w") as fh:
            fh.write("".join([f">chunk_{i}\n{s}\n" for i, s in enumerate(refs)]))
        alignments.to_csv(args.output_dir / "summ.txt", sep="\t")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--dataset", choices=["train", "valid"], default="valid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=9, type=int)
    parser.add_argument("--weights", default=0, type=None)
    parser.add_argument("--chunks", default=512, type=int)
    parser.add_argument("--batchsize", default=256, type=int)
    parser.add_argument("--standardise", action="store_true", default=False)
    return parser

