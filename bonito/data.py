import os
from pathlib import Path

import bonito_datasets.dataset as bd_dataset
import numpy as np
import toml
import torch
from datasets.arrow import get_dataframe, load_basecall_df
from torch.utils.data import DataLoader


class ChunkDataSet:
    def __init__(self, chunks, targets, lengths):
        self.chunks = np.expand_dims(chunks, axis=1)
        self.targets = targets
        self.lengths = lengths

    def __getitem__(self, i):
        return (
            self.chunks[i].astype(np.float32),
            self.targets[i].astype(np.int64),
            self.lengths[i].astype(np.int64),
        )

    def __len__(self):
        return len(self.lengths)


class WeightedConcatDatasetRandomSampler(torch.utils.data.Sampler):
    """
    Samples ``num_samples`` from ``datasets``, with replacement, where weights
    specifies the ratio of samples to draw from each dataset. Returned indices
    will be suitable for ``torch.utils.data.ConcatDataset(datasets)``.

    Args:
        datasets: datasets to sample from.

        weights: ratio of samples to draw from each dataset. Will be normalized
            internally to sum to 1.

        num_samples: number of samples to draw.

        seed: the random seed.

        inter_iter_randomness: if False, each call to ``__iter__`` is
            guaranteed to return an ``Iterator`` that will produce the **same**
            sequence of indices.  If True, then each call to ``__iter__`` will
            produce, to some very high probability, a different sequence of
            indices.
    """
    def __init__(self, datasets, weights, num_samples, seed, inter_iter_randomness=True):
        self.datasets = datasets
        self.weights = np.array(weights)
        self.weights = self.weights / self.weights.sum()
        self.num_samples = num_samples
        self.offsets = np.cumsum([0] + [len(ds) for ds in self.datasets][:-1])
        self.seed = seed
        self.inter_iter_randomness = inter_iter_randomness

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(seed=self.seed)
        if self.inter_iter_randomness:
            self.seed += 1
        for _ in range(self.num_samples):
            ds_idx = rng.choice(len(self.datasets), p=self.weights)
            val_idx = rng.integers(0, len(self.datasets[ds_idx]))
            yield self.offsets[ds_idx] + val_idx


def load_numpy(limit, directory, batch):
    """
    Returns training and validation DataLoaders for data in directory.
    """
    train_data = load_numpy_datasets(limit=limit, directory=directory)
    if os.path.exists(os.path.join(directory, 'validation')):
        valid_data = load_numpy_datasets(
            directory=os.path.join(directory, 'validation')
        )
    else:
        print("[validation set not found: splitting training set]")
        split = np.floor(len(train_data[0]) * 0.97).astype(np.int32)
        valid_data = [x[split:] for x in train_data]
        train_data = [x[:split] for x in train_data]

    dl_args = {"batch_size": batch, "num_workers": 4, "pin_memory": True}
    train_loader = DataLoader(ChunkDataSet(*train_data), **dl_args, shuffle=True)
    valid_loader = DataLoader(ChunkDataSet(*valid_data), **dl_args)
    return train_loader, valid_loader


def load_bonito_datasets(config, batch, train_chunks, valid_chunks, train_seed, valid_sample_seed=42):
    """
    Returns training and validation DataLoaders for data in config.
    """
    dataset_config = toml.load(config)

    (t_ds, t_weights), (v_ds, v_weights) = load_bd_datasets(
        dataset_config, valid_sample_seed
    )

    train_sampler = WeightedConcatDatasetRandomSampler(
        t_ds, t_weights, num_samples=train_chunks, seed=train_seed
    )
    valid_sampler = WeightedConcatDatasetRandomSampler(
        v_ds, v_weights, num_samples=valid_chunks, seed=valid_sample_seed,
        inter_iter_randomness=False
    )

    dl_args = {
        "batch_size": batch, "num_workers": 16, "pin_memory": True,
        "collate_fn": bd_dataset.collate_fn
    }
    train_loader = DataLoader(
        torch.utils.data.ConcatDataset(t_ds),
        **dl_args,
        sampler=train_sampler,
        drop_last=True,
        prefetch_factor=4,
    )
    valid_loader = DataLoader(
        torch.utils.data.ConcatDataset(v_ds),
        **dl_args,
        sampler=valid_sampler,
    )
    return train_loader, valid_loader


def load_numpy_datasets(limit=None, directory=None):
    """
    Returns numpy chunks, targets and lengths arrays.
    """
    if directory is None:
        directory = default_data

    chunks = np.load(os.path.join(directory, "chunks.npy"), mmap_mode='r')
    targets = np.load(os.path.join(directory, "references.npy"), mmap_mode='r')
    lengths = np.load(os.path.join(directory, "reference_lengths.npy"), mmap_mode='r')

    indices = os.path.join(directory, "indices.npy")

    if os.path.exists(indices):
        idx = np.load(indices, mmap_mode='r')
        idx = idx[idx < lengths.shape[0]]
        if limit:
            idx = idx[:limit]
        return chunks[idx, :], targets[idx, :], lengths[idx]

    if limit:
        chunks = chunks[:limit]
        targets = targets[:limit]
        lengths = lengths[:limit]

    return np.array(chunks), np.array(targets), np.array(lengths)


def load_bd_datasets(config, valid_sample_seed):
    """
    Returns (datasets, weights) tuples for training and validation.
    """
    train_data = load_subsets_data(config["train"].items(), config["params"])

    if "valid" in config:
        valid_data = load_subsets_data(config["valid"].items(), config["params"])
    else:
        valid_data = []
        for i in range(len(train_data)):
            df = train_data[i][0]
            train_df = df.sample(frac=0.97, random_state=valid_sample_seed)
            valid_df = df.drop(train_df.index)
            train_data[i][0] = train_df
            valid_data.append(
                [valid_df, train_data[i][1], train_data[i][2]]
            )

    train_ds = []
    train_weights = []
    for data, params, weight in train_data:
        train_ds.append(bd_dataset.ChunkDataset(**params, data=data))
        train_weights.append(weight)

    valid_ds = []
    valid_weights = []
    for data, params, weight in valid_data:
        valid_ds.append(bd_dataset.ChunkDataset(**params, data=data))
        valid_weights.append(weight)

    return (train_ds, train_weights), (valid_ds, valid_weights)


def load_subsets_data(subset_configs, params=None):
    subset_data = []
    params = {} if params is None else params.copy()
    for name, dataset_config in subset_configs:
        print(f"[loading '{name}']")
        df = load_df(dataset_config)
        params.update(dataset_config.get("params", {}))
        weight = dataset_config.get("weight", 1)
        subset_data.append([df, params, weight])
        print(f"[{name} params: {params}]")
        print(f"[{name} weighting: {weight}]")
    return subset_data


def load_df(dataset_config):
    if "signal_paths" in dataset_config:
        print("[using signal_paths from config not basecall table]")
        basecall_df = get_dataframe(dataset_config["basecall_path"])
        signal_df = get_dataframe(dataset_config["signal_paths"])
        df = basecall_df.merge(signal_df, on="read_id", how="inner")
    else:
        print("[using signal_paths from basecall table]")
        df = load_basecall_df(
            dataset_config["basecall_path"], load_signal=True
        )

    if "alignment_path" in dataset_config:
        alignment_path = dataset_config["alignment_path"]
    else:
        print("[using 'alignments.arrow' in 'basecall_path' directory]")
        alignment_path = Path(dataset_config["basecall_path"]).parent / "alignments.arrow"
    alignment_table = get_dataframe(alignment_path)
    alignment_table = alignment_table[alignment_table["alignment_is_primary"]]

    reference_table = get_dataframe(dataset_config["reference_paths"])

    df = (
        df
        .merge(alignment_table, on="read_id", how="inner")
        .merge(
            reference_table.set_index('name'),
            left_on='alignment_genome',
            right_index=True,
            how='left'
        )
        .set_index('read_id')
    )

    return df
