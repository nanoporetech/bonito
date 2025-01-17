import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from torch.utils.data import DataLoader


@dataclass
class DataSettings:
    training_data: Path
    num_train_chunks: int
    num_valid_chunks: int
    output_dir: Path

@dataclass
class ComputeSettings:
    batch_size: int
    num_workers: int
    seed: int
    pin_memory: bool = True

@dataclass
class ModelSetup:
    n_pre_context_bases: int
    n_post_context_bases: int
    standardisation: Dict


def load_data(data, model_setup, compute_settings):
    try:
        if (Path(data.training_data) / "chunks.npy").exists():
            print(f"[loading data] - chunks from {data.training_data}")
            train_loader_kwargs, valid_loader_kwargs = load_numpy(
                data.num_train_chunks,
                data.training_data,
                valid_chunks=data.num_valid_chunks,
            )
        elif (Path(data.training_data) / "dataset.py").exists():
            print(f"[loading data] - dynamically from {data.training_data}/dataset.py")
            train_loader_kwargs, valid_loader_kwargs = load_script(
                data.training_data,
                chunks=data.num_train_chunks,
                valid_chunks=data.num_valid_chunks,
                log_dir=data.output_dir,
                n_pre_context_bases=model_setup.n_pre_context_bases,
                n_post_context_bases=model_setup.n_post_context_bases,
                standardisation=model_setup.standardisation,
                seed=compute_settings.seed,
                batch_size=compute_settings.batch_size,
                num_workers=compute_settings.num_workers,
            )
        else:
            raise FileNotFoundError(f"No suitable training data found at: {data.training_data}")
    except Exception as e:
        raise IOError(f"Failed to load input data from {data.training_data}") from e

    default_settings = {
        "batch_size": compute_settings.batch_size,
        "num_workers": compute_settings.num_workers,
        "pin_memory": compute_settings.pin_memory,
    }

    # Allow options from the train/valid_loader to override the default_kwargs
    train_loader = DataLoader(**{**default_settings, **train_loader_kwargs})
    valid_loader = DataLoader(**{**default_settings, **valid_loader_kwargs})
    return train_loader, valid_loader


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


def load_script(directory, name="dataset", suffix=".py", **kwargs):
    directory = Path(directory)
    filepath = (directory / name).with_suffix(suffix)
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = module.Loader(**kwargs)
    return loader.train_loader_kwargs(**kwargs), loader.valid_loader_kwargs(**kwargs)


def load_numpy(limit, directory, valid_chunks=None):
    """
    Returns training and validation DataLoaders for data in directory.
    """
    train_data = load_numpy_datasets(limit=limit, directory=directory)
    if os.path.exists(os.path.join(directory, 'validation')):
        valid_data = load_numpy_datasets(limit=valid_chunks,
            directory=os.path.join(directory, 'validation')
        )
    else:
        print("[validation set not found: splitting training set]")
        if valid_chunks is None:
            split = np.floor(len(train_data[0]) * 0.97).astype(np.int32)
        else:
            split = max(0, len(train_data[0]) - valid_chunks)
        valid_data = [x[split:] for x in train_data]
        train_data = [x[:split] for x in train_data]

    train_loader_kwargs = {"dataset": ChunkDataSet(*train_data), "shuffle": True}
    valid_loader_kwargs = {"dataset": ChunkDataSet(*valid_data), "shuffle": False}
    return train_loader_kwargs, valid_loader_kwargs


def load_numpy_datasets(limit=None, directory=None):
    """
    Returns numpy chunks, targets and lengths arrays.
    """
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
