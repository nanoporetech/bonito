# Bonito model training

Bonito is a research-oriented model training tool designed for translating the raw-signals produced 
by Oxford Nanopore devices into genomic sequences.

## Should I train a custom model for my data using bonito?

For the vast majority Oxford Nanopore sequencing use cases **there is no need to train custom 
basecall models**. Oxford Nanopore releases production basecall models through the Dorado project 
(https://github.com/nanoporetech/dorado) and these are incorporated into the MinKnow device software
available on all Oxford Nanopore devices. These models are trained using a highly diverse pool of 
training samples and rigorously validated to ensure robust performance across a wide range of sample
types, making them the recommended choice for general use cases.

Model training can be a complex and resource-intensive process, requiring a good understanding of 
machine learning principles, bioinformatics, and nanopore sequencing data. While Bonito provides 
tools to facilitate model training, it is important to note that we are unable to offer extensive 
support for custom model training. Users undertaking this should be prepared to troubleshoot and 
refine their workflows independently, the examples below are intended as a starting point for 
further development.


## Training a model

Bonito models can be trained with the `bonito train` command. 
The choice of model design and hyperparameters for training will depend on the experiment and 
the data provided. Some default values are provided by Bonito, but these may need to be adjusted 
to fit your requirements. 

The required arguments are shown below, for a full argument  list including optional arguments
please see `bonito train --help` 

```
bonito train 
    {output_directory}                     # Positional argument. This is the directory that the output will be written to
    --directory {Path}                     # Path to the training data
    --epochs {int}                         # Number of epochs to train for
    --chunks {int}                         # Number of chunks read per epoch
    --valid-chunks {int}                   # Number of chunks used in validation at the end of each epoch
    ### One of the following arguments must be supplied
    --pretrained {Path}                    # Path to an existing model to finetune from. It is recommended to use a released model.
    --config {Path}                        # Path to a model definition file if training from scratch. Examples can be found in `bonito/models/configs`
```

## Input data type and structure

### Preparing training data with `--save-ctc`

The simplest way to prepare data for bonito training is using the `--save-ctc` flag with 
`bonito basecaller`:
```
bonito basecaller {initial_model} {input_data} --reference {ref.fasta} --save-ctc > basecalls.bam
```
In addition to the `basecalls.bam` output file this will generate a `chunks.npy`, `references.npy` 
and `reference_lengths.npy` file in the same output directory. This can be used immediately for 
basecaller training with `bonito train ... --directory {basecaller_output_dir}`

#### Understanding the data format

- `chunks.npy` is a numpy-array of shape `num_chunks * chunksize` where each row is a fixed size 'slice' of the raw signal data from the input pod5/fast5. 
- `references.npy` is a numpy array of shape `num_chunks * max_ref_len` where each row contains a 'slice' of the reference-sequence which the `chunk` should align to. Bases are labelled as `A=1, C=2, G=3, T=4`. Since the number of reference bases will vary between chunks the reference array is zero-padded and the width of the array is the longest reference sequence in the sample
- `reference_lengths.npy` is a numpy-array of shape `num_chunks`. This is a helper array that indicates the start-index of the zero-padding in `references.npy`

#### Combining multiple datasets
In some cases it may be necessary to make several calls to `bonito basecaller --save-ctc` and 
combine the results. In these cases the output arrays can be combined by concatenation. 

```
chunks_1 = np.load("dataset_1/chunks.npy")
chunks_2 = np.load("dataset_2/chunks.npy")
...

chunks_combined = np.concatenate([chunks_1, chunks_2, ...])
np.save("chunks.npy", chunks_combined)
```
In the case of the `references.npy` output each array must additionally be zero-padded to the 
width of the largest array. 
```
refs_1 = np.load("ctc_data/dataset_1/references.npy")
refs_2 = np.load("ctc_data/dataset_2/references.npy")[:,:1000]
...

max_width = max(refs_1.shape[1], refs_2.shape[1], ...)
refs_1_pad = np.pad(refs_1, ((0,0), (0, max_width - refs_1.shape[1])), mode='constant', constant_values=0)
refs_2_pad = np.pad(refs_2, ((0,0), (0, max_width - refs_2.shape[1])), mode='constant', constant_values=0)
... 

refs_combined = np.concatenate([refs_1_pad, refs_2_pad, ...])
np.save("references.npy", refs_combined)
```

#### Memory usage and limitations
During both the data generation with `bonito basecaller --save-ctc` and `bonito train` the entire 
dataset is loaded into RAM, which provides an upper bound to the amount of data which can be used 
for training by this method. 


### Providing data dynamically with `dataset.py` 
> :warning: WARNING: This method of feeding training data to bonito is provided for fleixbility in 
> training and will require additional user development. The examples below are illustrative only 
> and Oxford Nanopore is unable to support user implementations.

As an alternative to using the `--save-ctc` method described above data can alternatively be fed 
into Bonito training by dynamically loading a `dataset.py` file which serves chunks to the training
module. This more advanced method allows for more flexible data loading, but requires the user to 
perform several intermediate steps such as the signal-reference mapping and creation of a DataLoader. 

#### Interface
The data is loaded into bonito here: https://github.com/nanoporetech/bonito/blob/5a711385/bonito/cli/train.py#L65

- There should be a module named `dataset.py` in the folder passed to the `--directory` argument of `bonito train`
- This module should have a callable named `Loader` which instantiates a class with the methods `train_loader_kwargs` and `valid_loader_kwargs`
- The `train/valid_loader_kwargs` should return a dict with the keys: `dataset` and `sampler` or `shuffle`
- The `dataset` should provide the same interface as a `torch.utils.data.Dataset`: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
- The `sampler` should provide the same interface as a `torch.utils.data.Sampler`: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler


#### Example
This example shows how a `dataset.py` module can be used to load the default `--save-ctc` output format. It is possible to extend this method to utilise more complex datastructures for training.

```
"""
This is an example dataset.py file that can be loaded dynamically by bonito
"""

from functools import partial
from pathlib import Path

import numpy as np
from torch.utils.data import RandomSampler

from bonito.data import ChunkDataSet

class ChunkLoader:

    def __init__(self, train_data, valid_data, **kwargs):
        self.train_data = train_data
        self.valid_data = valid_data

    def train_loader_kwargs(self, **kwargs):
        train_ds = ChunkDataSet(*self.train_data)
        return {
            "dataset": train_ds,
            "sampler": RandomSampler(train_ds, num_samples=kwargs["chunks"]),
        }

    def valid_loader_kwargs(self, **kwargs):
        valid_ds = ChunkDataSet(*self.valid_data)
        return {
            "dataset": valid_ds,
            "shuffle": False,
        }


def load_chunks(input_folder):
    chunks = np.load(input_folder / "chunks.npy").astype(np.float32)
    refs = np.load(input_folder / "references.npy").astype(np.int64)
    ref_lens = np.load(input_folder / "reference_lengths.npy").astype(np.int64)
    return chunks, refs, ref_lens


chunks, refs, lens = load_chunks(Path("/data/ctc_output"))

# As an example, we take the first 1000 chunks for training and the last 100 for validation
# In practice more data will be required! 
train_data = chunks[:1000], refs[:1000], lens[:1000]
valid_data = chunks[100:], refs[100:], lens[100:]

Loader = partial(ChunkLoader, train_data, valid_data)

```
