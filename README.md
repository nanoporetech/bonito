# Bonito

[![PyPI version](https://badge.fury.io/py/ont-bonito.svg)](https://badge.fury.io/py/ont-bonito)
[![py39](https://img.shields.io/badge/python-3.9-brightgreen.svg)](https://img.shields.io/badge/python-3.9-brightgreen.svg)
[![py310](https://img.shields.io/badge/python-3.10-brightgreen.svg)](https://img.shields.io/badge/python-3.10-brightgreen.svg)
[![py311](https://img.shields.io/badge/python-3.11-brightgreen.svg)](https://img.shields.io/badge/python-3.11-brightgreen.svg)
[![py312](https://img.shields.io/badge/python-3.12-brightgreen.svg)](https://img.shields.io/badge/python-3.12-brightgreen.svg)
[![py313](https://img.shields.io/badge/python-3.13-brightgreen.svg)](https://img.shields.io/badge/python-3.13-brightgreen.svg)
[![cu118](https://img.shields.io/badge/cuda-11.8-blue.svg)](https://img.shields.io/badge/cuda-11.8-blue.svg)
[![cu124](https://img.shields.io/badge/cuda-12.4-blue.svg)](https://img.shields.io/badge/cuda-12.4-blue.svg)

Bonito is an open source research basecaller for Oxford Nanopore reads.

For anything other than basecaller training or method development please use [dorado](https://github.com/nanoporetech/dorado).

```bash
$ pip install --upgrade pip
$ pip install ont-bonito
$ bonito basecaller dna_r10.4.1_e8.2_400bps_hac@v5.0.0 /data/reads > basecalls.bam
```

Bonito supports writing aligned/unaligned `{fastq, sam, bam, cram}`.

```bash
$ bonito basecaller dna_r10.4.1_e8.2_400bps_hac@v5.0.0 --reference reference.mmi /data/reads > basecalls.bam
```

Bonito will download and cache the basecalling model automatically on first use but all models can be downloaded with -

``` bash
$ bonito download --models --show  # show all available models
$ bonito download --models         # download all available models
```

## Transformer Models

The `bonito.transformer` package requires
[flash-attn](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features).

This must be manually installed as the `flash-attn` packaging system prevents it from being listed as a normal dependency.

Setting `CUDA_HOME` to the relevant library directory will help avoid CUDA version mismatches between packages.

## Modified Bases

For modified-base calling with ont-supported mods please use [dorado](https://github.com/nanoporetech/dorado)
For development of modified base calling models please see [remora](https://github.com/nanoporetech/remora).

## Training your own model

For detailed information on the training process, please see the [Training Documentation](documentation/training.md).

## Developer Quickstart

```bash
$ git clone https://github.com/nanoporetech/bonito.git  # or fork first and clone that
$ cd bonito
$ python3 -m venv venv3
$ source venv3/bin/activate
(venv3) $ pip install --upgrade pip
(venv3) $ pip install -e .[cu118] --extra-index-url https://download.pytorch.org/whl/cu118
```

The `ont-bonito[cu118]` and `ont-bonito[cu121]` optional dependencies can be used, along
with the corresponding `--extra-index-url`, to ensure the PyTorch package matches the
local CUDA setup.

## Interface

 - `bonito view` - view a model architecture for a given `.toml` file and the number of parameters in the network.
 - `bonito train` - train a bonito model.
 - `bonito evaluate` - evaluate a model performance.
 - `bonito download` - download pretrained models and training datasets.
 - `bonito basecaller` - basecaller *(`.fast5` -> `.bam`)*.

### References

 - [Sequence Modeling With CTC](https://distill.pub/2017/ctc/)
 - [Quartznet: Deep Automatic Speech Recognition With 1D Time-Channel Separable Convolutions](https://arxiv.org/pdf/1910.10261.pdf)
 - [Pair consensus decoding improves accuracy of neural network basecallers for nanopore sequencing](https://www.biorxiv.org/content/10.1101/2020.02.25.956771v1.full.pdf)

### Licence and Copyright
(c) 2019 Oxford Nanopore Technologies Ltd.

Bonito is distributed under the terms of the Oxford Nanopore
Technologies, Ltd.  Public License, v. 1.0.  If a copy of the License
was not distributed with this file, You can obtain one at
http://nanoporetech.com

### Research Release

Research releases are provided as technology demonstrators to provide early access to features or stimulate Community development of tools. Support for this software will be minimal and is only provided directly by the developers. Feature requests, improvements, and discussions are welcome and can be implemented by forking and pull requests. However much as we would like to rectify every issue and piece of feedback users may have, the developers may have limited resource for support of this software. Research releases may be unstable and subject to rapid iteration by Oxford Nanopore Technologies.

### Citation

```
@software{bonito,
  title = {Bonito: A PyTorch Basecaller for Oxford Nanopore Reads},
  author = {{Chris Seymour, Oxford Nanopore Technologies Ltd.}},
  year = {2019},
  url = {https://github.com/nanoporetech/bonito},
  note = {Oxford Nanopore Technologies, Ltd. Public License, v. 1.0},
  abstract = {Bonito is an open source research basecaller for Oxford Nanopore reads. It provides a flexible platform for training and developing basecalling models using PyTorch.}
}
```