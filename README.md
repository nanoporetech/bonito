# Bonito

[![PyPI version](https://badge.fury.io/py/ont-bonito.svg)](https://badge.fury.io/py/ont-bonito) 
[![py37](https://img.shields.io/badge/python-3.7-brightgreen.svg)](https://img.shields.io/badge/python-3.7-brightgreen.svg)
[![py38](https://img.shields.io/badge/python-3.8-brightgreen.svg)](https://img.shields.io/badge/python-3.8-brightgreen.svg)
[![py39](https://img.shields.io/badge/python-3.9-brightgreen.svg)](https://img.shields.io/badge/python-3.9-brightgreen.svg)
[![py310](https://img.shields.io/badge/python-3.10-brightgreen.svg)](https://img.shields.io/badge/python-3.10-brightgreen.svg)
[![cu102](https://img.shields.io/badge/cuda-10.2-blue.svg)](https://img.shields.io/badge/cuda-10.2-blue.svg)
[![cu113](https://img.shields.io/badge/cuda-11.3-blue.svg)](https://img.shields.io/badge/cuda-11.3-blue.svg)

A PyTorch Basecaller for Oxford Nanopore Reads.

```bash
$ pip install --upgrade pip
$ pip install ont-bonito
$ bonito basecaller dna_r10.4_e8.1_sup@v3.4 /data/reads > basecalls.bam
```

By default `pip` will install `torch` which is build against CUDA 10.2. For CUDA 11.3 builds run:

```
$ pip install --extra-index-url https://download.pytorch.org/whl/cu113 ont-bonito
```

Bonito supports writing aligned/unaligned `{fastq, sam, bam, cram}`.

```bash
$ bonito basecaller dna_r10.4_e8.1_sup@v3.4 --reference reference.mmi /data/reads > basecalls.bam
```

Bonito will download and cache the basecalling model automatically on first use but all models can be downloaded with -

``` bash
$ bonito download --models --show  # show all available models
$ bonito download --models         # download all available models
```

## Modified Bases

Modified base calling is handled by [Remora](https://github.com/nanoporetech/remora).

```bash
$ bonito basecaller dna_r10.4_e8.1_sup@v3.4 /data/reads --modified-bases 5mC --reference ref.mmi > basecalls_with_mods.bam
```

To use the GPU-powered modified bases inference the `onnxruntime-gpu` package is required. 

See available modified base models with the ``remora model list_pretrained`` command.

## Training your own model

To train a model using your own reads, first basecall the reads with the additional `--save-ctc` flag and use the output directory as the input directory for training.

```bash
$ bonito basecaller dna_r9.4.1 --save-ctc --reference reference.mmi /data/reads > /data/training/ctc-data/basecalls.sam
$ bonito train --directory /data/training/ctc-data /data/training/model-dir
```

In addition to training a new model from scratch you can also easily fine tune one of the pretrained models.  

```bash
bonito train --epochs 1 --lr 5e-4 --pretrained dna_r10.4_e8.1_sup@v3.4 --directory /data/training/ctc-data /data/training/fine-tuned-model
```

If you are interested in method development and don't have you own set of reads then a pre-prepared set is provide.

```bash
$ bonito download --training
$ bonito train /data/training/model-dir
```

All training calls use Automatic Mixed Precision to speed up training. To disable this, set the `--no-amp` flag to True. 

## Developer Quickstart

```bash
$ git clone https://github.com/nanoporetech/bonito.git  # or fork first and clone that
$ cd bonito
$ python3 -m venv venv3
$ source venv3/bin/activate
(venv3) $ pip install --upgrade pip
(venv3) $ pip install -r requirements.txt
(venv3) $ python setup.py develop
```

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
