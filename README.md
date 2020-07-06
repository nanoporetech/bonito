# Bonito

[![PyPI version](https://badge.fury.io/py/ont-bonito.svg)](https://badge.fury.io/py/ont-bonito)

A convolutional basecaller inspired by QuartzNet.

## Features

 - Raw signal input.
 - Simple 5 state output `{BLANK, A, C, G, T}`.
 - CTC training.
 - Small Python codebase.

## Basecalling

```bash
$ pip install ont-bonito
$ bonito basecaller dna_r9.4.1 /data/reads > basecalls.fasta
```
 
 If you have a `turing` or `volta` GPU the `--half` flag can be uses to increase performance.
 
## Pair Decoding

Pair decoding takes a template and complement read to produce higher quaility calls.

```
$ bonito basecaller pairs.csv /data/reads > basecalls.fasta
```

The `pairs.csv` file is expected to contain pairs of read ids per line *(seperated by a single space)*.


## Training your own model

To train your own model first download the training data.

```bash
$ bonito download --training
$ bonito train --amp /data/model-dir
```

Automatic mixed precision can be used to speed up training with the `--amp` flag *(however [apex](https://github.com/nvidia/apex#quick-start) needs to be installed manually)*.

For multi-gpu training use the `$CUDA_VISIBLE_DEVICES` environment variable to select which GPUs and add the `--multi-gpu` flag.

```bash
$ export CUDA_VISIBLE_DEVICES=0,1,2,3
$ bonito train --amp --multi-gpu --batch 256 /data/model-dir
```

To evaluate the pretrained model run `bonito evaluate dna_r9.4.1 --half`.

For a model you have trainined yourself, replace `dna_r9.4.1` with the model directory.

## Interface

 - `bonito view` - view a model architecture for a given `.toml` file and the number of parameters in the network.
 - `bonito tune` - distributed tuning of network hyperparameters.
 - `bonito train` - train a bonito model.
 - `bonito convert` - convert a hdf5 training file into a bonito format.
 - `bonito evaluate` - evaluate a model performance.
 - `bonito download` - download pretrained models and training datasets.
 - `bonito basecaller` - basecaller *(`.fast5` -> `.fasta`)*.

## Developer Quickstart

```bash
$ git clone https://github.com/nanoporetech/bonito.git  # or fork first and clone that
$ cd bonito
$ python3 -m venv venv3
$ source venv3/bin/activate
(venv3) $ pip install --upgrade pip
(venv3) $ pip install -r requirements.txt
(venv3) $ python setup.py develop
(venv3) $ bonito download --all
```

## Medaka

The Medaka can be downloaded from [here](https://nanoporetech.box.com/shared/static/u5gncwjbtg2k3dkw26nmvdvck65ab3xh.hdf5).

It has been trained on Zymo: *E. faecalis, P. aeruginosa, S. enterica1, S.aureus and E.coli (with L. monocytogenes and B. subtilis held out)*.

| Coverage | B. subtilis | E. coli | E. faecalis | L. monocytogenes | S. aureus | S. enterica |
| -------- |:-----------:|:-------:|:-----------:|:----------------:|:---------:|:-----------:|
|       25 |       36.92 |   39.51 |       36.68 |            37.33 |     36.87 |       37.70 |
|       50 |       41.55 |   43.98 |       40.97 |            42.22 |     42.22 |       42.22 |
|       75 |       43.01 |   45.23 |       42.22 |            43.01 |     43.01 |       43.98 |
|      100 |       43.01 |   45.23 |       43.98 |            43.47 |     44.56 |       45.23 |
|      125 |       45.23 |   46.99 |       43.98 |            45.23 |     45.23 |       45.23 |
|      150 |       45.23 |   46.99 |       45.23 |            45.23 |     45.23 |       46.99 |
|      175 |       46.12 |   46.99 |       45.23 |            46.99 |     46.99 |       46.99 |
|      200 |       46.99 |   46.99 |       45.23 |            45.23 |     46.99 |       46.99 |

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
