# Bonito

A convolutional basecaller inspired by QuartzNet.

## Features

 - Raw signal input.
 - Simple 5 state output `{BLANK, A, C, G, T}`.
 - CTC training.
 - Small Python codebase.

## Quickstart

```bash
$ git clone https://github.com/nanoporetech/bonito.git
$ cd bonito
$ python3 -m venv venv3
$ source venv3/bin/activate
(venv3) $ pip install --upgrade pip
(venv3) $ pip install -r requirements.txt
(venv3) $ python setup.py develop
```

## Training a model

```bash
(venv3) $ # download the training data and train a model with the default settings
(venv3) $ ./scripts/get-training-data
(venv3) $ bonito train ./data/model-dir ./config/quartznet5x5.toml
(venv3) $
(venv3) $ # train on gpu 1, use mixed precision, larger batch size and use 1,000,000 chunks
(venv3) $ export CUDA_VISIBLE_DEVICES=1
(venv3) $ bonito train ./data/model-dir ./config/quartznet5x5.toml --amp --batch 64 --chunks 1000000
```

Automatic mixed precision can be used for speeding up training by passing the `--amp` flag to the training script, however the [apex](https://github.com/nvidia/apex#quick-start) package will need to be installed manually.

Pretrained models can be downloaded by running `./scripts/get-models`.

## Basecalling

```bash
(venv3) $ bonito basecaller <READ_DIR> <MODEL_DIR> > basecalls.fasta
```

## Scripts

 - `bonito view` - view a model architecture for a given `.toml` file and the number of parameters in the network.
 - `bonito train` - train a bonito model.
 - `bonito evaluate` - evaluate a model performance on a chunk basis.
 - `bonito basecaller` - basecaller *(`.fast5` -> `.fasta`)*.

### References

 - [Sequence Modeling With CTC](https://distill.pub/2017/ctc/)
 - [Quartznet: Deep Automatic Speech Recognition With 1D Time-Channel Separable Convolutions](https://arxiv.org/pdf/1910.10261.pdf)

### Licence and Copyright
(c) 2019 Oxford Nanopore Technologies Ltd.

Bonito is distributed under the terms of the Oxford Nanopore
Technologies, Ltd.  Public License, v. 1.0.  If a copy of the License
was not distributed with this file, You can obtain one at
http://nanoporetech.com

### Research Release

Research releases are provided as technology demonstrators to provide early access to features or stimulate Community development of tools. Support for this software will be minimal and is only provided directly by the developers. Feature requests, improvements, and discussions are welcome and can be implemented by forking and pull requests. However much as we would like to rectify every issue and piece of feedback users may have, the developers may have limited resource for support of this software. Research releases may be unstable and subject to rapid iteration by Oxford Nanopore Technologies.
