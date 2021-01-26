# Bonito

[![PyPI version](https://badge.fury.io/py/ont-bonito.svg)](https://badge.fury.io/py/ont-bonito)

A PyTorch Basecaller for Oxford Nanopore Reads.

```bash
$ pip install ont-bonito
$ bonito basecaller dna_r9.4.1 /data/reads > basecalls.fasta
```

If a reference is provided in either `.fasta` or `.mmi` format then bonito will output in `sam` format.

```bash
$ bonito basecaller dna_r9.4.1 --reference reference.mmi /data/reads > basecalls.sam
```

Current available models are `dna_r9.4.1`, `dna_r10.3`.

## Developer Quickstart

```bash
$ git clone https://github.com/nanoporetech/bonito.git  # or fork first and clone that
$ cd bonito
$ python3 -m venv venv3
$ source venv3/bin/activate
(venv3) $ pip install pip==20.3.4
(venv3) $ pip install -r requirements.txt
(venv3) $ python setup.py develop
(venv3) $ bonito download --models --latest
```

## Training your own model

To train a model using your own reads, first basecall the reads with the additional `--save-ctc` flag and use the output directory as the input directory for training.

```bash
$ bonito basecaller dna_r9.4.1 --save-ctc --reference reference.mmi /data/reads > /data/training/ctc-data/basecalls.sam
$ bonito train --amp --directory /data/training/ctc-data /data/training/model-dir
```

If you are interested in method development and don't have you own set of reads then a pre-prepared set is provide.

```bash
$ bonito download --training
$ bonito train --amp /data/training/model-dir
```

Automatic mixed precision can be used to speed up training with the `--amp` flag *(however [apex](https://github.com/nvidia/apex#quick-start) needs to be installed manually)*.

For multi-gpu training use the `$CUDA_VISIBLE_DEVICES` environment variable to select which GPUs and add the `--multi-gpu` flag.

```bash
$ export CUDA_VISIBLE_DEVICES=0,1,2,3
$ bonito train --amp --multi-gpu --batch 256 /data/model-dir
```

To evaluate the pretrained model run `bonito evaluate dna_r9.4.1`.

For a model you have trainined yourself, replace `dna_r9.4.1` with the model directory.

## Pair Decoding

Pair decoding takes a template and complement read to produce higher quaility calls.

```bash
$ bonito pair pairs.csv /data/reads > basecalls.fasta
```

The `pairs.csv` file is expected to contain pairs of read ids per line *(seperated by a single space)*.

## Interface

 - `bonito view` - view a model architecture for a given `.toml` file and the number of parameters in the network.
 - `bonito tune` - distributed tuning of network hyperparameters.
 - `bonito train` - train a bonito model.
 - `bonito convert` - convert a hdf5 training file into a bonito format.
 - `bonito evaluate` - evaluate a model performance.
 - `bonito download` - download pretrained models and training datasets.
 - `bonito basecaller` - basecaller *(`.fast5` -> `.fasta`)*.

# Medaka

The Medaka can be downloaded from [here](https://nanoporetech.box.com/shared/static/ve8445ceb2bnwod1zaj0z2ptuwsvxd64.hdf5).

| Coverage | B. subtilis | E. coli | E. faecalis | L. monocytogenes | P. aeruginosa | S. aureus | S. enterica |
| -------- |:-----------:|:-------:|:-----------:|:----------------:|:-------------:|:---------:|:-----------:|
|       25 |       38.86 |   42.60 |       40.24 |            41.55 |         41.55 |     43.98 |       36.78 |
|       50 |       39.36 |   45.23 |       43.01 |            43.98 |         45.34 |     46.99 |       38.07 |
|       75 |       43.98 |   45.23 |       45.23 |            45.23 |         50.00 |     46.99 |       38.36 |
|      100 |       43.98 |   46.99 |       45.23 |            46.99 |         50.00 |     50.00 |       39.39 |
|      125 |       45.23 |   45.23 |       45.23 |            45.23 |         50.00 |     50.00 |       39.39 |
|      150 |       45.23 |   46.99 |       46.99 |            46.99 |         50.00 |     50.00 |       39.59 |
|      175 |       45.23 |   46.99 |       46.99 |            46.99 |         50.00 |     50.00 |       39.59 |
|      200 |       46.99 |   46.99 |       50.00 |            50.00 |         50.00 |     50.00 |       40.00 |

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
