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

## Developer Quickstart

```bash
$ git clone https://github.com/nanoporetech/bonito.git  # or fork first and clone that
$ cd bonito
$ python3 -m venv venv3
$ source venv3/bin/activate
(venv3) $ pip install --upgrade pip
(venv3) $ pip install -r requirements.txt
(venv3) $ python setup.py develop
(venv3) $ bonito download --models --latest
```

## Models

The following pretrained models are available to download with `bonito download`.

| Model | Type | Bonito Version  | 
| ------ | ------ |------ |
| `dna_r9.4.1@v3.3`, `dna_r10.3@v3.3`  | CRF-CTC RNN _(fixed blank score)_ | v0.3.7 |
| `dna_r9.4.1@v3.2`, `dna_r10.3@v3.2`  | CRF-CTC RNN | v0.3.6 |
| `dna_r10.3@v3` | CRF-CTC RNN  | v0.3.2 |
| `dna_r9.4.1@v3.1`  | CRF-CTC RNN  | v0.3.1 |
| `dna_r9.4.1@v3`  | CRF-CTC RNN  | v0.3.0 |
| `dna_r9.4.1@v2` | CTC CNN _(Custom QuartzNet)_ | v0.2.0 | 
| `dna_r9.4.1@v1` | CTC CNN _(5x5 QuartzNet)_ | v0.1.2 |

All models can be downloaded with `bonito download --models` or if you just want the latest version then `bonito download --models --latest -f`.

## Training your own model

To train a model using your own reads, first basecall the reads with the additional `--save-ctc` flag and use the output directory as the input directory for training.

```bash
$ bonito basecaller dna_r9.4.1 --save-ctc --reference reference.mmi /data/reads > /data/training/ctc-data/basecalls.sam
$ bonito train --directory /data/training/ctc-data /data/training/model-dir
```

In addition to training a new model from scratch you can also easily fine tune one of the pretrained models.  

```bash
bonito train --epochs 1 --lr 5e-4 --pretrained dna_r9.4.1@v3.3 --directory /data/training/ctc-data /data/training/fine-tuned-model
```

If you are interested in method development and don't have you own set of reads then a pre-prepared set is provide.

```bash
$ bonito download --training
$ bonito train /data/training/model-dir
```

All training calls use Automatic Mixed Precision to speed up training. To disable this, set the `--no-amp` flag to True. 

## Duplex

Duplex calling takes template and complement reads and produces a single higher quality call.

```bash
$ bonito duplex dna_r9.4.1 /data/reads --pairs pairs.txt --reference ref.mmi > basecalls.sam
```

The `pairs.csv` file is expected to contain pairs of read ids per line *(seperated by a single space)*.

Follow on reads can also be automatically paired if an alignment summary file is provided instead of a `pairs.csv`.

```bash
$ bonito duplex dna_r9.4.1 /data/reads --summary sequencing_summary.txt --reference ref.mmi > basecalls.sam
```

## Interface

 - `bonito view` - view a model architecture for a given `.toml` file and the number of parameters in the network.
 - `bonito train` - train a bonito model.
 - `bonito convert` - convert a hdf5 training file into a bonito format.
 - `bonito evaluate` - evaluate a model performance.
 - `bonito download` - download pretrained models and training datasets.
 - `bonito basecaller` - basecaller *(`.fast5` -> `.fasta`)*.

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
