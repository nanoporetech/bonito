# Bonito

A convolutional basecaller inspired by QuartzNet.

## Features

 - Raw signal input.
 - Simple 5 state output `{BLANK, A, C, G, T}`.
 - CTC training.
 - Small Python codebase.

## Quickstart

```
$ python3 -m venv venv3
$ source venv3/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ python setup.py develop
```

## Training a model

```
(venv3) $ ./bin/get-training-data
(venv3) $ ./bin/train /data/model-dir ./config/quartznet5x5.toml
(venv3) $ # train on gpu 1, use mixed precision, larger batch size and only 20,000 chunks
(venv3) $ CUDA_VISIBLE_DEVICES=1 ./bin/train /data/model-dir config/quartznet5x5.toml --amp --batch 64 --chunks 20000
```

The default configuration is for the QuartzNet 5x5 architecture.

Automatic mixed precision can be used for speeding up training by passing the `--amp` flag to the training script, however the [apex](https://github.com/nvidia/apex#quick-start) package will need to be installed manually.

## Scripts

 - `./bin/view` view a model architecture for a given `.toml` file and the number of parameters in the network.
 - `./bin/train` train a bonito model.
 - `./bin/call` evaluate a model performance on chunk basis.
 - `./bin/basecall` full basecaller `.fast5` in, `.fasta` out.

### References

 - [Sequence Modeling With CTC](https://distill.pub/2017/ctc/)
 - [Quartznet: Deep Automatic Speech Recognition With 1D Time-Channel Separable Convolutions](https://arxiv.org/pdf/1910.10261.pdf)

### Licence and Copyright
(c) 2019 Oxford Nanopore Technologies Ltd.

Bonito is distributed under the terms of the Oxford Nanopore
Technologies, Ltd.  Public License, v. 1.0.  If a copy of the License
was not distributed with this file, You can obtain one at
http://nanoporetech.com
