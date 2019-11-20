# Bonito

A convolutional basecaller based of QuartzNet using a simple 5 state encoding `{BLANK, A, C, G, T}` and trained with CTC.

## Quickstart

```
$ python3 -m venv venv3
$ source venv3/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ python setup.py develop
```

## Training a model

Download the training from [here](https://nanoporetech.ent.box.com/s/zvdpnbztlc727igiv61hees4v45391ho).

```
(venv3) $ ./bin/train /data/training/example ./config/quartznet5x5.toml
```

The default configuration is for the QuartzNet 5x5 architecture.

Automatic mixed precision can be use for speeding up training by passing the `--amp` flag to the training script, however the [apex](https://github.com/nvidia/apex#quick-start) package will need to be installed manually.

## Scripts

 - `./bin/view` view a model architecture for a given `.toml` file and the number of parameters in the network.
 - `./bin/train` train a bonito model.
 - `./bin/call` evaluate a model performance on chunk basis.
 - `./bin/basecall` full basecaller `.fast5` in, `.fasta` out.

### References

 - [Sequence Modeling With CTC](https://distill.pub/2017/ctc/)
 - [Quartznet: Deep Automatic Speech Recognition With 1D Time-Channel Separable Convolutions](https://arxiv.org/pdf/1910.10261.pdf)
