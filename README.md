# Bonito

A convolutional basecaller inspired by Jasper/Quartznet using a simple 5 state `{BLANK, A, C, G, T}` representation and trained with CTC. 

## Quickstart

```
$ python3 -m venv .venv 
$ source ./.venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ python setup.py develop
```

## Training a model

Download the training from [here](https://nanoporetech.ent.box.com/s/zvdpnbztlc727igiv61hees4v45391ho).

```
(.venv) $ ./bin/train /data/training/example ./config/quartznet5x5.toml
```

The default configuration is for the QuartzNet 5x5 architecture.

## Scripts

 - `./bin/view` view a model architecture for a given `.toml` file and the number of parameters in the network.
 - `./bin/train` train a bonito model.
 - `./bin/call` evaluate a model performance on chunk basis.
 - `./bin/basecall` full basecaller `.fast5` in, `.fasta` out.

### References

 - [Sequence Modeling With CTC](https://distill.pub/2017/ctc/)
 - [Jasper: An End-to-End Convolutional Neural Acoustic Model](https://arxiv.org/pdf/1904.03288.pdf)
 - [Quartznet: Deep Automatic Speech Recognition With 1D Time-Channel Separable Convolutions](https://arxiv.org/pdf/1910.10261.pdf)
