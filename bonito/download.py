"""
Bonito Download
"""

import os
import re
from abc import ABC
from zipfile import ZipFile
from argparse import Namespace
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.util import __data__, __models__, __url__
from bonito.convert import main as convert
from bonito.convert import argparser as cargparser

import requests
from tqdm import tqdm


class BoxFile:
    """
    Small class for downloading models and training assets.
    """
    def __init__(self, path, url_frag):
        self.path = path
        self.url = os.path.join(__url__, url_frag)

    def location(self, filename):
        return os.path.join(self.path, filename)

    def exists(self, filename):
        return os.path.exists(self.location(filename))

    def download(self):
        req = requests.get(self.url, stream=True)
        total = int(req.headers.get('content-length', 0))
        fname = re.findall('filename="([^"]+)', req.headers['content-disposition'])[0]

        if self.exists(fname.strip('.zip')):
            print("[skipping %s]" % fname)

            # convert chunkify training files to bonito
            if fname.endswith('.hdf5'):
                print("[converting %s]" % fname)
                args = cargparser().parse_args([
                    self.location(fname),
                    self.location(fname).strip('.hdf5')
                ])
                convert(args)

            return

        with tqdm(total=total, unit='iB', ascii=True, ncols=100, unit_scale=True, leave=False) as t:
            with open(self.location(fname), 'wb') as f:
                for data in req.iter_content(1024):
                    f.write(data)
                    t.update(len(data))

        print("[downloaded %s]" % fname)

        # unzip model directories
        if fname.endswith('.zip'):
            with ZipFile(self.location(fname), 'r') as zfile:
                zfile.extractall(self.path)
            os.remove(self.location(fname))

        # convert chunkify training files to bonito
        if fname.endswith('.hdf5'):
            print("[converting %s]" % fname)
            args = cargparser().parse_args([
                self.location(fname),
                self.location(fname).strip('.hdf5')
            ])
            convert(args)


models = [
    BoxFile(__models__, "iatumku2snun5xfwn70wyym9cdo548mv.zip"),
]

training = [
    BoxFile(__data__, "cmh91cxupa0are1kc3z9aok425m75vrb.hdf5"),
]


def main(args):
    """
    Download models and training sets
    """
    if args.models or args.all:
        print("[downloading models]")
        for model in models:
            model.download()

    if args.training or args.all:
        print("[downloading training data]")
        for train in training:
            train.download()


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true')
    group.add_argument('--models', action='store_true')
    group.add_argument('--training', action='store_true')
    return parser
