"""
Bonito Download
"""

import os
import re
import sys
from shutil import rmtree
from zipfile import ZipFile
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.util import __data__, __models__
from bonito.cli.convert import main as convert
from bonito.cli.convert import argparser as cargparser

import requests
from tqdm import tqdm


class File:
    """
    Small class for downloading models and training assets.
    """
    __url__ = "https://cdn.oxfordnanoportal.com/software/analysis/bonito/"

    def __init__(self, path, url_frag, force=False):
        self.path = path
        self.force = force
        self.filename = url_frag
        self.url = os.path.join(self.__url__, "%s.zip" % url_frag)

    def location(self, filename):
        return os.path.join(self.path, filename)

    def exists(self, filename):
        return os.path.exists(self.location(filename))

    def download(self):
        """
        Download the remote file
        """
        # create the requests for the file
        req = requests.get(self.url, stream=True)
        total = int(req.headers.get('content-length', 0))
        fname = "%s.zip" % self.filename

        # skip download if local file is found
        if self.exists(fname.strip('.zip')) and not self.force:
            print("[skipping %s]" % fname, file=sys.stderr)
            return

        if self.exists(fname.strip('.zip')) and self.force:
            rmtree(self.location(fname.strip('.zip')))

        # download the file
        with tqdm(total=total, unit='iB', ascii=True, ncols=100, unit_scale=True, leave=False) as t:
            with open(self.location(fname), 'wb') as f:
                for data in req.iter_content(1024):
                    f.write(data)
                    t.update(len(data))

        print("[downloaded %s]" % fname, file=sys.stderr)

        # unzip .zip files
        if fname.endswith('.zip'):
            with ZipFile(self.location(fname), 'r') as zfile:
                zfile.extractall(self.path)
            os.remove(self.location(fname))

        # convert chunkify training files to bonito
        if fname.endswith('.hdf5'):
            print("[converting %s]" % fname, file=sys.stderr)
            args = cargparser().parse_args([
                self.location(fname),
                self.location(fname).strip('.hdf5')
            ])
            convert(args)


models = [

    "dna_r10.4.1_e8.2_260bps_fast@v4.1.0",
    "dna_r10.4.1_e8.2_260bps_hac@v4.1.0",
    "dna_r10.4.1_e8.2_260bps_sup@v4.1.0",

    "dna_r10.4.1_e8.2_400bps_fast@v4.1.0",
    "dna_r10.4.1_e8.2_400bps_hac@v4.1.0",
    "dna_r10.4.1_e8.2_400bps_sup@v4.1.0",

    "dna_r10.4.1_e8.2_260bps_fast@v4.0.0",
    "dna_r10.4.1_e8.2_260bps_hac@v4.0.0",
    "dna_r10.4.1_e8.2_260bps_sup@v4.0.0",

    "dna_r10.4.1_e8.2_400bps_fast@v4.0.0",
    "dna_r10.4.1_e8.2_400bps_hac@v4.0.0",
    "dna_r10.4.1_e8.2_400bps_sup@v4.0.0",

    "dna_r10.4.1_e8.2_260bps_fast@v3.5.2",
    "dna_r10.4.1_e8.2_260bps_hac@v3.5.2",
    "dna_r10.4.1_e8.2_260bps_sup@v3.5.2",

    "dna_r10.4.1_e8.2_400bps_fast@v3.5.2",

    "dna_r10.4.1_e8.2_400bps_hac@v3.5.2",
    "dna_r10.4.1_e8.2_400bps_sup@v3.5.2",

    "dna_r9.4.1_e8_sup@v3.3",
    "dna_r9.4.1_e8_hac@v3.3",
    "dna_r9.4.1_e8_fast@v3.4",
]


training = [
    "dna_r9.4.1.hdf5",
]


def main(args):
    """
    Download models and training sets
    """
    if args.models or args.all:
        
        if args.show:
            print("[available models]", file=sys.stderr)
            for model in models:
                print(f" - {model}", file=sys.stderr)
        else:
            print("[downloading models]", file=sys.stderr)
            for model in models:
                File(__models__, model, args.force).download()
    if args.training or args.all:
        print("[downloading training data]", file=sys.stderr)
        for train in training:
            File(__data__, train, args.force).download()


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true')
    group.add_argument('--models', action='store_true')
    group.add_argument('--training', action='store_true')
    parser.add_argument('--list', '--show', dest='show', action='store_true')
    parser.add_argument('-f', '--force', action='store_true')
    return parser
