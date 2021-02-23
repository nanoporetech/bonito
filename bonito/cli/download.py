"""
Bonito Download
"""

import os
import re
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
    __url__ = "https://nanoporetech.box.com/shared/static/"

    def __init__(self, path, url_frag, force):
        self.path = path
        self.force = force
        self.url = os.path.join(self.__url__, url_frag)

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
        fname = re.findall('filename="([^"]+)', req.headers['content-disposition'])[0]

        # skip download if local file is found
        if self.exists(fname.strip('.zip')) and not self.force:
            print("[skipping %s]" % fname)
            return

        if self.exists(fname.strip('.zip')) and self.force:
            rmtree(self.location(fname.strip('.zip')))

        # download the file
        with tqdm(total=total, unit='iB', ascii=True, ncols=100, unit_scale=True, leave=False) as t:
            with open(self.location(fname), 'wb') as f:
                for data in req.iter_content(1024):
                    f.write(data)
                    t.update(len(data))

        print("[downloaded %s]" % fname)

        # unzip .zip files
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


r9_models = [
    "n8c07gc9ro09zt0ivgcoeuz6krnwsnf6.zip", # dna_r9.4.1@v1
    "nas0uhf46fd1lh2jndhx2a54a9vvhxp4.zip", # dna_r9.4.1@v2
    "1wodp3ur4jhvqvu5leowfg6lrw54jxp2.zip", # dna_r9.4.1@v3
    "uetgwsnb8yfqvuyoka8p09mxilgskqc7.zip", # dna_r9.4.1@v3.1
    "47t2y48zw4waly25lmzx6sagf4bbbqqz.zip", # dna_r9.4.1@v3.2
    "arqi4qwcj9btsd6bbjsnlbai0s6dg8yd.zip",
]

r10_models = [
    "e70s615lh3i24rkhz006i0e4u4m8y2xa.zip", # dna_r10.3_q20ea
    "hnr5mwlm8vmdsfpvn5fsxn3mvhbucy5f.zip", # dna_r10.3@v3
    "yesf11tisfrncmod5hj2xtx9kbdveuqt.zip", # dna_r10.3@v3.2
    "4cunv5z7nwjag7v2bun0g7vk2lf8rqnc.zip",
]

training = [
    "cmh91cxupa0are1kc3z9aok425m75vrb.hdf5",
]


def main(args):
    """
    Download models and training sets
    """
    if args.models or args.all:
        print("[downloading models]")
        for model in r9_models[-1 if args.latest else 0:]:
            File(__models__, model, args.force).download()
        for model in r10_models[-1 if args.latest else 0:]:
            File(__models__, model, args.force).download()

    if args.training or args.all:
        print("[downloading training data]")
        for train in training:
            File(__models__, train, args.force).download()


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true')
    group.add_argument('--models', action='store_true')
    group.add_argument('--training', action='store_true')
    parser.add_argument('-f', '--force', action='store_true')
    parser.add_argument('--latest', action='store_true')
    return parser
