import argparse

from . import basecaller, evaluate, trainprog, view, convert_data

__version__ = '0.0.1'

def main():
    parser = argparse.ArgumentParser(
        'bonito',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--version', action='version',
        version='%(prog)s {}'.format(__version__))

    subparsers = parser.add_subparsers(
        title='subcommands', description='valid commands',
        help='additional help', dest='command')
    subparsers.required = True

    for module in ('basecaller', 'evaluate', 'trainprog', 'view', 'convert_data'):
        mod = globals()[module]
        p = subparsers.add_parser(module, parents=[mod.argparser()])
        p.set_defaults(func=mod.main)

    args = parser.parse_args()
    args.func(args)
