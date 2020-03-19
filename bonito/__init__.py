from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from bonito import basecaller, evaluate, view
modules = ['basecaller', 'evaluate', 'view']

try:
    from bonito import train, tune
    modules.extend(['train', 'tune'])
except ImportError:
    pass


__version__ = '0.1.0'


def main():
    parser = ArgumentParser(
        'bonito', 
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-v', '--version', action='version',
        version='%(prog)s {}'.format(__version__)
    )

    subparsers = parser.add_subparsers(
        title='subcommands', description='valid commands',
        help='additional help', dest='command'
    )
    subparsers.required = True

    for module in modules:
        mod = globals()[module]
        p = subparsers.add_parser(module, parents=[mod.argparser()])
        p.set_defaults(func=mod.main)

    args = parser.parse_args()
    args.func(args)
