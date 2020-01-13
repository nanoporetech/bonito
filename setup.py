import os
import re
from setuptools import setup, find_packages

__pkg_name__ = 'bonito'

verstrline = open(os.path.join(__pkg_name__, '__init__.py'), 'r').read()
vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "{}/__init__.py".'.format(__pkg_name__))


setup(
    name="ont-%s" % __pkg_name__,
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    author='Oxford Nanopore Technologies, Ltd',
    author_email='support@nanoporetech.com',
    url='https://github.com/nanoporetech/bonito',
    entry_points = {
        'console_scripts': [
            '{0} = {0}:main'.format(__pkg_name__)
        ]
    },
    install_requires=[
        'toml==0.10.0',
        'tqdm==4.31.1',
        'torch==1.3.1',
        'parasail==1.1.19',
        'ont-fast5-api==2.0.0',
    ]
)

