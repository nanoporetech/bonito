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

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="ont-%s" % __pkg_name__,
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    author='Oxford Nanopore Technologies, Ltd',
    author_email='support@nanoporetech.com',
    url='https://github.com/nanoporetech/bonito',
    entry_points = {
        'console_scripts': [
            '{0} = {0}:main'.format(__pkg_name__)
        ]
    },
)

