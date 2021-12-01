import os
import re
from setuptools import setup, find_packages
from setuptools.command.install import install


__pkg_name__ = 'bonito'

verstrline = open(os.path.join(__pkg_name__, '__init__.py'), 'r').read()
vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "{}/__init__.py".'.format(__pkg_name__))


CUDA_VERSION = os.environ.get('CUDA_VERSION')
assert(CUDA_VERSION in {'111', '113', None})


if CUDA_VERSION:
    print("Building with CUDA %s" % CUDA_VERSION)
    require_file = 'requirements-cuda%s.txt' % CUDA_VERSION
    package_name = "ont-%s-cuda%s" % (__pkg_name__, CUDA_VERSION)
else:
    print("Building with CUDA 10.2")
    require_file = 'requirements.txt'
    package_name = "ont-%s" % __pkg_name__

with open(require_file) as f:
    requirements = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name=package_name,
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Oxford Nanopore Technologies, Ltd',
    author_email='support@nanoporetech.com',
    url='https://github.com/nanoporetech/bonito',
    entry_points = {
        'console_scripts': [
            '{0} = {0}:main'.format(__pkg_name__)
        ]
    },
    dependency_links=[
        'https://download.pytorch.org/whl/torch_stable.html',
    ]
)
