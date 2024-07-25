import os
import re
from setuptools import setup, find_packages


__pkg_name__ = 'bonito'
require_file = 'requirements.txt'
package_name = "ont-%s" % __pkg_name__

verstrline = open(os.path.join(__pkg_name__, '__init__.py'), 'r').read()
vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "{}/__init__.py".'.format(__pkg_name__))


with open(require_file) as f:
    requirements = [r.split()[0] for r in f.read().splitlines()]

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
    python_requires='>=3.8',
    extras_require={
        # --extra-index-url https://download.pytorch.org/whl/cu118
        "cu118": ["torch==2.1.2+cu118"],
        # --extra-index-url https://download.pytorch.org/whl/cu121
        "cu121": ["torch==2.2.0"],
    },
    entry_points={
        'console_scripts': [
            '{0} = {0}:main'.format(__pkg_name__)
        ]
    },
)
