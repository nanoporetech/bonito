import os
import re
import setuptools
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from torch.utils.cpp_extension import BuildExtension

ROOT_DIR = os.path.dirname(__file__)
# temporary directory to store third-party packages
THIRDPARTY_SUBDIR = "bonito/thirdparty_files"


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


flash_attn_version = "2.5.6"
install_dir = os.path.join(ROOT_DIR, THIRDPARTY_SUBDIR)
os.mkdir(install_dir, exist_ok=True)
subprocess.check_output(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        f"--target={install_dir}",
        "einops",  # Dependency of flash-attn.
        f"flash-attn=={flash_attn_version}",
        "--no-dependencies",  # Required to avoid re-installing torch.
    ],
    env=dict(os.environ, CC="gcc"),
    stderr=subprocess.STDOUT,
    shell=True
)

# Copy the FlashAttention package into the vLLM package after build.
class build_ext(BuildExtension):

    def run(self):
        super().run()
        target_dir = os.path.join(self.build_lib, THIRDPARTY_SUBDIR)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        self.copy_tree(install_dir, target_dir)

class BinaryDistribution(setuptools.Distribution):

    def has_ext_modules(self):
        return True


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
    entry_points={
        'console_scripts': [
            '{0} = {0}:main'.format(__pkg_name__)
        ]
    },
    cmdclass={"build_ext": build_ext},
    distclass=BinaryDistribution,
)
