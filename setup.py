import setuptools
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call


# Classes for running "make" to compile the bin/upsample
class PostDevelopCommand(develop):
    """Post-installation for development mode, installs from Makefile."""

    def run(self):
        print("=========================================================")
        check_call("make")
        print("=========================================================")
        develop.run(self)


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        print("=========================================================")
        check_call("make")
        print("=========================================================")
        install.run(self)


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="insar",
    version="1.0",
    author="Scott Staniewicz",
    author_email="scott.stanie@utexas.com",
    description="Tools for gathering and preprocessing InSAR data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scottstanie/insar",
    packages=setuptools.find_packages(),
    include_package_data=True,
    # Extra command to compile the upsample.c script
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
    },
    classifiers=(
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: C",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ),
    install_requires=["numpy", "requests", "matplotlib", "click"],
    entry_points={
        "console_scripts": [
            "insar=insar.scripts.cli:cli",
        ],
    },
    data_files=[("bin", ["bin/upsample"])],
    zip_safe=False)
