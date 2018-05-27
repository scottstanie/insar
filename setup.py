import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

CLASSIFIERS = """\
        Development Status :: 2 - Pre-Alpha
        Intended Audience :: Science/Research
        License :: OSI Approved :: MIT License
        Programming Language :: C
        Programming Language :: Python
        Programming Language :: Python :: 2
        Programming Language :: Python :: 3
        Topic :: Scientific/Engineering
        Operating System :: POSIX
        """

setuptools.setup(
    name="insar",
    version="0.0.1",
    author="Scott",
    author_email="scott.stanie@utexas.com",
    description="Tools for gathering and preprocessing InSAR data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scottstanie/insar",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires=['numpy', 'scipy', 'requests', 'matplotlib', 'beautifulsoup4'],
    entry_points={
        'console_scripts':
        ['create-dem=scripts.create_dem:main', 'download-eofs=scripts.download_eofs:main'],
    })
