#!/usr/bin/env python
"""Change .unw outputs into .tif files using dishgtfile

    Usage: convert_snaphu.py [--file "/path/to/unwrapped.unw"] [--output unwrapped.unw.tif]
        convert_snaph.py [--path "/path/to/igrams/"] # will convert all files in path

    With no arguments, converts all .unw files in current directory

"""

import argparse
import glob
import sys
import subprocess
import os.path
from os.path import abspath, dirname

import insar.sario
from insar.log import get_log

UNWRAPPED_EXT = '.unw'
logger = get_log()


def _find_unw_files(dir_path):
    return glob.glob(os.path.join(dir_path, '*.unw'))


def unw_to_tif(filename, num_cols, num_rows, max_height):
    """Uses dishgtfile program to convert a .unw to .tif"""
    # The "1" is "firstline" option
    convert_command = "dishgtfile {filename} {num_cols} 1 {num_rows} {max_height}"
    convert_cmd = convert_command.format(
        filename=filename, num_cols=num_cols, num_rows=num_rows, max_height=max_height)
    logger.info("Running %s", convert_cmd)
    subprocess.check_call(convert_cmd.split(' '))

    # Default output for dishgtfile is named "dishgt.tif" in current dir
    # TODO: is there anyway to specify the name instead? cant find source code
    output_name = "dishgt.tif"
    newfile_name = filename + '.tif'  # full extension is .unw.tif
    move_cmd = "mv {out} {new}".format(out=output_name, new=newfile_name)
    logger.info("Running %s", move_cmd)
    subprocess.check_call(move_cmd.split(' '))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="filename.out", help="Output filename")
    parser.add_argument("--file", "-f", help="Single .unw file to convert")
    parser.add_argument("--path", "-p", help="Path to directory of .unw files")
    parser.add_argument(
        "--max-height",
        "-m",
        default=100,
        help="Maximum height/max absolute phase in .unw files "
        "(used for contour_interval option to dishgt)")
    args = parser.parse_args()

    if args.file and args.path:
        logger.error("Can only specify one of options --file and --path")
        sys.exit(1)
    elif args.path:
        dir_path = args.path
        files_to_convert = _find_unw_files(dir_path)
    elif args.file:
        file_ext = insar.sario.get_file_ext(args.file)
        if file_ext != UNWRAPPED_EXT:
            logger.error("Must be a .unw file to convert: %s", args.file)
        files_to_convert = [args.file]
        dir_path = dirname(args.file)  # Init variable for saving later
    else:
        logger.info("Searching in current directory for .unw files.")
        dir_path = './'
        files_to_convert = _find_unw_files(dir_path)

    dem_rsc_file = os.path.join(dir_path, 'dem.rsc')
    rsc_data = insar.sario.load_dem_rsc(dem_rsc_file)
    for file_ in files_to_convert:
        logger.info("Converting %s", file_)
        unw_to_tif(file_, rsc_data['WIDTH'], rsc_data['FILE_LENGTH'], args.max_height)


if __name__ == '__main__':
    main()
