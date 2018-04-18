#! /usr/bin/env python
"""Author: Scott Staniewicz
Helper functions to prepare and process UAVSAR data
Email: scott.stanie@utexas.edu
Version: 1.0
"""


import os.path
import sys
import glob
import argparse
import numpy as np


def image_load(filename, ann_info):
    dat = np.fromfile(filename, '<f4')
    im2 = realdat.reshape([nrows, ncols]) + 1j*imdat.reshape([nrows, ncols])


def make_ann_filename(mlc_filename):
    """Take the name of a mlc file and return corresponding .ann name"""
    shortname = mlc_filename.replace('HHHH', '').replace('HVHV', '').replace('VVVV', '')
    return shortname.replace('.mlc', '.ann')


def save_array(amplitude_array, outfilename):
    """Save the numpy array as a .png file


    amplitude_array (np.array, dtype=float32)
    outfilename (str)
    """
    if outfilename.endswith('.png'):
        print("Only .png is supported for now. Write in the other formats to change.")
        return
    plt.imsave(outfilename, amplitude_array, cmap='gray', vmin=0, vmax=0, format='.png')


def parse_ann_file(ann_filename, data_entries=None):
    """Returns the requested info from the annotation in ann_filename

    data_entries (list): names of pieces of data to return from the ann file
        possible strings valid in data_entries:
            "mlc size" (returns mlc_pwr.set_rows and .set_cols)

    Returns:
        ann_data (dict): key-values of requested data from .ann file
    """
    def _parse_line(line):
        l = line.split()
        # Pick the entry after the equal sign when splitting the line
        return l[l.index('=') + 1]

    def _parse_int(line):
        return int(_parse_line(line))

    def _parse_float(line):
        return float(_parse_line(line))

    ann_data = {}

    with open(ann_filename, 'r') as f:
        for line in f.readlines():
            if line.startswith('mlc_pwr.set_rows'):
                ann_data['rows'] = _parse_int(line)
            elif line.startswith('mlc_pwr.set_cols'):
                ann_data['cols'] = _parse_int(line)
            # Example: get the name of the mlc for HHHH polarization
            elif line.startswith('mlcHHHH'):
                ann_data['mlcHHHH'] = _parse_line(line)
            # TODO: Add more!

    return ann_data



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Specify the input UAVSAR filename")
    args = parser.parse_args()

    if args.filename.endswith('.mlc'):
        ann_filename = make_ann_filename(args.filename)
        ann_data = parse_ann_file(ann_filename)
        print(ann_data)
    else:
        print("Only taking in .mlc files for now")



if __name__ == "__main__":
    main()
