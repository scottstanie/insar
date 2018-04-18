#! /usr/bin/env python
"""Author: Scott Staniewicz
Helper functions to prepare and process UAVSAR data
Email: scott.stanie@utexas.edu
"""

import os.path
import sys
import glob
import argparse
import numpy as np


def get_file_ext(filename):
    return os.path.splitext(filename)[1]


def load_file(filename, ann_info=None):
    """Examines file type for real/complex and runs appropriate load"""

    if not ann_info:
        ann_info = parse_ann_file(filename)

    # TODO: are there other filetypes we want?
    complex_exts = ['.int', '.mlc', '.slc']
    real_exts = ['.amp', '.cor']  # NOTE: .cor might only be real for UAVSAR

    ext = get_file_ext(filename)

    if ext in complex_exts:
        return load_complex(filename, ann_info)
    elif ext in real_exts:
        return load_real(filename, ann_info)
    else:
        raise ValueError('Invalid filetype for load_file: %s\n '
                         'Allowed types: %s' %
                         (ext, ' '.join(complex_exts + real_exts)))


def load_real(filename, ann_info):
    """Reads in real 4-byte per pixel files""

    Valid filetypes: .amp, .cor (for UAVSAR)
    """
    data = np.fromfile(filename, '<f4')
    return data.reshape([rows, cols])


def load_complex(filename, ann_info):
    """Combines real and imaginary values from a filename to make complex image

    Valid filetypes: .slc, .mlc, .int
    """
    data = np.fromfile(filename, '<f4')
    rows = ann_info['rows']
    cols = ann_info['cols']
    real_data = data[::2].reshape([rows, cols])
    imag_data = data[1::2].reshape([rows, cols])
    return real_data + 1j * imag_data


def save_array(amplitude_array, outfilename):
    """Save the numpy array as a .png file


    amplitude_array (np.array, dtype=float32)
    outfilename (str)
    """

    def _is_little_endian():
        """All UAVSAR data products save in little endian byte order"""
        return sys.byteorder == 'little'

    ext = get_file_ext(outfilename)

    if ext == '.png':
        plt.imsave(
            outfilename,
            amplitude_array,
            cmap='gray',
            vmin=0,
            vmax=0,
            format='.png')

    elif ext in ('.cor', '.amp', '.int', '.mlc', '.slc'):
        # If machine order is big endian, need to byteswap (TODO: test on big-endian)
        if not _is_little_endian():
            amplitude_array.byteswap(inplace=True)

        amplitude_array.tofile(outfilename)
    else:
        raise NotImplementedError("{} saving not implemented.".format(ext))


def make_ann_filename(filename):
    """Take the name of a data file and return corresponding .ann name"""

    # The .mlc files have polarizations added, which .ann files don't have
    shortname = filename.replace('HHHH', '').replace('HVHV', '').replace(
        'VVVV', '')
    ext = get_file_ext(filename)
    return shortname.replace(ext, '.ann')


def parse_ann_file(filename, ext=None):
    """Returns the requested info from the annotation in ann_filename

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

    if get_file_ext(filename) == '.ann' and not ext:
        raise ValueError(
            'parse_ann_file needs ext argument if the data filename not provided.'
        )

    ext = ext or get_file_ext(filename)  # Use what's passed by default
    ann_filename = make_ann_filename(filename)

    ann_data = {}
    line_keywords = {'.slc': 'slc_mag', '.mlc': 'mlc_pwr', '.int': 'slt'}
    row_starts = {k: v + '.set_rows' for k, v in line_keywords.items()}
    col_starts = {k: v + '.set_cols' for k, v in line_keywords.items()}
    row_key = row_starts[ext]
    col_key = col_starts[ext]

    with open(ann_filename, 'r') as f:
        for line in f.readlines():
            # TODO: disambiguate which ones to use, and when
            if line.startswith(row_key):
                ann_data['rows'] = _parse_int(line)
            elif line.startswith(col_key):
                ann_data['cols'] = _parse_int(line)

            # Example: get the name of the mlc for HHHH polarization
            elif line.startswith('mlcHHHH'):
                ann_data['mlcHHHH'] = _parse_line(line)
            # TODO: Add more parsing! whatever is useful from .ann file

    return ann_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename", type=str, help="Specify the input UAVSAR filename")
    args = parser.parse_args()

    ann_data = parse_ann_file(args.filename)
    print(ann_data)


if __name__ == "__main__":
    main()
