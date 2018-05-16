#! /usr/bin/env python
"""Author: Scott Staniewicz
Functions to assist input and output of SAR data
Email: scott.stanie@utexas.edu
"""
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

from utils import get_file_ext


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
                         'Allowed types: %s' % (ext, ' '.join(complex_exts + real_exts)))


def load_real(filename, ann_info):
    """Reads in real 4-byte per pixel files""

    Valid filetypes: .amp, .cor (for UAVSAR)
    """
    data = np.fromfile(filename, '<f4')
    # rows = ann_info['rows']
    cols = ann_info['cols']
    return data.reshape([-1, cols])


def parse_complex_data(complex_data, rows, cols):
    """Splits a 1-D array of real/imag bytes to 2 square arrays"""
    # TODO: double check if I don't need rows all the time
    real_data = complex_data[::2].reshape([-1, cols])
    imag_data = complex_data[1::2].reshape([-1, cols])
    return real_data, imag_data


def combine_real_imag(real_data, imag_data):
    """Combines two float data arrays into one complex64 array"""
    return real_data + 1j * imag_data


def load_complex(filename, ann_info):
    """Combines real and imaginary values from a filename to make complex image

    Valid filetypes: .slc, .mlc, .int
    """
    data = np.fromfile(filename, '<f4')
    rows = ann_info['rows']
    cols = ann_info['cols']
    real_data, imag_data = parse_complex_data(data, rows, cols)
    return combine_real_imag(real_data, imag_data)


def save_array(filename, amplitude_array):
    """Save the numpy array as a .png file

    amplitude_array (np.array, dtype=float32)
    filename (str)
    """

    def _is_little_endian():
        """All UAVSAR data products save in little endian byte order"""
        return sys.byteorder == 'little'

    ext = get_file_ext(filename)

    if ext == '.png':  # TODO: or ext == '.jpg':
        # TODO
        # from PIL import Image
        # im = Image.fromarray(amplitude_array)
        # im.save(filename)
        plt.imsave(filename, amplitude_array, cmap='gray', vmin=0, vmax=1, format=ext.strip('.'))

    elif ext in ('.cor', '.amp', '.int', '.mlc', '.slc'):
        # If machine order is big endian, need to byteswap (TODO: test on big-endian)
        if not _is_little_endian():
            amplitude_array.byteswap(inplace=True)

        amplitude_array.tofile(filename)
    else:
        raise NotImplementedError("{} saving not implemented.".format(ext))


# TODO: possibly separate into a "parser" file
def make_ann_filename(filename):
    """Take the name of a data file and return corresponding .ann name"""

    # The .mlc files have polarizations added, which .ann files don't have
    shortname = filename.replace('HHHH', '').replace('HVHV', '').replace('VVVV', '')
    # If this is a block we split up and names .1.int, remove that since
    # all have the same .ann file
    shortname = re.sub(r'\.\d\.int', '.int', shortname)
    shortname = re.sub(r'\.\d\.cor', '.cor', shortname)

    ext = get_file_ext(filename)
    return shortname.replace(ext, '.ann')


def parse_ann_file(filename, ext=None):
    """Returns the requested info from the annotation in ann_filename

    Returns:
        ann_data (dict): key-values of requested data from .ann file
    """

    def _parse_line(line):
        wordlist = line.split()
        # Pick the entry after the equal sign when splitting the line
        return wordlist[wordlist.index('=') + 1]

    def _parse_int(line):
        return int(_parse_line(line))

    def _parse_float(line):
        return float(_parse_line(line))

    if get_file_ext(filename) == '.ann' and not ext:
        raise ValueError('parse_ann_file needs ext argument if the data filename not provided.')

    ext = ext or get_file_ext(filename)  # Use what's passed by default
    ann_filename = make_ann_filename(filename)

    ann_data = {}
    line_keywords = {
        '.slc': 'slc_mag',
        '.mlc': 'mlc_pwr',
        '.int': 'slt',
        '.cor': 'slt',
        '.amp': 'slt',
    }
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
