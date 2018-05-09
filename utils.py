#! /usr/bin/env python
"""Author: Scott Staniewicz
Helper functions to prepare and process UAVSAR data
Email: scott.stanie@utexas.edu
"""

import argparse
import os.path
import re
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt


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
    rows = ann_info['rows']
    cols = ann_info['cols']
    return data.reshape([rows, cols])


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
        plt.imsave(
            filename,
            amplitude_array,
            cmap='gray',
            vmin=0,
            vmax=1,
            format=ext.strip('.'))

    elif ext in ('.cor', '.amp', '.int', '.mlc', '.slc'):
        # If machine order is big endian, need to byteswap (TODO: test on big-endian)
        if not _is_little_endian():
            amplitude_array.byteswap(inplace=True)

        amplitude_array.tofile(filename)
    else:
        raise NotImplementedError("{} saving not implemented.".format(ext))


def downsample_im(image, rate=10):
    """Takes a numpy matrix of an image and returns a smaller version

    inputs:
        image (np.array) 2D array of an image
        rate (int) the reduction rate to downsample
    """
    return image[::rate, ::rate]


def clip(image):
    """Convert float image to only range 0 to 1 (clips)"""
    return np.clip(np.abs(image), 0, 1)


def log(image):
    """Converts magnitude amplitude image to log scale"""
    return 20 * np.log10(image)


def make_ann_filename(filename):
    """Take the name of a data file and return corresponding .ann name"""

    # The .mlc files have polarizations added, which .ann files don't have
    shortname = filename.replace('HHHH', '').replace('HVHV', '').replace(
        'VVVV', '')
    # If this is a block we split up and names .1.int, remove that since
    # all have the same .ann file
    shortname = re.sub(r'\.\d\.int', '.int', shortname)

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
        raise ValueError(
            'parse_ann_file needs ext argument if the data filename not provided.'
        )

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


def split_array_into_blocks(data):
    """Takes a long rectangular array (like UAVSAR) and creates blocks

    Useful to look at small data pieces at a time in dismph
    Returns:
        blocks (list[np.ndarray])
    """
    rows, cols = data.shape
    blocks = np.array_split(data, rows // cols + 1)
    return blocks


def split_and_save(filename):
    """Creates several files from one long data file

    Saves them with same filename with .1,.2,.3... at end before ext
    e.g. brazos_14937_17087-002_17088-003_0001d_s01_L090HH_01.int produces
        brazos_14937_17087-002_17088-003_0001d_s01_L090HH_01.1.int
        brazos_14937_17087-002_17088-003_0001d_s01_L090HH_01.2.int...

    Output:
        newpaths (list[str]): full paths to new files created 
    """

    data = load_file(filename)
    blocks = split_array_into_blocks(data)

    ext = get_file_ext(filename)
    newpaths = []

    for idx, block in enumerate(blocks, start=1):
        fname = filename.replace(ext, ".{}{}".format(str(idx), ext))
        print("Saving {}".format(fname))
        save_array(fname, block)
        newpaths.append(fname)

    return newpaths


def combine_cor_amp(corfilename, save=True):
    """Takes a .cor file from UAVSAR (which doesn't contain amplitude),
    and creates a new file with amplitude data interleaved for dishgt

    dishgt brazos_14937_17087-002_17088-003_0001d_s01_L090HH_01_withamp.cor 3300 1 5000 1
      where 3300 is number of columns/samples, and we want the first 5000 rows. the final
      1 is needed for the contour interval to set a max of 1 for .cor data

    Inputs:
        corfilename (str): string filename of the .cor from UAVSAR
        save (bool): True if you want to save the combined array

    Returns:
        cor_with_amp (np.ndarray) combined correlation + amplitude (as complex64)
        outfilename (str): same name as corfilename, but _withamp.cor
            Saves a new file under outfilename
    Note: .ann and .int files must be in same directory as .cor
    """
    ext = get_file_ext(corfilename)
    assert ext == '.cor', 'corfilename must be a .cor file'

    intfilename = corfilename.replace('.cor', '.int')

    intdata = load_file(intfilename)
    amp = np.abs(intdata)

    cordata = load_file(corfilename)
    # For dishgt, it expects the two matrices stacked [[amp]; [cor]]
    cor_with_amp = np.vstack((amp, cordata))

    outfilename = corfilename.replace('.cor', '_withamp.cor')
    save_array(outfilename, cor_with_amp)
    return cor_with_amp, outfilename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", type=str, help="Specify command to run on file.")
    parser.add_argument(
        "filename", type=str, help="Specify the input UAVSAR filename")
    args = parser.parse_args()

    if args.command == 'info':
        ann_data = parse_ann_file(args.filename)
        print(ann_data)
    elif args.command == 'split':
        split_and_save(args.filename)


if __name__ == "__main__":
    main()
