#! /usr/bin/env python
"""Author: Scott Staniewicz
Functions to assist input and output of SAR data
Email: scott.stanie@utexas.edu
"""
import collections
import os.path
from pprint import pprint
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

# For UAVSAR:
REAL_POLs = ('HHHH', 'HVHV', 'VVVV')
COMPLEX_POLS = ('HHHV', 'HHVV', 'HVVV')
POLARIZATIONS = REAL_POLs + COMPLEX_POLS


def get_file_ext(filename):
    """Extracts the file extension, including the '.' (e.g.: .slc)"""
    return os.path.splitext(filename)[1]


def load_file(filename, ann_info=None):
    """Examines file type for real/complex and runs appropriate load"""

    if get_file_ext(filename) in ('.hgt', '.dem'):
        return load_elevation(filename)

    if not ann_info:
        ann_info = parse_ann_file(filename)

    if is_complex(filename, ann_info):
        return load_complex(filename, ann_info)
    else:
        return load_real(filename, ann_info)


def load_elevation(filename):
    """Loads a digital elevation map from either .hgt file or .dem

    .hgt is the NASA SRTM files given. Documentation on format here:
    https://dds.cr.usgs.gov/srtm/version2_1/Documentation/SRTM_Topo.pdf
    Key point: Big-endian 2byte integers

    .dem is format used by Zebker geo-coded SAR software
    Only difference is data is stored little-endian (like other SAR data)

    Note on both formats: gaps in coverage are given by INT_MIN -32768,
    so either manually set data(data == np.min(data)) = 0,
        data = np.clip(data, 0, None), or when plotting, plt.imshow(data, vmin=0)
    """

    ext = get_file_ext(filename)
    data_type = "<i2" if ext == '.dem' else ">i2"
    data = np.fromfile(filename, dtype=data_type)
    # Make sure we're working with little endian
    if data_type == '>i2':
        data = data.astype('<i2')

    # Reshape to correct size.
    # Either get info from .dem.rsc
    if ext == '.dem':
        info = load_dem_rsc(filename)
        dem_img = data.reshape((info['file_length'], info['width']))

    # Or check if we are using STRM1 (3601x3601) or SRTM3 (1201x1201)
    else:
        if (data.shape[0] / 3601) == 3601:
            # STRM1- 1 arc second data, 30 meter data
            dem_img = data.reshape((3601, 3601))
        elif (data.shape[0] / 1201) == 1201:
            # STRM3- 3 arc second data, 90 meter data
            dem_img = data.reshape((1201, 1201))
        else:
            raise ValueError("Invalid .hgt data size: must be square size 1201 or 3601")
        # TODO: makeDEM.m did this... do always want?? Why does AWS have so many more
        # negative values in their SRTM1 tile than NASA?
        dem_img = np.clip(dem_img, 0, None)

    return dem_img


def load_dem_rsc(filename):
    """Loads and parses the .dem.rsc file

    Args:
        filename (str) path to either the .dem or .dem.rsc file.
            Function will add .rsc to path if passed .dem file

    example file:
    WIDTH         10801
    FILE_LENGTH   7201
    X_FIRST       -157.0
    Y_FIRST       21.0
    X_STEP        0.000277777777
    Y_STEP        -0.000277777777
    X_UNIT        degrees
    Y_UNIT        degrees
    Z_OFFSET      0
    Z_SCALE       1
    PROJECTION    LL
    """

    # Use OrderedDict so that upsample_dem_rsc creates with same ordering as old
    output_data = collections.OrderedDict()
    # Second part in tuple is used to cast string to correct type
    field_tups = (('WIDTH', int), ('FILE_LENGTH', int), ('X_STEP', float), ('Y_STEP', float),
                  ('X_FIRST', float), ('Y_FIRST', float), ('X_UNIT', str), ('Y_UNIT', str),
                  ('Z_OFFSET', int), ('Z_SCALE', int), ('PROJECTION', str))

    rsc_filename = '{}.rsc'.format(filename) if not filename.endswith('.rsc') else filename
    with open(rsc_filename, 'r') as f:
        for line in f.readlines():
            for field, num_type in field_tups:
                if line.startswith(field):
                    output_data[field] = num_type(line.split()[1])

    return output_data


def load_real(filename, ann_info):
    """Reads in real 4-byte per pixel files""

    Valid filetypes: .amp, .cor (for UAVSAR)
    """
    data = np.fromfile(filename, '<f4')
    # rows = ann_info['rows']
    cols = ann_info['cols']
    return data.reshape([-1, cols])


def is_complex(filename, ann_info):
    """Helper to determine if file data is real or complex

    Based on https://uavsar.jpl.nasa.gov/science/documents/polsar-format.html
    Note: differences between 3 polarizations for .mlc files: half real, half complex
    """
    # TODO: are there other filetypes we want?
    complex_exts = ['.int', '.mlc', '.slc']
    real_exts = ['.amp', '.cor']  # NOTE: .cor might only be real for UAVSAR

    ext = get_file_ext(filename)
    if ext not in complex_exts and ext not in real_exts:
        raise ValueError('Invalid filetype for load_file: %s\n '
                         'Allowed types: %s' % (ext, ' '.join(complex_exts + real_exts)))
    if ext == '.mlc':
        # Check if filename has one of the complex polarizations
        return any(pol in filename for pol in COMPLEX_POLS)
    else:
        return ext in complex_exts


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
        # TODO: Do we need to do this at all??
        if not _is_little_endian():
            amplitude_array.byteswap(inplace=True)

        amplitude_array.tofile(filename)
    else:
        raise NotImplementedError("{} saving not implemented.".format(ext))


# TODO: possibly separate into a "parser" file
def make_ann_filename(filename):
    """Take the name of a data file and return corresponding .ann name"""

    # The .mlc files have polarization added to filename, .ann files don't
    shortname = filename
    for p in POLARIZATIONS:
        shortname = shortname.replace(p, '')
    # If this is a block we split up and names .1.int, remove that since
    # all have the same .ann file

    # TODO: figure out where to get this list from
    ext = get_file_ext(filename)
    shortname = re.sub('\.\d' + ext, ext, shortname)

    return shortname.replace(ext, '.ann')


def parse_ann_file(filename, ext=None, verbose=False):
    """Returns the requested data from the annotation in ann_filename

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

    if verbose:
        pprint(ann_data)
    return ann_data
