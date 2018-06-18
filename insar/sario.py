#! /usr/bin/env python
"""Author: Scott Staniewicz
Functions to assist input and output of SAR data
Email: scott.stanie@utexas.edu
"""
import collections
import glob
import os
import pprint
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

from insar.log import get_log
logger = get_log()

SENTINEL_EXTS = ['.geo', '.cc', '.int', '.amp', '.unw']
UAVSAR_EXTS = ['.int', '.mlc', '.slc', '.amp', '.cor']

COMPLEX_EXTS = ['.int', '.mlc', '.slc', '.geo', '.cc', '.unw']
REAL_EXTS = ['.amp', '.cor']  # NOTE: .cor might only be real for UAVSAR

# For UAVSAR:
REAL_POLs = ('HHHH', 'HVHV', 'VVVV')
COMPLEX_POLS = ('HHHV', 'HHVV', 'HVVV')
POLARIZATIONS = REAL_POLs + COMPLEX_POLS


def get_file_ext(filename):
    """Extracts the file extension, including the '.' (e.g.: .slc)"""
    return os.path.splitext(filename)[1]


def load_file(filename, rsc_file=None, ann_info=None, verbose=False):
    """Examines file type for real/complex and runs appropriate load
    
    Args:
        filename (str)
        rsc_file (str)
        ann_info (dict)
        verbose (bool): print extra logging info while loading files
    """

    def _find_rsc_file(filename, verbose=False):
        basepath = os.path.split(filename)[0]
        # Should be just elevation.dem.rsc (for .geo folder) or dem.rsc (for igrams)
        possible_rscs = glob.glob(os.path.join(basepath, '*.rsc'))
        if verbose:
            logger.info("Possible rsc files:")
            logger.info(possible_rscs)
        return possible_rscs[0]

    ext = get_file_ext(filename)
    if ext in ('.hgt', '.dem'):
        return load_elevation(filename)

    # Sentinel files should have .rsc file: check for dem.rsc, or elevation.rsc
    if rsc_file:
        rsc_data = load_dem_rsc(rsc_file)
    if ext in SENTINEL_EXTS:
        rsc_file = rsc_file if rsc_file else _find_rsc_file(filename)
        rsc_data = load_dem_rsc(rsc_file)
        if verbose:
            logger.info("Loaded rsc_data from %s", rsc_file)
            logger.info(pprint.pformat(rsc_data))

    # UAVSAR files have an annotation file for metadata
    if not ann_info and not rsc_data and ext in UAVSAR_EXTS:
        ann_info = parse_ann_file(filename, verbose=verbose)

    if is_complex(filename):
        return load_complex(filename, ann_info=ann_info, rsc_data=rsc_data)
    else:
        return load_real(filename, ann_info=ann_info, rsc_data=rsc_data)


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
        dem_img = data.reshape((info['FILE_LENGTH'], info['WIDTH']))

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


def _get_file_width(ann_info=None, rsc_data=None):
    if rsc_data:
        return rsc_data['WIDTH']
    elif ann_info:
        return ann_info['cols']
    else:
        raise ValueError("needs either ann_info or rsc_data to find number of cols")


def load_real(filename, ann_info=None, rsc_data=None):
    """Reads in real 4-byte per pixel files""

    Valid filetypes: See sario.REAL_EXTS
    """
    data = np.fromfile(filename, '<f4')
    # rows = ann_info['rows']
    cols = _get_file_width(ann_info=ann_info, rsc_data=rsc_data)
    return data.reshape([-1, cols])


def load_complex(filename, ann_info=None, rsc_data=None):
    """Combines real and imaginary values from a filename to make complex image

    Valid filetypes: See sario.COMPLEX_EXTS
    """
    data = np.fromfile(filename, '<f4')
    # rows = ann_info['rows']  # Might not ever need rows: just the width
    cols = _get_file_width(ann_info=ann_info, rsc_data=rsc_data)
    real_data, imag_data = parse_complex_data(data, cols)
    return combine_real_imag(real_data, imag_data)


def is_complex(filename):
    """Helper to determine if file data is real or complex

    Uses https://uavsar.jpl.nasa.gov/science/documents/polsar-format.html for UAVSAR
    Note: differences between 3 polarizations for .mlc files: half real, half complex
    """
    ext = get_file_ext(filename)
    if ext not in COMPLEX_EXTS and ext not in REAL_EXTS:
        raise ValueError('Invalid filetype for load_file: %s\n '
                         'Allowed types: %s' % (ext, ' '.join(COMPLEX_EXTS + REAL_EXTS)))
    if ext == '.mlc':
        # Check if filename has one of the complex polarizations
        return any(pol in filename for pol in COMPLEX_POLS)
    else:
        return ext in COMPLEX_EXTS


def parse_complex_data(complex_data, cols):
    """Splits a 1-D array of real/imag bytes to 2 square arrays"""
    # double check if I ever need rows
    real_data = complex_data[::2].reshape([-1, cols])
    imag_data = complex_data[1::2].reshape([-1, cols])
    return real_data, imag_data


def combine_real_imag(real_data, imag_data):
    """Combines two float data arrays into one complex64 array"""
    return real_data + 1j * imag_data


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

    elif ext in ('.cor', '.amp', '.int', '.mlc', '.slc', '.unw'):
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
        ext (str): extension of desired data file, if filename is the .ann file 
            instead of a data filepath
        verbose (bool): print extra logging into about file loading
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
    if verbose:
        logger.info("Trying to load ann_data from %s", ann_filename)
    if not os.path.exists(ann_filename):
        if verbose:
            logger.info("No file found: returning None")
        return None

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
    row_key = row_starts.get(ext)
    col_key = col_starts.get(ext)

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
        logger.info(pprint.pformat(ann_data))
    return ann_data
