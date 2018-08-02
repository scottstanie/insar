#! /usr/bin/env python
"""Author: Scott Staniewicz
Functions to assist input and output of SAR data
Email: scott.stanie@utexas.edu
"""
import collections
import glob
import math
import os
import pprint
import sys
import numpy as np
import matplotlib.pyplot as plt

import insar.parsers
import insar.utils
from insar.log import get_log
logger = get_log()

FLOAT_32_LE = np.dtype('<f4')
INT_16_LE = np.dtype('<i2')
INT_16_BE = np.dtype('>i2')

SENTINEL_EXTS = ['.geo', '.cc', '.int', '.amp', '.unw']
UAVSAR_EXTS = ['.int', '.mlc', '.slc', '.amp', '.cor', '.grd']

# Notes: .grd, .mlc can be either real or complex for UAVSAR,
# .amp files are real only for UAVSAR, complex for sentinel processing
# However, we label them as real here since we can tell .amp files
# are from sentinel if there exists .rsc files in the same dir
COMPLEX_EXTS = ['.int', '.slc', '.geo', '.cc', '.unw', '.mlc', '.grd']
REAL_EXTS = ['.amp', '.cor', '.mlc', '.grd']  # NOTE: .cor might only be real for UAVSAR
# Note about UAVSAR Multi-look products:
# Will end with `_ML5X5.grd`, e.g., for 5x5 downsampled

ELEVATION_EXTS = ['.dem', '.hgt']

# These file types are not simple complex matrices: see load_stacked for detail
STACKED_FILES = ['.cc', '.unw']
# real or complex for these depends on the polarization
UAVSAR_POL_DEPENDENT = ['.grd', '.mlc']


def find_files(directory, search_term):
    """Searches for files in `directory` using globbing on search_term

    Path to file is also included.

    Examples:
    >>> import shutil, tempfile
    >>> temp_dir = tempfile.mkdtemp()
    >>> open(os.path.join(temp_dir, "afakefile.txt"), "w").close()
    >>> print('afakefile.txt' in find_files(temp_dir, "*.txt")[0])
    True
    >>> shutil.rmtree(temp_dir)
    """
    return glob.glob(os.path.join(directory, search_term))


def load_file(filename, downsample=None, rsc_file=None, ann_info=None, verbose=False):
    """Examines file type for real/complex and runs appropriate load

    Args:
        filename (str): path to the file to open
        rsc_file (str): path to a dem.rsc file (if Sentinel)
        ann_info (dict): data parsed from annotation file (UAVSAR)
        downsample (int): rate at which to downsample the file
            None is equivalent to 1, no downsampling.
        verbose (bool): print extra logging info while loading files

    Returns:
        ndarray: a 2D array of the data from a file

    Raises:
        ValueError: if sentinel files loaded without a .rsc file in same path
            to give the file width
    """

    def _find_rsc_file(filename, verbose=False):
        basepath = os.path.split(filename)[0]
        # Should be just elevation.dem.rsc (for .geo folder) or dem.rsc (for igrams)
        possible_rscs = find_files(basepath, '*.rsc')
        if verbose:
            logger.info("Possible rsc files:")
            logger.info(possible_rscs)
        if len(possible_rscs) < 1:
            raise ValueError("{} needs a .rsc file with it for width info.".format(filename))
        return possible_rscs[0]

    if downsample and (downsample < 1 or not isinstance(downsample, int)):
        raise ValueError("downsample must be a positive integer")
    else:
        downsample = downsample or 1

    ext = insar.utils.get_file_ext(filename)
    # Elevation and rsc files can be immediately loaded without extra data
    if ext in ELEVATION_EXTS:
        return load_elevation(filename)[::downsample, ::downsample]
    elif ext == '.rsc':
        return load_dem_rsc(filename)

    # Sentinel files should have .rsc file: check for dem.rsc, or elevation.rsc
    rsc_data = None
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
        u = insar.parsers.Uavsar(filename, verbose=verbose)
        ann_info = u.parse_ann_file()

    if ext in STACKED_FILES:
        return load_stacked(filename, rsc_data)[::downsample, ::downsample]
    # having rsc_data implies that this is not a UAVSAR file, so is complex
    elif rsc_data or is_complex(filename):
        return load_complex(
            filename, ann_info=ann_info, rsc_data=rsc_data)[::downsample, ::downsample]
    else:
        return load_real(filename, ann_info=ann_info, rsc_data=rsc_data)[::downsample, ::downsample]


# Make a shorter alias for load_file
load = load_file


def load_elevation(filename):
    """Loads a digital elevation map from either .hgt file or .dem

    .hgt is the NASA SRTM files given. Documentation on format here:
    https://dds.cr.usgs.gov/srtm/version2_1/Documentation/SRTM_Topo.pdf
    Key point: Big-endian 2 byte (16-bit) integers

    .dem is format used by Zebker geo-coded and ROI-PAC SAR software
    Only difference is data is stored little-endian (like other SAR data)

    Note on both formats: gaps in coverage are given by INT_MIN -32768,
    so either manually set data(data == np.min(data)) = 0,
        data = np.clip(data, 0, None), or when plotting, plt.imshow(data, vmin=0)
    """

    ext = insar.utils.get_file_ext(filename)
    data_type = INT_16_LE if ext == '.dem' else INT_16_BE
    data = np.fromfile(filename, dtype=data_type)
    # Make sure we're working with little endian
    if data_type == INT_16_BE:
        data = data.astype(INT_16_LE)

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
        # TODO: makeDEM.m did this... do we always want this??
        dem_img = np.clip(dem_img, 0, None)

    return dem_img


def load_dem_rsc(filename, lower=False):
    """Loads and parses the .dem.rsc file

    Args:
        filename (str) path to either the .dem or .dem.rsc file.
            Function will add .rsc to path if passed .dem file
        lower (bool): make keys of the dict lowercase

    Returns:
        dict: dem.rsc file parsed out, keys are all caps

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

    if lower:
        output_data = {k.lower(): d for k, d in output_data.items()}
    return output_data


def format_dem_rsc(rsc_dict):
    """Creates the .dem.rsc file string from key/value pairs of an OrderedDict

    Output of function can be written to a file as follows
        with open('my.dem.rsc', 'w') as f:
            f.write(outstring)

    Args:
        rsc_dict (OrderedDict): data about dem in ordered key/value format
            See `load_dem_rsc` output for example

    Returns:
        outstring (str) formatting string to be written to .dem.rsc

    """
    outstring = ""
    for field, value in rsc_dict.items():
        # Files seemed to be left justified with 13 spaces? Not sure why 13
        if field.lower() in ('x_step', 'y_step'):
            # give step floats proper sig figs to not output scientific notation
            outstring += "{field:<14s}{val:0.12f}\n".format(field=field.upper(), val=value)
        else:
            outstring += "{field:<14s}{val}\n".format(field=field.upper(), val=value)

    return outstring


def _get_file_rows_cols(ann_info=None, rsc_data=None):
    """Wrapper function to find file width for different SV types"""
    if (not rsc_data and not ann_info) or (rsc_data and ann_info):
        raise ValueError("needs either ann_info or rsc_data (but not both) to find number of cols")
    elif rsc_data:
        return rsc_data['FILE_LENGTH'], rsc_data['WIDTH']
    elif ann_info:
        return ann_info['rows'], ann_info['cols']


def _assert_valid_size(data, cols):
    """Make sure the width of the image is valid for the data size

    Note that only width is considered- The number of rows is ignored
    """
    error_str = "Invalid number of cols (%s) for file size %s." % (cols, len(data))
    # math.modf returns (fractional remainder, integer remainder)
    assert math.modf(float(len(data)) / cols)[0] == 0, error_str


def load_real(filename, ann_info=None, rsc_data=None):
    """Reads in real 4-byte per pixel files""

    Valid filetypes: See sario.REAL_EXTS

    Args:
        filename (str): path to the file to open
        rsc_data (dict): output from load_dem_rsc, gives width of file
        ann_info (dict): data parsed from UAVSAR annotation file

    Returns:
        ndarray: float32 values for the real 2D matrix

    """
    data = np.fromfile(filename, FLOAT_32_LE)
    rows, cols = _get_file_rows_cols(ann_info=ann_info, rsc_data=rsc_data)
    _assert_valid_size(data, cols)
    return data.reshape([-1, cols])


def load_complex(filename, ann_info=None, rsc_data=None):
    """Combines real and imaginary values from a filename to make complex image

    Valid filetypes: See sario.COMPLEX_EXTS

    Args:
        filename (str): path to the file to open
        rsc_data (dict): output from load_dem_rsc, gives width of file
        ann_info (dict): data parsed from UAVSAR annotation file

    Returns:
        ndarray: imaginary numbers of the combined floats (dtype('complex64'))
    """
    data = np.fromfile(filename, FLOAT_32_LE)
    rows, cols = _get_file_rows_cols(ann_info=ann_info, rsc_data=rsc_data)
    _assert_valid_size(data, cols)

    real_data, imag_data = parse_complex_data(data, cols)
    return combine_real_imag(real_data, imag_data)


def load_stacked(filename, rsc_data, return_amp=False):
    """Helper function to load .unw and .cor files

    Format is two stacked matrices:
        [[first], [second]] where the first "cols" number of floats
        are the first matrix, next "cols" are second, etc.
    For .unw height files, the first is amplitude, second is phase (unwrapped)
    For .cc correlation files, first is amp, second is correlation (0 to 1)

    Args:
        filename (str): path to the file to open
        rsc_data (dict): output from load_dem_rsc, gives width of file
        return_amp (bool): flag to request the amplitude data to be returned

    Returns:
        ndarray: dtype=float32, the second matrix (height, correlation, ...) parsed
        if return_amp == True, returns a tuple (ndarray, ndarray)

    Example illustrating how strips of data alternate:
    reading unw (unwrapped phase) data

    data = np.fromfile('20141128_20150503.unw', '<f4')

    # The first section of data is amplitude data
    # The amplitude has a different, larger range of values
    amp = data[:cols]
    print(np.max(amp), np.min(amp))
    # Output: (27140.396, 118.341095)

    # The next part of the data is a line of phases:
    phase = data[cols:2*cols])
    print(np.max(phase), np.min(phase))
    # Output: (8.011558, -2.6779003)
    """
    data = np.fromfile(filename, FLOAT_32_LE)
    rows, cols = _get_file_rows_cols(rsc_data=rsc_data)
    _assert_valid_size(data, cols)

    first = data.reshape((rows, 2 * cols))[:, :cols]
    second = data.reshape((rows, 2 * cols))[:, cols:]
    if return_amp:
        return first, second
    else:
        return second


def is_complex(filename):
    """Helper to determine if file data is real or complex

    Uses https://uavsar.jpl.nasa.gov/science/documents/polsar-format.html for UAVSAR
    Note: differences between 3 polarizations for .mlc files: half real, half complex
    """
    ext = insar.utils.get_file_ext(filename)
    if ext not in COMPLEX_EXTS and ext not in REAL_EXTS:
        raise ValueError('Invalid filetype for load_file: %s\n '
                         'Allowed types: %s' % (ext, ' '.join(COMPLEX_EXTS + REAL_EXTS)))

    if ext in UAVSAR_POL_DEPENDENT:
        # Check if filename has one of the complex polarizations
        return any(pol in filename for pol in insar.parsers.Uavsar.COMPLEX_POLS)
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


def save(filename, array):
    """Save the numpy array in one of known formats

    Args:
        filename (str) Output path to save file in
        array (ndarray) matrix to save
    Returns:
        None

    Raises:
        NotImplementedError: if file extension of filename not a known ext
    """

    def _is_little_endian():
        """All UAVSAR data products save in little endian byte order"""
        return sys.byteorder == 'little'

    ext = insar.utils.get_file_ext(filename)

    if ext == '.png':  # TODO: or ext == '.jpg':
        # from PIL import Image
        # im = Image.fromarray(array)
        # im.save(filename)
        plt.imsave(filename, array, cmap='gray', vmin=0, vmax=1, format=ext.strip('.'))

    elif (ext in COMPLEX_EXTS + REAL_EXTS + ELEVATION_EXTS) and (ext not in STACKED_FILES):
        # If machine order is big endian, need to byteswap (TODO: test on big-endian)
        # TODO: Do we need to do this at all??
        if not _is_little_endian():
            array.byteswap(inplace=True)

        array.tofile(filename)
    elif ext in STACKED_FILES:
        # TODO
        raise NotImplementedError("{} saving not yet implemented (TODO).".format(ext))
    else:
        raise NotImplementedError("{} saving not implemented.".format(ext))


# TODO: possibly separate into a "parser" file
