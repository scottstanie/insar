#! /usr/bin/env python
"""Author: Scott Staniewicz
Helper functions to prepare and process data
Email: scott.stanie@utexas.edu
"""
from __future__ import division
import glob
from math import floor, sin, cos, sqrt, atan2, radians
import errno
import os
import shutil
import numpy as np
import multiprocessing as mp

import insar.sario
from insar.log import get_log, log_runtime

logger = get_log()


def mkdir_p(path):
    """Emulates bash `mkdir -p`, in python style
    Used for igrams directory creation
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def which(program):
    """Mimics UNIX which

    Used from https://stackoverflow.com/a/377028"""

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def downsample_im(image, rate=10):
    """Takes a numpy matrix of an image and returns a smaller version

    Args:
        image (ndarray) 2D array of an image
        rate (int) the reduction rate to downsample
    """
    return image[::rate, ::rate]


def floor_float(num, ndigits):
    """Like rounding to ndigits, but flooring

    Used for .dem.rsc creation, because rounding to 12 sigfigs
    causes the fortran routines to overstep the matrix and fail,
    since 0.000277777778*3600 = 1.00000000079.. , but
    0.000277777777*3600 = 0.99999999719

    Example:
        >>> floor_float(1/3600, 12)
        0.000277777777
    """
    return floor((10**ndigits) * num) / (10**ndigits)


def clip(image):
    """Convert float image to only range 0 to 1 (clips)"""
    return np.clip(np.abs(image), 0, 1)


def log(image):
    """Converts magnitude amplitude image to log scale"""
    if np.iscomplexobj(image):
        image = np.abs(image)
    return 20 * np.log10(image)


# Alias: convert
db = log


def mag(db_image):
    """Reverse of log/db: decibel to magnitude"""
    return 10**(db_image / 20)


def percent_zero(filepath=None, arr=None):
    """Function to give the percentage of a file that is exactly zero

    Used as a quality assessment check

    Args:
        filepath (str): path to file to check
        arr (ndarray): pre-loaded array to check

    Returns:
        float: decimal from 0 to 1, ratio of zeros to total entries

    Example:
        >>> a = np.array([[1 + 1j, 0.0], [1, 0.0001]])
        >>> print(percent_zero(arr=a))
        0.25
    """
    if filepath:
        arr = insar.sario.load(filepath)
    return (np.sum(arr == 0) / arr.size)


def _check_and_move(fp, zero_threshold, test, mv_dir):
    """Wrapper func for clean_files multiprocessing"""
    logger.debug("Checking {}".format(fp))
    pct = percent_zero(filepath=fp)
    if pct > zero_threshold:
        logger.info("Moving {} for having {:.2f}% zeros to {}".format(fp, 100 * pct, mv_dir))
        if not test:
            shutil.move(fp, mv_dir)


@log_runtime
def clean_files(ext, path=".", zero_threshold=0.50, test=True):
    """Move files of type ext from path with a high pct of zeros

    Args:
        ext (str): file extension to open. Must be loadable by sario.load
        path (str): path of directory to search
        zero_threshold (float): between 0 and 1, threshold to delete files
            if they contain greater ratio of zeros
        test (bool): If true, doesn't delete files, just lists
    """

    file_glob = os.path.join(path, "*{}".format(ext))
    logger.info("Searching {} for files with zero threshold {}".format(file_glob, zero_threshold))

    # Make a folder to store the bad geos
    mv_dir = os.path.join(path, 'bad_{}'.format(ext.replace('.', '')))
    mkdir_p(mv_dir) if not test else logger.info("Test mode: not moving files.")

    max_procs = mp.cpu_count() // 2
    pool = mp.Pool(processes=max_procs)
    results = [
        pool.apply_async(_check_and_move, (fp, zero_threshold, test, mv_dir))
        for fp in glob.glob(file_glob)
    ]
    # Now ask for results so processes launch
    [res.get() for res in results]


def split_array_into_blocks(data):
    """Takes a long rectangular array (like UAVSAR) and creates blocks

    Useful to look at small data pieces at a time in dismph

    Returns:
        blocks (list[np.ndarray])
    """
    rows, cols = data.shape
    blocks = np.array_split(data, np.ceil(rows / cols))
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

    data = insar.sario.load_file(filename)
    blocks = split_array_into_blocks(data)

    ext = insar.sario.get_file_ext(filename)
    newpaths = []

    for idx, block in enumerate(blocks, start=1):
        fname = filename.replace(ext, ".{}{}".format(str(idx), ext))
        print("Saving {}".format(fname))
        insar.sario.save(fname, block)
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
    ext = insar.sario.get_file_ext(corfilename)
    assert ext == '.cor', 'corfilename must be a .cor file'

    intfilename = corfilename.replace('.cor', '.int')

    intdata = insar.sario.load_file(intfilename)
    amp = np.abs(intdata)

    cordata = insar.sario.load_file(corfilename)
    # For dishgt, it expects the two matrices stacked [[amp]; [cor]]
    cor_with_amp = np.vstack((amp, cordata))

    outfilename = corfilename.replace('.cor', '_withamp.cor')
    insar.sario.save(outfilename, cor_with_amp)
    return cor_with_amp, outfilename


def sliding_window_view(x, shape, step=None):
    """
    Create sliding window views of the N dimensions array with the given window
    shape. Window slides across each dimension of `x` and provides subsets of `x`
    at any window position.

    Adapted from https://github.com/numpy/numpy/pull/10771

    Args:
        x (ndarray): Array to create sliding window views.
        shape (sequence of int): The shape of the window.
            Must have same length as number of input array dimensions.
        step: (sequence of int), optional
            The steps of window shifts for each dimension on input array at a time.
            If given, must have same length as number of input array dimensions.
            Defaults to 1 on all dimensions.
    Returns:
        ndarray: Sliding window views (or copies) of `x`.
            view.shape = (x.shape - shape) // step + 1

    Notes
    -----
    ``sliding_window_view`` create sliding window views of the N dimensions array
    with the given window shape and its implementation based on ``as_strided``.
    The returned views are *readonly* due to the numpy sliding tricks.
    Examples
    --------
    >>> i, j = np.ogrid[:3,:4]
    >>> x = 10*i + j
    >>> shape = (2,2)
    >>> sliding_window_view(x, shape)[0, 0]
    array([[ 0,  1],
           [10, 11]])
    >>> sliding_window_view(x, shape)[1, 2]
    array([[12, 13],
           [22, 23]])
    """
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False)

    try:
        shape = np.array(shape, np.int)
    except ValueError:
        raise TypeError('`shape` must be a sequence of integer')
    else:
        if shape.ndim > 1:
            raise ValueError('`shape` must be one-dimensional sequence of integer')
        if len(x.shape) != len(shape):
            raise ValueError("`shape` length doesn't match with input array dimensions")
        if np.any(shape <= 0):
            raise ValueError('`shape` cannot contain non-positive value')

    if step is None:
        step = np.ones(len(x.shape), np.intp)
    else:
        try:
            step = np.array(step, np.intp)
        except ValueError:
            raise TypeError('`step` must be a sequence of integer')
        else:
            if step.ndim > 1:
                raise ValueError('`step` must be one-dimensional sequence of integer')
            if len(x.shape) != len(step):
                raise ValueError("`step` length doesn't match with input array dimensions")
            if np.any(step <= 0):
                raise ValueError('`step` cannot contain non-positive value')

    o = (np.array(x.shape) - shape) // step + 1  # output shape
    if np.any(o <= 0):
        raise ValueError('window shape cannot larger than input array shape')

    strides = x.strides
    view_strides = strides * step

    view_shape = np.concatenate((o, shape), axis=0)
    view_strides = np.concatenate((view_strides, strides), axis=0)
    view = np.lib.stride_tricks.as_strided(x, view_shape, view_strides, writeable=False)

    return view


def latlon_to_dist(lat_lon_start, lat_lon_end, R=6378):
    """Find the distance between two lat/lon points on Earth

    Uses the haversine formula: https://en.wikipedia.org/wiki/Haversine_formula
    so it does not account for the ellopsoidal Earth shape. Will be with about
    0.5-1% of the correct value.

    Notes: lats and lons are in degrees, and the values used for R Earth
    (6373 km) are optimized for locations around 39 degrees from the equator

    Reference: https://andrew.hedges.name/experiments/haversine/

    Args:
        lat_lon_start (tuple[int, int]): (lat, lon) in degrees of start
        lat_lon_end (tuple[int, int]): (lat, lon) in degrees of end
        R (float): Radius of earth

    Returns:
        float: distance between two points in km

    Examples:
        >>> round(latlon_to_dist((38.8, -77.0), (38.9, -77.1)), 1)
        14.1
    """
    lat1, lon1 = lat_lon_start
    lat2, lon2 = lat_lon_end
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = (sin(dlat / 2)**2) + (cos(lat1) * cos(lat2) * sin(dlon / 2)**2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c
