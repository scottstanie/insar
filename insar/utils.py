#! /usr/bin/env python
"""Author: Scott Staniewicz
utils.py: Helper functions to prepare and process data
Email: scott.stanie@utexas.edu
"""
from __future__ import division
import collections
import itertools
import glob
import errno
import os
import shutil
import numpy as np
from scipy.ndimage.interpolation import shift
import multiprocessing as mp

import insar.sario
import insar.parsers
from insar.log import get_log, log_runtime

logger = get_log()


def get_file_ext(filename):
    """Extracts the file extension, including the '.' (e.g.: .slc)

    Examples:
        >>> print(get_file_ext('radarimage.slc'))
        .slc
        >>> print(get_file_ext('unwrapped.lowpass.unw'))
        .unw

    """
    return os.path.splitext(filename)[1]


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


def mask_zeros(image):
    """Turn image into masked array, 0s masked"""
    return np.ma.masked_equal(image, 0)


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
    pool.close()


def offset(img_info1, img_info2, axis=None):
    """Calculates how many pixels two images are offset

    Finds offset FROM img_info2 TO img_info1

    If image2 is 3 pixels down and 2 left of image1, the returns would
    be offset(im1, im2) = (3, 2), offset(im1, im2, axis=1) = 2

    To align image2 with image1, you can do:
    offsets = offset(img_info1, img_info2)
    Examples:
    >>> fake_info1 = {'x_first': -155.0, 'x_step': 0.1, 'y_first': 19.5, 'y_step': -0.2}
    >>> fake_info1 = {'x_first': -155.0, 'x_step': 0.1, 'y_first': 19.5, 'y_step': -0.2}

    """
    if img_info1['y_step'] != img_info2['y_step']:
        raise ValueError("Step sizes must be the same for the two images")

    row_offset = (img_info2['y_first'] - img_info1['y_first']) / img_info1['y_step']
    col_offset = (img_info2['x_first'] - img_info1['x_first']) / img_info1['x_step']
    output_tuple = (row_offset, col_offset)
    if axis is None:
        return output_tuple
    else:
        if not isinstance(axis, int):
            raise ValueError("axis must be an int less than 2")
        return output_tuple[axis]


def align_image_pair(image_pair, info_list, verbose=True):
    """Takes two images, shifts the second to align with the first

    Args:
        image_pair (tuple[ndarray, ndarray]): two images to align
        info_list (tuple[dict, dict]): the associated rsc_data/ann_info
            for the two images

    Returns:
        ndarray: shifted version of the 2nd image of image_pair
    """

    cropped_images = crop_to_smallest(image_pair)
    img1, img2 = cropped_images
    img1_ann, img2_ann = info_list

    offset_tup = offset(img1_ann, img2_ann)
    if verbose:
        logger.info("Offset (rows, cols): {}".format(offset_tup))
    # Note: we use order=1 since default order=3 spline was giving
    # negative values for images (leading to invalid nonsense)
    return shift(img2, offset_tup, order=1)


def crop_to_smallest(image_list):
    """Makes all images the smallest dimension so they are alignable

    Args:
        image_list (iterable[ndarray]): list of images, or 3D array
            with 1st axis as the image number
    Returns:
        list[ndarray]: images of all same size

    Example:
    >>> a = np.arange(10).reshape((5, 2))
    >>> b = np.arange(9).reshape((3, 3))
    >>> cropped = crop_to_smallest((a, b))
    >>> print(all(img.shape == (3, 2) for img in cropped))
    True
    """
    shapes = np.array([i.shape for i in image_list])
    min_rows, min_cols = np.min(shapes, axis=0)
    return [img[:min_rows, :min_cols] for img in image_list]


def remove_dupes(lat_lon_list, xyz_list):
    """De-duplicates list of lat/lons with LOS vectors

    Example:
    >>> ll_list = [(1, 2), (1, 2), (1, 2), (3, 4)]
    >>> xyz_list = [(0,0,0), (1,2,3), (0,0,0), (4, 5, 6)]
    >>> ll2, xyz2 = remove_dupes(ll_list, xyz_list)
    >>> ll2
    [(1, 2), (3, 4)]
    >>> xyz2
    [(1, 2, 3), (4, 5, 6)]
    """

    latlons, xyzs = [], []
    idx = -1  # Will increment to 1 upon first append
    for latlon, xyz in zip(lat_lon_list, xyz_list):
        if latlon in latlons:
            # If we added a (0,0,0) vector, check to update it
            if any(xyz) and not any(xyzs[idx]):
                xyzs[idx] = xyz
        else:
            latlons.append(latlon)
            xyzs.append(xyz)
            idx += 1
    return latlons, xyzs


def read_los_output(los_file, dedupe=True):
    """Reads file of x,y,z positions, parses for lat/lon and vectors

    Example line:
     19.0  -155.0
        0.94451263868681301      -0.30776088245682498      -0.11480032487005554
         6399       4259

    Where first line is "gps station position", or "lat lon",
    next line are the 3 LOS vector coordinates to satellite in XYZ,
    next is x position, y position within the DEM grid

    Args:
        los_file (str): Name of file with line of sight vectors
        dedupe (bool): Remove duplicate lat,lon points which may appear
            recorded with 0s from a wrong .db file

    Returns:
        lat_lon_list (list[tuple]): (lat, lon) tuples of points in file
        xyz_list (list[tuple]): (x, y, z) components of line of sight
    """

    def _line_to_floats(line, split_char=None):
        return tuple(map(float, line.split(split_char)))

    with open(los_file) as f:
        los_lines = f.read().splitlines()

    lat_lon_list = [_line_to_floats(line) for line in los_lines[::3]]
    xyz_list = [_line_to_floats(line) for line in los_lines[1::3]]
    return remove_dupes(lat_lon_list, xyz_list) if dedupe else (lat_lon_list, xyz_list)


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

    for ix_step, block in enumerate(blocks, start=1):
        fname = filename.replace(ext, ".{}{}".format(str(ix_step), ext))
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


# Randoms using the sentinelapi
def find_slc_products(api, gj_obj, date_start, date_end, area_relation='contains'):
    """Query for Sentinel 1 SCL products with common options

    from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
    api = api = SentinelAPI(user, pw)
    pecosgeo = geojson_to_wkt(geojson.read_json('pecosBig.geojson'))
    find_slc_products(pecosgeo, '20150101', '20171230')

    Returns:
        OrderedDict: key = '528c0630-bbbf-4a95-8415-c55aa5ce915a', the sentinel
    """
    # area_relation : 'Intersection', 'Contains', 'IsWithin'
    # contains means that the Sentinel footprint completely contains your geojson object
    return api.query(
        gj_obj,
        date=(date_start, date_end),
        platformname='Sentinel-1',
        producttype='SLC',
        area_relation=area_relation)


def show_titles(products):
    return [p['title'] for p in products.values()]


def combine_complex(img1, img2):
    """Combine two complex images which partially overlap

    Used for SLCs/.geos of adjacent Sentinel frames

    Args:
        img1 (ndarray): complex image of first .geo
        img2 (ndarray): complex image of second .geo
    Returns:
        ndarray: Same size as both, with pixels combined
    """
    if img1.shape != img2.shape:
        raise ValueError("img1 and img2 must be same shape to combine.")
    # Start with each one where the other is nonzero
    new_img = np.copy(img1)
    new_img += img2
    # Now only on overlap, take the first's pixels
    overlap_idxs = (img1 != 0) & (img2 != 0)
    new_img[overlap_idxs] = img1[overlap_idxs]

    return new_img


def fullpath(path):
    """Expands ~ and returns an absolute path"""
    return os.path.abspath(os.path.expanduser(path))


def force_symlink(src, dest):
    """python equivalent to 'ln -f -s': force overwrite """
    try:
        os.symlink(fullpath(src), fullpath(dest))
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(fullpath(dest))
            os.symlink(fullpath(src), fullpath(dest))


def rm_if_exists(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise if different error


def stitch_same_dates(geo_path=".", output_path="."):
    """Combines .geo files of the same date in one directory
    """

    def _group_geos_by_date(geolist):
        """Groups into sub-lists sharing dates
        example input:
        [Sentinel S1B, path 78 from 2017-10-13,
         Sentinel S1B, path 78 from 2017-10-13,
         Sentinel S1B, path 78 from 2017-10-25,
         Sentinel S1B, path 78 from 2017-10-25]

        Output:
        [(datetime.date(2017, 10, 13),
          [Sentinel S1B, path 78 from 2017-10-13,
           Sentinel S1B, path 78 from 2017-10-13]),
         (datetime.date(2017, 10, 25),
          [Sentinel S1B, path 78 from 2017-10-25,
           Sentinel S1B, path 78 from 2017-10-25])]

        """
        return [(date, list(g)) for date, g in itertools.groupby(geolist, key=lambda x: x.date)]

    geos = [insar.parsers.Sentinel(g) for g in glob.glob(os.path.join(geo_path, "*.geo"))]
    # Find the dates that have multiple frames/.geos
    date_counts = collections.Counter([g.date for g in geos])
    dates_duped = set([date for date, count in date_counts.items() if count > 1])
    double_geo_files = sorted(
        (g for g in geos if g.date in dates_duped), key=lambda g: g.start_time)
    grouped_geos = _group_geos_by_date(double_geo_files)
    for date, geolist in grouped_geos:
        print("Stitching geos for %s" % date)
        # TODO: Make combine handle more than 2!
        g1, g2 = geolist[:2]

        stitched_img = combine_complex(
            insar.sario.load(g1.filename),
            insar.sario.load(g2.filename),
        )
        new_name = "{}_{}.geo".format(g1.mission, g1.date.strftime("%Y%m%d"))
        new_name = os.path.join(output_path, new_name)
        print("Saving stitched to %s" % new_name)
        # Remove any file with same name before saving
        # This prevents symlink overwriting old files
        rm_if_exists(new_name)
        insar.sario.save(new_name, stitched_img)

    return grouped_geos
