"""prepare.py

Preprocessing insar data for timeseries analysis

Forms stacks as .h5 files for easy access to depth-wise slices
"""
import h5py
import os
import re
import datetime
import numpy as np
from scipy.ndimage.filters import uniform_filter

from sardem.loading import load_dem_rsc
from apertools.parsers import Sentinel
from apertools import sario, utils, latlon
from insar import mask
import apertools.gps
from apertools.log import get_log

SENTINEL_WAVELENGTH = 5.5465763  # cm
PHASE_TO_CM = SENTINEL_WAVELENGTH / (-4 * np.pi)
DATE_FMT = "%Y%m%d"
logger = get_log()


def prepare_stacks(igram_path, geolist_ignore_file="geolist_missing.txt"):
    geolist, intlist = load_geolist_intlist(igram_path,
                                            geolist_ignore_file=geolist_ignore_file,
                                            parse=True)

    # Prepare B matrix and timediffs used for each pixel inversion
    timediffs = find_time_diffs(geolist)

    logger.debug("Reading unw stack")
    unw_stack, mask_stack, geo_mask_columns = load_unw_masked_stack(
        igram_path,
        num_timediffs=len(timediffs),
        unw_ext='.unw',
        deramp=deramp,
        deramp_order=deramp_order,
        masking=masking,
    )

    # TODO: Process the correlation, mask very bad corr pixels in the igrams

    # Use the given reference, or find one on based on max correlation
    if any(r is None for r in reference):
        # Make a latlon image to check for gps data containment
        # TODO: maybe i need to search for masks? dont wanna pick a garbage one by accident
        latlon_image = latlon.LatlonImage(data=unw_stack[0],
                                          dem_rsc_file=os.path.join(igram_path, 'dem.rsc'))
        ref_row, ref_col = find_reference_location(latlon_image,
                                                   igram_path,
                                                   mask_stack,
                                                   gps_dir=None)
    else:
        ref_row, ref_col = reference

    logger.info("Starting shift_stack: using %s, %s as ref_row, ref_col", ref_row, ref_col)
    unw_stack = shift_stack(unw_stack, ref_row, ref_col, window=window)
    logger.info("Shifting stack complete")


def create_hdf5_stack(outfile_name=None,
                      compression=None,
                      file_list=None,
                      directory=None,
                      file_ext=None,
                      **kwargs):
    """Make stack as hdf5 file from a list of files

    Args:
        outfile_name (str): if none provided, creates a file `[file_ext]_stack.h5`

    Returns:
        outfile_name
    """
    if not outfile_name:
        if not file_ext:
            file_ext = utils.get_file_ext(file_list[0])
        outfile_name = "{fext}_stack.h5".format(fext=file_ext.strip("."))
        logger.info("Creating stack file %s" % outfile_name)

    if utils.get_file_ext(outfile_name) not in (".h5", ".hdf5"):
        raise ValueError("outfile_name must end in .h5 or .hdf5")

    # TODO: do we want to replace the .unw files with .h5 files, then make a Virtual dataset?
    # layout = h5py.VirtualLayout(shape=(len(file_list), nrows, ncols), dtype=dtype)
    if file_list is None:
        file_list = _load_stack_files(directory, file_ext)

    stack = load_stack(file_list=file_list, **kwargs)
    with h5py.File(outfile_name, "a") as hf:
        hf.create_dataset(
            "stack",
            data=stack,
        )
        hf.create_dataset(
            "mean_stack",
            data=np.mean(stack, axis=0),
        )
        # vsource = h5py.VirtualSource()

    return outfile_name


def _parse(datestr):
    return datetime.datetime.strptime(datestr, DATE_FMT).date()


def _strip_geoname(name):
    """Leaves just date from format S1A_YYYYmmdd.geo"""
    return name.replace('S1A_', '').replace('S1B_', '').replace('.geo', '')


def read_geolist(filepath="./geolist", parse=True):
    """Reads in the list of .geo files used, in time order

    Args:
        filepath (str): path to the geolist file or directory
        parse (bool): default True: output the geolist as parsed datetimes

    Returns:
        list[date]: the parse dates of each .geo used, in date order

    """
    if os.path.isdir(filepath):
        filepath = os.path.join(filepath, 'geolist')

    with open(filepath) as f:
        if not parse:
            return [fname for fname in f.read().splitlines()]
        else:
            # Stripped of path for parser
            geolist = [os.path.split(geoname)[1] for geoname in f.read().splitlines()]

    if re.match(r'S1[AB]_\d{8}\.geo', geolist[0]):  # S1A_YYYYmmdd.geo
        return sorted([_parse(_strip_geoname(geo)) for geo in geolist])
    else:  # Full sentinel product name
        return sorted([Sentinel(geo).start_time.date() for geo in geolist])


def read_intlist(filepath="./intlist", parse=True):
    """Reads the list of igrams to return dates of images as a tuple

    Args:
        filepath (str): path to the intlist directory, or file
        parse (bool): output the intlist as parsed datetime tuples

    Returns:
        tuple(date, date) of master, slave dates for all igrams (if parse=True)
            if parse=False: returns list[str], filenames of the igrams

    """

    if os.path.isdir(filepath):
        filepath = os.path.join(filepath, 'intlist')

    with open(filepath) as f:
        intlist = f.read().splitlines()

    if parse:
        intlist = [intname.strip('.int').split('_') for intname in intlist]
        return [(_parse(master), _parse(slave)) for master, slave in intlist]
    else:
        dirname = os.path.dirname(filepath)
        return [os.path.join(dirname, igram) for igram in intlist]


def load_geolist_intlist(filepath, geolist_ignore_file=None, parse=True):
    """Load the geolist and intlist from a directory with igrams"""
    geolist = read_geolist(filepath, parse=parse)
    intlist = read_intlist(filepath, parse=parse)
    if geolist_ignore_file is not None:
        ignore_filepath = os.path.join(filepath, geolist_ignore_file)
        geolist, intlist = ignore_geo_dates(geolist,
                                            intlist,
                                            ignore_file=ignore_filepath,
                                            parse=parse)
    return geolist, intlist


def ignore_geo_dates(geolist, intlist, ignore_file="geolist_missing.txt", parse=True):
    """Read extra file to ignore certain dates of interferograms"""
    ignore_geos = set(read_geolist(ignore_file, parse=parse))
    logger.info("Ignoreing the following .geo dates:")
    logger.info(sorted(ignore_geos))
    valid_geos = [g for g in geolist if g not in ignore_geos]
    valid_igrams = [i for i in intlist if i[0] not in ignore_geos and i[1] not in ignore_geos]
    return valid_geos, valid_igrams


def find_time_diffs(geolist):
    """Finds the number of days between successive .geo files

    Args:
        geolist (list[date]): dates of the .geo SAR acquisitions

    Returns:
        np.array: days between each datetime in geolist
            dtype=int, length is a len(geolist) - 1"""
    return np.array([difference.days for difference in np.diff(geolist)])


def shift_stack(stack, ref_row, ref_col, window=3, window_func=np.mean):
    """Subtracts reference pixel group from each layer

    Args:
        stack (ndarray): 3D array of images, stacked along axis=0
        ref_row (int): row index of the reference pixel to subtract
        ref_col (int): col index of the reference pixel to subtract
        window (int): size of the group around ref pixel to avg for reference.
            if window=1 or None, only the single pixel used to shift the group.
        window_func (str): default='mean', choices ='max', 'min', 'mean'
            numpy function to use on window. With 'mean', takes the mean of the
            window and subtracts value from rest of layer.

    Raises:
        ValueError: if window is not a positive int, or if ref pixel out of bounds
    """
    win = window // 2
    for idx, layer in enumerate(stack):
        patch = layer[ref_row - win:ref_row + win + 1, ref_col - win:ref_col + win + 1]
        stack[idx] -= np.mean(patch)  # yapf: disable

    return stack
    # means = apertools.utils.window_stack(stack, ref_row, ref_col, window, window_func)
    # return stack - means[:, np.newaxis, np.newaxis]  # pad with axes to broadcast


def _geolist_to_str(geolist):
    return [d.strftime(DATE_FMT) for d in geolist]


def _intlist_to_str(intlist):
    return [(a.strftime(DATE_FMT), b.strftime(DATE_FMT)) for a, b in intlist]


def save_deformation(igram_path,
                     deformation,
                     geolist,
                     defo_name='deformation.h5',
                     geolist_name='geolist.npy'):
    """Saves deformation ndarray and geolist dates as .npy file"""
    np.save(os.path.join(igram_path, defo_name), deformation)
    np.save(os.path.join(igram_path, geolist_name), geolist)


def matrix_indices(shape, flatten=True):
    """Returns a pair of vectors for all indices of a 2D array

    Convenience function to help remembed mgrid syntax

    Example:
        >>> a = np.arange(12).reshape((4, 3))
        >>> print(a)
        [[ 0  1  2]
         [ 3  4  5]
         [ 6  7  8]
         [ 9 10 11]]
        >>> rs, cs = matrix_indices(a.shape)
        >>> rs
        array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        >>> cs
        array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        >>> print(a[rs[1], cs[1]] == a[0, 1])
        True
    """
    nrows, ncols = shape
    row_block, col_block = np.mgrid[0:nrows, 0:ncols]
    if flatten:
        return row_block.flatten(), col_block.flatten()
    else:
        return row_block, col_block


def load_unw_masked_stack(igram_path,
                          num_timediffs=None,
                          unw_ext='.unw',
                          deramp=True,
                          deramp_order=1,
                          masking=True):

    int_file_names = read_intlist(igram_path, parse=False)
    if deramp:
        # Deramp each .unw file
        # TODO: do I ever really want 2nd order?
        # # For larger areas, use quadratic ramp. Otherwise, linear
        # max_linear = 20  # km
        # order = 1 if (width < max_linear or height < max_linear) else 2

        width, height = latlon.grid_size(**load_dem_rsc(os.path.join(igram_path, 'dem.rsc')))
        logger.info("Dem size %.2f by %.2f km: using order %s surface to deramp", width, height,
                    deramp_order)
        unw_stack = deramp_stack(int_file_names, unw_ext, order=deramp_order)
    else:
        unw_file_names = [f.replace('.int', unw_ext) for f in int_file_names]
        unw_stack = sario.load_stack(file_list=unw_file_names)

    unw_stack = unw_stack.view(np.ma.MaskedArray)

    if masking:
        int_mask_file_names = [n + '.mask.npy' for n in int_file_names]
        if not all(os.path.exists(f) for f in int_mask_file_names):
            logger.info("Creating and saving igram masks")
            row_looks, col_looks = utils.find_looks_taken(igram_path)
            create_igram_masks(igram_path, row_looks=row_looks, col_looks=col_looks)

        # Note: using .geo masks as well as .int masks since valid data in one
        # .geo produces non-zero igram, but it is still garbage

        # need to specify file_list or the order is different from read_intlist
        logger.info("Reading igram masks")
        mask_stack = sario.load_stack(file_list=int_mask_file_names)
        unw_stack.mask = mask_stack
    else:
        mask_stack = np.full_like(unw_stack, False, dtype=bool)
        num_ints, rows, cols = unw_stack.shape
        geo_mask_columns = np.full((num_timediffs, rows * cols), False)

    logger.info("Finished loading unw_stack")
    return unw_stack, mask_stack, geo_mask_columns


def _estimate_ramp(z, order):
    """Takes a 2D array an fits a linear plane to the data

    Args:
        z (ndarray): 2D array, interpreted as heights
        order (int): degree of surface estimation
            order = 1 removes linear ramp, order = 2 fits quadratic surface
        order (int)

    Returns:
        ndarray: the estimated coefficients of the surface
            For order = 1, it will be 3 numbers, a, b, c from
                 ax + by + c = z
            For order = 2, it will be 6:
                f + ax + by + cxy + dx^2 + ey^2
    """
    if order > 2:
        raise ValueError("Order only implemented for 1 and 2")
    # Note: rows == ys, cols are xs
    yidxs, xidxs = matrix_indices(z.shape, flatten=True)
    # c_ stacks 1D arrays as columns into a 2D array
    if order == 1:
        A = np.c_[np.ones(xidxs.shape), xidxs, yidxs]
    elif order == 2:
        A = np.c_[np.ones(xidxs.shape), xidxs, yidxs, xidxs * yidxs, xidxs**2, yidxs**2]

    coeffs, _, _, _ = np.linalg.lstsq(A, z.flatten(), rcond=None)
    # coeffs will be a, b, c in the equation z = ax + by + c
    return coeffs


def remove_ramp(z, order=1):
    """Estimates a linear plane through data and subtracts to flatten

    Used to remove noise artifacts from unwrapped interferograms

    Args:
        z (ndarray): 2D array, interpreted as heights
        order (int): degree of surface estimation
            order = 1 removes linear ramp, order = 2 fits quadratic surface

    Returns:
        ndarray: flattened 2D array with estimated surface removed
    """
    coeffs = _estimate_ramp(z, order)
    if order == 1:
        # We want full blocks, as opposed to matrix_index flattened
        c, a, b = coeffs
        y_block, x_block = matrix_indices(z.shape, flatten=False)
        return z - (a * x_block + b * y_block + c)
    elif order == 2:
        yy, xx = matrix_indices(z.shape, flatten=True)
        idx_matrix = np.c_[np.ones(xx.shape), xx, yy, xx * yy, xx**2, yy**2]
        z_fit = np.dot(idx_matrix, coeffs).reshape(z.shape)
        return z - z_fit


def deramp_stack(int_file_list, unw_ext, order=1):
    """Handles removing linear ramps for all files in a stack

    Saves the files to a ".unwflat" version

    Args:
        inf_file_list (list[str]): names of .int files
        unw_ext (str): file extension of unwrapped igrams (usually .unw)
        order (int): order of polynomial surface to use to deramp
            1 is linear, 2 is quadratic
    """
    logger.info("Removing any ramp from each stack layer")
    # Get file names to save results/ check if we deramped already
    flat_ext = unw_ext + 'flat'

    unw_file_names = [f.replace('.int', unw_ext) for f in int_file_list]
    unw_file_names = [f for f in unw_file_names if flat_ext not in f]
    flat_file_names = [filename.replace(unw_ext, flat_ext) for filename in unw_file_names]

    if not all(os.path.exists(f) for f in flat_file_names):
        logger.info("Deramped files don't exist: Creating %s files and and saving." % flat_ext)
        nrows, ncols = sario.load(unw_file_names[0]).shape
        out_stack = np.empty((len(unw_file_names), nrows, ncols), dtype=sario.FLOAT_32_LE)
        # Shape of sario.load_stack with return_amp is (nlayers, 2, nrows, ncols)
        for idx, (amp, height) in enumerate(
                (sario.load(f, return_amp=True) for f in unw_file_names)):  # yapf: disable
            # return_amp gives a 3D ndarray, [amp, height]
            r = remove_ramp(height, order=order)
            new_unw = np.stack((amp, r), axis=0)
            sario.save(flat_file_names[idx], new_unw)
            out_stack[idx] = r
        return out_stack
    else:
        logger.info("Loading previous %s deramped files." % flat_ext)
        return sario.load_stack(file_list=flat_file_names)


def find_reference_location(latlon_image, igram_path=None, mask_stack=None, gps_dir=None):
    ref_row, ref_col = None, None
    logger.info("Searching for gps station within area")
    stations = apertools.gps.stations_within_image(latlon_image, mask_invalid=True, gps_dir=gps_dir)
    if len(stations) > 0:
        # TODO: pick best station somehow? maybe higher mean correlation?
        logger.info("Station options:")
        logger.info(stations)

        name, lon, lat = stations[0]
        logger.info("Using station %s at (lon, lat) (%s, %s)", name, lon, lat)
        ref_row, ref_col = latlon_image.nearest_pixel(lon=lon, lat=lat)

    if ref_row is None:
        logger.warning("GPS station search failed, reverting to coherence")
        logger.info("Finding most coherent patch in stack.")
        cc_stack = sario.load_stack(directory=igram_path, file_ext=".cc")
        cc_stack = np.ma.array(cc_stack, mask=mask_stack)
        ref_row, ref_col = find_coherent_patch(cc_stack)
        logger.info("Using %s as .unw reference point", (ref_row, ref_col))
        del cc_stack  # In case of big memory

    return ref_row, ref_col


def find_coherent_patch(correlations, window=11):
    """Looks through 3d stack of correlation layers and finds strongest correlation patch

    Also accepts a 2D array of the pre-compute means of the 3D stack.
    Uses a window of size (window x window), finds the largest average patch

    Args:
        correlations (ndarray, possibly masked): 3D array of correlations:
            correlations = sario.load_stack('path/to/correlations', '.cc')

        window (int): size of the patch to consider

    Returns:
        tuple[int, int]: the row, column of center of the max patch

    Example:
        >>> corrs = np.arange(25).reshape((5, 5))
        >>> print(find_coherent_patch(corrs, window=3))
        (3, 3)
        >>> corrs = np.stack((corrs, corrs), axis=0)
        >>> print(find_coherent_patch(corrs, window=3))
        (3, 3)
    """
    correlations = correlations.view(np.ma.MaskedArray)  # Force to be type np.ma
    if correlations.ndim == 2:
        mean_stack = correlations
    elif correlations.ndim == 3:
        mean_stack = np.ma.mean(correlations, axis=0)
    else:
        raise ValueError("correlations must be a 2D mean array, or 3D correlations")

    # Run a 2d average over the image, then convert to masked array
    conv = uniform_filter(mean_stack, size=window, mode='constant')
    conv = np.ma.array(conv, mask=correlations.mask.any(axis=0))
    # Now find row, column of the max value
    max_idx = conv.argmax()
    return np.unravel_index(max_idx, mean_stack.shape)


def create_igram_masks(igram_path, row_looks=1, col_looks=1):
    intlist = read_intlist(filepath=igram_path)
    int_file_names = read_intlist(filepath=igram_path, parse=False)
    geolist = read_geolist(filepath=igram_path)

    geo_path = os.path.dirname(os.path.abspath(igram_path))
    mask.save_int_masks(
        int_file_names,
        intlist,
        geolist,
        geo_path=geo_path,
        row_looks=row_looks,
        col_looks=col_looks,
    )
    return
