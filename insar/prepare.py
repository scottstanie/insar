"""prepare.py

Preprocessing insar data for timeseries analysis

Forms stacks as .h5 files for easy access to depth-wise slices
"""
import h5py
import json
import os
import numpy as np
from scipy.ndimage.filters import uniform_filter

from apertools import sario, utils, latlon
import apertools.gps
from apertools.log import get_log, log_runtime

logger = get_log()

DATE_FMT = "%Y%m%d"

MASK_FILENAME = "masks.h5"
INT_FILENAME = "int_stack.h5"
UNW_FILENAME = "unw_stack.h5"
CC_FILENAME = "cc_stack.h5"

# dataset names for general 3D stacks
STACK_DSET = "stack"
STACK_MEAN_DSET = "mean_stack"
STACK_FLAT_DSET = "deramped_stack"
STACK_FLAT_SHIFTED_DSET = "deramped_shifted_stack"

# Mask file datasets
GEO_MASK_DSET = "geo"
GEO_MASK_SUM_DSET = "geo_sum"
IGRAM_MASK_DSET = "igram"
IGRAM_MASK_SUM_DSET = "igram_sum"

DEM_RSC_DSET = "dem_rsc"

GEOLIST_DSET = "geo_dates"
INTLIST_DSET = "int_dates"


@log_runtime
def prepare_stacks(
        igram_path,
        overwrite=False,
        gps_dir=None,
):
    int_stack_file = os.path.join(igram_path, INT_FILENAME)
    unw_stack_file = os.path.join(igram_path, UNW_FILENAME)
    cc_stack_file = os.path.join(igram_path, CC_FILENAME)
    mask_stack_file = os.path.join(igram_path, MASK_FILENAME)

    create_igram_stacks(
        igram_path,
        int_stack_file=int_stack_file,
        unw_stack_file=unw_stack_file,
        cc_stack_file=cc_stack_file,
        overwrite=overwrite,
    )

    create_mask_stacks(igram_path, overwrite=overwrite)
    deramp_stack(unw_stack_file=unw_stack_file, order=1, overwrite=overwrite)

    ref_row, ref_col = find_reference_location(
        unw_stack_file=unw_stack_file,
        cc_stack_file=cc_stack_file,
        mask_stack_file=mask_stack_file,
        gps_dir=gps_dir,
    )

    shift_unw_file(unw_stack_file=unw_stack_file,
                   ref_row=ref_row,
                   ref_col=ref_col,
                   window=3,
                   overwrite=overwrite)


@log_runtime
def create_igram_stacks(
        igram_path,
        int_stack_file=INT_FILENAME,
        unw_stack_file=UNW_FILENAME,
        cc_stack_file=CC_FILENAME,
        overwrite=False,
):
    stack_dicts = (
        dict(file_ext=".int", create_mean=False, filename=int_stack_file),
        dict(file_ext=".unw", create_mean=False, filename=unw_stack_file),
        dict(file_ext=".cc", create_mean=True, filename=cc_stack_file),
    )
    for d in stack_dicts:
        if d["filename"] is None:
            continue
        logger.info("Creating hdf5 stack %s" % d["filename"])
        create_hdf5_stack(directory=igram_path, overwrite=overwrite, **d)
        store_geolist(igram_path, d["filename"], overwrite=overwrite)
        store_intlist(igram_path, d["filename"], overwrite=overwrite)


@log_runtime
def create_mask_stacks(igram_path, mask_filename=None, geo_path=None, overwrite=False):
    """Create mask stacks for areas in .geo and .int

    Uses .geo dead areas as well as correlation
    """
    if mask_filename is None:
        mask_file = os.path.join(igram_path, MASK_FILENAME)

    if geo_path is None:
        geo_path = utils.get_parent_dir(igram_path)

    # Used to shrink the .geo masks to save size as .int masks
    row_looks, col_looks = apertools.utils.find_looks_taken(igram_path, geo_path=geo_path)

    dem_rsc = sario.load(sario.find_rsc_file(directory=igram_path))
    save_dem_to_h5(mask_file, dem_rsc, dset_name=DEM_RSC_DSET, overwrite=overwrite)
    store_geolist(igram_path, mask_file, overwrite=overwrite)
    store_intlist(igram_path, mask_file, overwrite=overwrite)

    # if create_geos:
    save_geo_masks(
        geo_path,
        mask_file,
        dem_rsc=dem_rsc,
        row_looks=row_looks,
        col_looks=col_looks,
        overwrite=overwrite,
    )

    int_date_list = sario.find_igrams(directory=igram_path)
    int_file_list = sario.find_igrams(directory=igram_path, parse=False)

    geo_date_list = sario.find_geos(directory=geo_path)
    geo_file_list = sario.find_geos(directory=geo_path, parse=False)

    compute_int_masks(
        mask_file=mask_file,
        int_file_list=int_file_list,
        int_date_list=int_date_list,
        geo_file_list=geo_file_list,
        geo_date_list=geo_date_list,
        row_looks=row_looks,
        col_looks=col_looks,
        dem_rsc=dem_rsc,
        overwrite=overwrite,
    )
    # TODO: now add the correlation check
    return mask_file


def save_geo_masks(directory,
                   mask_file=MASK_FILENAME,
                   dem_rsc=None,
                   dset_name=GEO_MASK_DSET,
                   row_looks=1,
                   col_looks=1,
                   overwrite=False):
    """Creates .mask files for geos where zeros occur

    Makes look arguments are to create arrays the same size as the igrams
    Args:
        overwrite (bool): erase the dataset from the file if it exists and recreate
    """

    def _get_geo_mask(geo_arr):
        return np.ma.make_mask(geo_arr == 0, shrink=False)

    geo_file_list = sario.find_files(directory=directory, search_term="*.geo")
    # Make the empty stack, or delete if exists
    shape = _find_file_shape(dem_rsc=dem_rsc,
                             file_list=geo_file_list,
                             row_looks=row_looks,
                             col_looks=col_looks)
    if not _check_dset(mask_file, dset_name, overwrite):
        return
    _create_dset(mask_file, dset_name, shape=shape, dtype=bool)

    with h5py.File(mask_file) as f:
        dset = f[dset_name]
        for idx, geo_fname in enumerate(geo_file_list):
            g = sario.load(geo_fname, looks=(row_looks, col_looks))
            # ipdb.set_trace()
            print('Saving %s to stack' % geo_fname)
            dset[idx] = _get_geo_mask(g)

        # Also add a composite mask depthwise
        f[GEO_MASK_SUM_DSET] = np.sum(dset, axis=0)


def compute_int_masks(
        mask_file=None,
        int_file_list=None,
        int_date_list=None,
        geo_file_list=None,
        geo_date_list=None,
        row_looks=None,
        col_looks=None,
        dem_rsc=None,
        dset_name=IGRAM_MASK_DSET,
        overwrite=False,
):
    """Creates igram masks by taking the logical-or of the two .geo files

    Assumes save_geo_masks already run
    """
    # Make the empty stack, or delete if exists
    shape = _find_file_shape(dem_rsc=dem_rsc,
                             file_list=int_file_list,
                             row_looks=row_looks,
                             col_looks=col_looks)
    if not _check_dset(mask_file, dset_name, overwrite):
        return
    if not _check_dset(mask_file, IGRAM_MASK_SUM_DSET, overwrite):
        return

    _create_dset(mask_file, dset_name, shape=shape, dtype=bool)

    with h5py.File(mask_file, "a") as f:
        geo_mask_stack = f[GEO_MASK_DSET]
        int_mask_dset = f[dset_name]
        for idx, (early, late) in enumerate(int_date_list):
            early_idx = geo_date_list.index(early)
            late_idx = geo_date_list.index(late)
            early_mask = geo_mask_stack[early_idx]
            late_mask = geo_mask_stack[late_idx]

            int_mask_dset[idx] = np.logical_or(early_mask, late_mask)

        # Also create one image of the total masks
        f[IGRAM_MASK_SUM_DSET] = np.sum(int_mask_dset, axis=0)


def create_hdf5_stack(filename=None,
                      directory=None,
                      file_ext=None,
                      create_mean=True,
                      save_rsc=True,
                      overwrite=False,
                      **kwargs):
    """Make stack as hdf5 file from a group of existing files

    Args:
        filename (str): if none provided, creates a file `[file_ext]_stack.h5`

    Returns:
        filename
    """
    if not filename:
        fname = "{fext}_stack.h5".format(fext=file_ext.strip("."))
        filename = os.path.abspath(os.path.join(directory, fname))
        logger.info("Creating stack file %s" % filename)

    if utils.get_file_ext(filename) not in (".h5", ".hdf5"):
        raise ValueError("filename must end in .h5 or .hdf5")

    # TODO: do we want to replace the .unw files with .h5 files, then make a Virtual dataset?
    # layout = h5py.VirtualLayout(shape=(len(file_list), nrows, ncols), dtype=dtype)
    if not _check_dset(filename, STACK_DSET, overwrite):
        return

    file_list = sario.find_files(directory=directory, search_term="*" + file_ext)

    testf = sario.load(file_list[0])
    shape = (len(file_list), testf.shape[0], testf.shape[1])
    _create_dset(filename, STACK_DSET, shape, dtype=testf.dtype)
    with h5py.File(filename, "a") as hf:
        dset = hf[STACK_DSET]
        dset.attrs["filenames"] = file_list
        for idx, f in enumerate(file_list):
            dset[idx] = sario.load(f)

    if save_rsc:
        dem_rsc = sario.load(sario.find_rsc_file(directory=directory))
        save_dem_to_h5(filename, dem_rsc, dset_name=DEM_RSC_DSET, overwrite=overwrite)

    if create_mean:
        with h5py.File(filename, "a") as hf:
            hf.create_dataset(
                STACK_MEAN_DSET,
                data=np.mean(hf[STACK_DSET], axis=0),
            )

    return filename


# TODO: Process the correlation, mask very bad corr pixels in the igrams


def _check_dset(h5file, dset_name, overwrite):
    """Returns false if the dataset exists and overwrite is False

    If overwrite is set to true, will delete the dataset to make
    sure a new one can be created
    """
    with h5py.File(h5file, "a") as f:
        if dset_name in f:
            logger.info("{dset} already exists in {file},".format(dset=dset_name, file=h5file))
            if overwrite:
                logger.info("Overwrite true: Deleting.")
                del f[dset_name]
            else:
                logger.info("Skipping.")
                return False

        return True


def _create_dset(h5file, dset_name, shape, dtype=bool):
    with h5py.File(h5file, "a") as f:
        f.create_dataset(dset_name, shape=shape, dtype=dtype)


def _find_file_shape(dem_rsc=None, file_list=None, row_looks=None, col_looks=None):
    if not dem_rsc:
        try:
            g = sario.load(file_list[0], looks=(row_looks, col_looks))
        except IndexError:
            raise ValueError("No .geo files found in s")
        except TypeError:
            raise ValueError("Need file_list if no dem_rsc")

        return (len(file_list), g.shape[0], g.shape[1])
    else:
        return (len(file_list), dem_rsc["file_length"], dem_rsc["width"])


def load_dem_from_h5(h5file=None, dset="dem_rsc"):
    with h5py.File(h5file, "r") as f:
        return json.loads(f[dset][()])


def save_dem_to_h5(h5file, dem_rsc, dset_name="dem_rsc", overwrite=True):
    if not _check_dset(h5file, dset_name, overwrite):
        return

    with h5py.File(h5file, "a") as f:
        f[dset_name] = json.dumps(dem_rsc)


def shift_unw_file(unw_stack_file, ref_row, ref_col, window, overwrite=False):
    """Runs a reference point shift on flattened stack of unw files stored in .h5"""
    logger.info("Starting shift_stack: using %s, %s as ref_row, ref_col", ref_row, ref_col)
    if not _check_dset(unw_stack_file, STACK_FLAT_SHIFTED_DSET, overwrite):
        return

    with h5py.File(unw_stack_file, "a") as f:
        if STACK_FLAT_DSET not in f:
            raise ValueError("Need %s to be created in %s before"
                             " shift stack can be run" % (STACK_FLAT_DSET, unw_stack_file))

        stack_in = f[STACK_FLAT_DSET]
        f.create_dataset(
            STACK_FLAT_SHIFTED_DSET,
            shape=f[STACK_FLAT_DSET].shape,
            dtype=f[STACK_FLAT_DSET].dtype,
        )
        stack_out = f[STACK_FLAT_SHIFTED_DSET]
        shift_stack(stack_in, stack_out, ref_row, ref_col, window=window)
        f[STACK_FLAT_SHIFTED_DSET].attrs["reference"] = (ref_row, ref_col)

    logger.info("Shifting stack complete")


def shift_stack(stack_in, stack_out, ref_row, ref_col, window=3):
    """Subtracts reference pixel group from each layer

    Args:
        stack_in (ndarray-like): 3D array of images, stacked along axis=0
        stack_out (ndarray-like): empty 3D array, will hold output
            Both can be hdf5 datasets
        ref_row (int): row index of the reference pixel to subtract
        ref_col (int): col index of the reference pixel to subtract
        window (int): size of the group around ref pixel to avg for reference.
            if window=1 or None, only the single pixel used to shift the group.

    Raises:
        ValueError: if window is not a positive int, or if ref pixel out of bounds
    """
    win = window // 2
    for idx, layer in enumerate(stack_in):
        patch = layer[ref_row - win:ref_row + win + 1, ref_col - win:ref_col + win + 1]
        stack_out[idx] = layer - np.mean(patch)


def load_reference(unw_stack_file=UNW_FILENAME):
    with h5py.File(unw_stack_file, "r") as f:
        try:
            return f[STACK_FLAT_SHIFTED_DSET].attrs["reference"]
        except KeyError:
            return None, None


def save_deformation(igram_path,
                     deformation,
                     geo_date_list,
                     defo_file_name='deformation.h5',
                     dset_name="stack",
                     geolist_file_name='geolist.npy'):
    """Saves deformation ndarray and geolist dates as .npy file"""
    if utils.get_file_ext(defo_file_name) in (".h5", ".hdf5"):
        with h5py.File(defo_file_name, "a") as f:
            f[dset_name] = deformation
        store_geolist(igram_path, defo_file_name, overwrite=True)
    elif defo_file_name.endswith(".npy"):
        np.save(os.path.join(igram_path, defo_file_name), deformation)
        np.save(os.path.join(igram_path, geolist_file_name), geo_date_list)


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


@log_runtime
def deramp_stack(
        unw_stack_file=UNW_FILENAME,
        order=1,
        overwrite=False,
):
    """Handles removing linear ramps for all files in a stack

    Saves the files to a new dataset in the same unw stack .h5 file

    Args:
        unw_stack_file (str): Filename for the .h5 stack of .unw
            These layers will be deramped and saved do a new dset
        order (int): order of polynomial surface to use to deramp
            1 is linear (default), 2 is quadratic
    """
    logger.info("Removing any ramp from each stack layer")
    # Get file names to save results/ check if we deramped already

    # Make sure the normal .unw stack file has been created
    with h5py.File(unw_stack_file, "r") as f:
        if STACK_DSET not in f:
            raise ValueError("unw stack dataset doesn't exist at %s" % unw_stack_file)

    if not _check_dset(unw_stack_file, STACK_FLAT_DSET, overwrite):
        return

    with h5py.File(MASK_FILENAME) as fmask:
        mask_dset = fmask[IGRAM_MASK_DSET]
        with h5py.File(unw_stack_file, "a") as f:
            logger.info("Creating dataset %s in %s" % (STACK_FLAT_DSET, unw_stack_file))

            f.create_dataset(
                STACK_FLAT_DSET,
                shape=f[STACK_DSET].shape,
                dtype=f[STACK_DSET].dtype,
            )
            # Shape of sario.load_stack with return_amp is (nlayers, 2, nrows, ncols)
            for idx, layer in enumerate(f[STACK_DSET]):
                mask = mask_dset[idx]
                try:
                    f[STACK_FLAT_DSET][idx] = remove_ramp(layer, order=order, mask=mask)
                except np.linalg.linalg.LinAlgError:
                    logger.info("Failed to estimate ramp on layer %s: setting to 0" % idx)
                    f[STACK_FLAT_DSET][idx] = np.zeros_like(layer)


def remove_ramp(z, order=1, mask=np.ma.nomask):
    """Estimates a linear plane through data and subtracts to flatten

    Used to remove noise artifacts from unwrapped interferograms

    Args:
        z (ndarray): 2D array, interpreted as heights
        order (int): degree of surface estimation
            order = 1 removes linear ramp, order = 2 fits quadratic surface

    Returns:
        ndarray: flattened 2D array with estimated surface removed
    """
    # Make a version of the image with nans in masked places
    z_masked = z.copy()
    z_masked[mask] = np.nan
    # Use this constrained version to find the plane fit
    z_fit = estimate_ramp(z_masked, order)
    # Then use the non-masked as return value
    return z - z_fit


def estimate_ramp(z, order):
    """Takes a 2D array an fits a linear plane to the data

    Ignores pixels that have nan values

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
    zflat = z.flatten()
    good_idxs = ~np.isnan(zflat)
    if order == 1:
        A = np.c_[np.ones(xidxs.shape), xidxs, yidxs]
        coeffs, _, _, _ = np.linalg.lstsq(A[good_idxs], zflat[good_idxs], rcond=None)
        # coeffs will be a, b, c in the equation z = ax + by + c
        c, a, b = coeffs
        # We want full blocks, as opposed to matrix_index flattened
        y_block, x_block = matrix_indices(z.shape, flatten=False)
        z_fit = (a * x_block + b * y_block + c)

    elif order == 2:
        A = np.c_[np.ones(xidxs.shape), xidxs, yidxs, xidxs * yidxs, xidxs**2, yidxs**2]
        # coeffs will be 6 elements for the quadratic
        coeffs, _, _, _ = np.linalg.lstsq(A[good_idxs], zflat[good_idxs], rcond=None)
        yy, xx = matrix_indices(z.shape, flatten=True)
        idx_matrix = np.c_[np.ones(xx.shape), xx, yy, xx * yy, xx**2, yy**2]
        z_fit = np.dot(idx_matrix, coeffs).reshape(z.shape)

    return z_fit


def find_reference_location(
        unw_stack_file=UNW_FILENAME,
        mask_stack_file=MASK_FILENAME,
        cc_stack_file=CC_FILENAME,
        gps_dir=None,
):
    """Find reference pixel on based on GPS availability and mean correlation
    """
    dem_rsc = load_dem_from_h5(h5file=unw_stack_file, dset="dem_rsc")

    # Make a latlon image to check for gps data containment
    with h5py.File(unw_stack_file, "r") as f:
        latlon_image = latlon.LatlonImage(data=f[STACK_DSET][0], dem_rsc=dem_rsc)

    ref_row, ref_col = None, None
    logger.info("Searching for gps station within area")
    # Don't make the invalid GPS here in case the random image chosed above is bad:
    # We'll use the mask ll image to decide which pixels are bad
    stations = apertools.gps.stations_within_image(latlon_image,
                                                   mask_invalid=False,
                                                   gps_dir=gps_dir)
    # Make a latlon image From the total masks
    with h5py.File(mask_stack_file, "r") as f:
        mask_ll_image = latlon.LatlonImage(data=f[IGRAM_MASK_SUM_DSET], dem_rsc=dem_rsc)

    with h5py.File(cc_stack_file, "r") as f:
        mean_cor = f[STACK_MEAN_DSET][:]
        mean_cor_ll_image = latlon.LatlonImage(data=mean_cor, dem_rsc=dem_rsc)

    if len(stations) > 0:
        logger.info("Station options:")
        logger.info(stations)
        num_masks = [mask_ll_image[lat, lon] for _, lon, lat in stations]
        pixel_correlations = [mean_cor_ll_image[lat, lon] for _, lon, lat in stations]

        logger.info("Sorting by fewer masked dates and highest correlation")
        # Note: make cor negative to sort large numbers to the front
        sorted_stations = sorted(
            zip(num_masks, pixel_correlations, stations),
            key=lambda tup: (tup[0], -tup[1]),
        )
        logger.info(sorted_stations)

        name, lon, lat = sorted_stations[0][-1]
        logger.info("Using station %s at (lon, lat) (%s, %s)", name, lon, lat)
        ref_row, ref_col = latlon_image.nearest_pixel(lon=lon, lat=lat)

    if ref_row is None:
        logger.warning("GPS station search failed, reverting to coherence")
        logger.info("Finding most coherent patch in stack.")
        ref_row, ref_col = find_coherent_patch(mean_cor)

    logger.info("Using %s as .unw reference point", (ref_row, ref_col))
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


def store_geolist(igram_path=None, stack_file=None, overwrite=False, geo_date_list=None):
    if geo_date_list is None:
        geo_date_list, _ = load_geolist_intlist(igram_path, parse=True)

    if not _check_dset(stack_file, GEOLIST_DSET, overwrite):
        return

    logger.debug("Saving geo dates to %s / %s" % (stack_file, GEOLIST_DSET))
    with h5py.File(stack_file, "a") as f:
        # JSON gets messed from doing from julia to h5py for now
        # f[GEOLIST_DSET] = json.dumps(_geolist_to_str(geo_date_list))
        f[GEOLIST_DSET] = _geolist_to_str(geo_date_list)


def store_intlist(igram_path=None, stack_file=None, overwrite=False, int_date_list=None):
    if int_date_list is None:
        _, int_date_list = load_geolist_intlist(igram_path)

    if not _check_dset(stack_file, INTLIST_DSET, overwrite):
        return

    logger.info("Saving igram dates to %s / %s" % (stack_file, INTLIST_DSET))
    with h5py.File(stack_file, "a") as f:
        f[INTLIST_DSET] = _intlist_to_str(int_date_list)


def load_geolist_intlist(directory, geolist_ignore_file=None, parse=True):
    """Load the geo_date_list and int_date_list from a directory with igrams

    Assumes that the .geo files are one diretory up from the igrams
    """
    int_date_list = sario.find_igrams(directory, parse=parse)
    geo_date_list = sario.find_geos(utils.get_parent_dir(directory), parse=parse)

    if geolist_ignore_file is not None:
        ignore_filepath = os.path.join(directory, geolist_ignore_file)
        geo_date_list, int_date_list = ignore_geo_dates(geo_date_list,
                                                        int_date_list,
                                                        ignore_file=ignore_filepath,
                                                        parse=parse)
    return geo_date_list, int_date_list


def ignore_geo_dates(geo_date_list, int_date_list, ignore_file="geolist_missing.txt", parse=True):
    """Read extra file to ignore certain dates of interferograms"""
    ignore_geos = set(sario.find_geos(ignore_file, parse=parse))
    logger.info("Ignoreing the following .geo dates:")
    logger.info(sorted(ignore_geos))
    valid_geos = [g for g in geo_date_list if g not in ignore_geos]
    valid_igrams = [i for i in int_date_list if i[0] not in ignore_geos and i[1] not in ignore_geos]
    return valid_geos, valid_igrams


def _geolist_to_str(geo_date_list):
    return np.array([d.strftime(DATE_FMT) for d in geo_date_list]).astype("S")


def _intlist_to_str(int_date_list):
    return np.array([(a.strftime(DATE_FMT), b.strftime(DATE_FMT))
                     for a, b in int_date_list]).astype("S")


def load_composite_mask(geo_date_list=None,
                        perform_mask=True,
                        deformation_filename=None,
                        mask_filename=MASK_FILENAME,
                        directory=None):
    if not perform_mask:
        return np.ma.nomask

    if directory is not None:
        mask_filename = os.path.join(directory, mask_filename)

    # If they pass a deformation .h5 stack, get only the dates actually used
    # instead of all possible dates stored in the mask stack
    if deformation_filename is not None:
        if directory is not None:
            deformation_filename = os.path.join(directory, deformation_filename)
            geo_date_list = sario.load_geolist_from_h5(deformation_filename)

    # Get the indices of the mask layers that were used in the deformation stack
    all_geo_dates = apertools.sario.load_geolist_from_h5(mask_filename)
    if geo_date_list is None:
        used_bool_arr = np.full(len(all_geo_dates), True)
    else:
        used_bool_arr = np.array([g in geo_date_list for g in all_geo_dates])

    with h5py.File(mask_filename) as f:
        # Maks a single mask image for any pixel that has a mask
        # Note: not using GEO_MASK_SUM_DSET since we may be sub selecting layers
        stack_mask = np.sum(f[GEO_MASK_DSET][used_bool_arr, :, :], axis=0)
        stack_mask = stack_mask > 0
        return stack_mask


def load_single_mask(int_date_string=None,
                     date_pair=None,
                     mask_filename=MASK_FILENAME,
                     int_date_list=None):
    """Load one mask from the `mask_filename`

    Can either pass a tuple of Datetimes in date_pair, or a string like
    `20170101_20170104.int` or `20170101_20170303` to int_date_string
    """
    if int_date_list is None:
        int_date_list = sario.load_intlist_from_h5(mask_filename)

    if int_date_string is not None:
        # If the pass string with ., only take first part
        date_str_pair = int_date_string.split('.')[0].split('_')
        date_pair = sario.parse_intlist_strings([date_str_pair])[0]

    with h5py.File(mask_filename, "r") as f:
        idx = int_date_list.index(date_pair)
        return f[IGRAM_MASK_DSET][idx]


@log_runtime
def zero_masked_areas(igram_path=".", mask_filename=None, verbose=True):
    logger.info("Zeroing out masked area in .cc and .int files")

    if mask_filename is None:
        mask_filename = os.path.join(igram_path, MASK_FILENAME)

    int_date_list = sario.load_intlist_from_h5(mask_filename)

    with h5py.File(mask_filename, "r") as f:
        igram_mask_dset = f[IGRAM_MASK_DSET]
        for idx, (early, late) in enumerate(int_date_list):
            cur_mask = igram_mask_dset[idx]
            base_str = "%s_%s" % (early.strftime(DATE_FMT), late.strftime(DATE_FMT))

            if verbose:
                logger.info("Zeroing {0}.cc and {0}.int".format(base_str))

            int_filename = base_str + ".int"
            zero_file(int_filename, cur_mask, is_stacked=False)

            cc_filename = base_str + ".cc"
            zero_file(cc_filename, cur_mask, is_stacked=True)


def zero_file(filename, mask, is_stacked=False):
    if is_stacked:
        amp, img = sario.load(filename, return_amp=True)
        img[mask] = 0
        sario.save(filename, np.stack((amp, img), axis=0))
    else:
        img = sario.load(filename)
        img[mask] = 0
        sario.save(filename, img)


@log_runtime
def merge_files(filename1, filename2, new_filename, overwrite=False):
    """Merge together 2 (currently mask) hdf5 files into a new file"""

    def _merge_lists(list1, list2, merged_list, dset_name, dset1, dset2):
        logger.info("%s: %s from %s and %s from %s into %s in file %s" % (
            dset_name,
            len(list1),
            filename1,
            len(list2),
            filename2,
            len(merged_list),
            new_filename,
        ))
        for idx in range(len(merged_list)):
            cur_item = merged_list[idx]
            if cur_item in list1:
                jdx = list1.index(cur_item)
                fnew[dset_name][idx] = dset1[jdx]
            else:
                jdx = list2.index(cur_item)
                fnew[dset_name][idx] = dset2[jdx]

    if overwrite:
        _check_dset(new_filename, IGRAM_MASK_DSET, overwrite)
        _check_dset(new_filename, GEO_MASK_DSET, overwrite)

    f1 = h5py.File(filename1)
    f2 = h5py.File(filename2)
    igram_dset1 = f1[IGRAM_MASK_DSET]
    igram_dset2 = f2[IGRAM_MASK_DSET]
    geo_dset1 = f1[GEO_MASK_DSET]
    geo_dset2 = f2[GEO_MASK_DSET]

    intlist1 = sario.load_intlist_from_h5(filename1)
    intlist2 = sario.load_intlist_from_h5(filename2)
    geolist1 = sario.load_geolist_from_h5(filename1)
    geolist2 = sario.load_geolist_from_h5(filename2)
    merged_intlist = sorted(set(intlist1) | set(intlist2))
    merged_geolist = sorted(set(geolist1) | set(geolist2))

    store_intlist(stack_file=new_filename, overwrite=True, int_date_list=merged_intlist)
    store_geolist(stack_file=new_filename, overwrite=True, geo_date_list=merged_geolist)

    new_geo_shape = (len(merged_geolist), geo_dset1.shape[1], geo_dset1.shape[2])
    _create_dset(new_filename, GEO_MASK_DSET, new_geo_shape, dtype=igram_dset1.dtype)
    new_igram_shape = (len(merged_intlist), igram_dset1.shape[1], igram_dset1.shape[2])
    _create_dset(new_filename, IGRAM_MASK_DSET, new_igram_shape, dtype=igram_dset1.dtype)

    fnew = h5py.File(new_filename, "a")
    try:
        _merge_lists(geolist1, geolist2, merged_geolist, GEO_MASK_DSET, geo_dset1, geo_dset2)
        _merge_lists(intlist1, intlist2, merged_intlist, IGRAM_MASK_DSET, igram_dset1, igram_dset2)

    finally:
        f1.close()
        f2.close()
        fnew.close()
