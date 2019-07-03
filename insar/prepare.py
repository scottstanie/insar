"""prepare.py

Preprocessing insar data for timeseries analysis

Forms stacks as .h5 files for easy access to depth-wise slices
"""
import h5py
import json
import os
import datetime
import numpy as np
from scipy.ndimage.filters import uniform_filter

from sardem.loading import load_dem_rsc
from apertools import sario, utils, latlon
import apertools.gps
from apertools.log import get_log

logger = get_log()
DATE_FMT = "%Y%m%d"

MASK_FILENAME = "masks.h5"
GEO_MASK_DSET = "geo"
IGRAM_MASK_DSET = "igram"
DEM_RSC_DSET = "dem_rsc"


def prepare_stacks(igram_path, overwrite=False, geolist_ignore_file="geolist_missing.txt"):
    create_igram_stacks(igram_path, overwrite=overwrite)
    create_mask_stacks(igram_path, overwrite=overwrite)
    # geo_date_list, int_date_list = load_geolist_intlist(igram_path,
    # geolist_ignore_file=geolist_ignore_file,
    # parse=True)


def create_igram_stacks(igram_path, overwrite=False):
    stack_dicts = (
        dict(file_ext=".unw", create_mean=False, fname="unw_stack.h5"),
        dict(file_ext=".cc", create_mean=True, fname="cc_stack.h5"),
    )
    for d in stack_dicts:
        filename = os.path.join(igram_path, d["fname"])
        logger.info("Creating hdf5 stack %s" % filename)
        d["filename"] = filename
        create_hdf5_stack(directory=igram_path, overwrite=overwrite, **d)


def create_mask_stacks(igram_path, geo_path=None, overwrite=False):
    """Create mask stacks for areas in .geo and .int

    Uses .geo dead areas as well as correlation
    """
    create_geos = True
    create_igram = True
    mask_file = os.path.join(igram_path, MASK_FILENAME)
    # Check if either dataset exists
    if os.path.exists(mask_file):
        with h5py.File(mask_file) as f:
            if GEO_MASK_DSET in f:
                create_geos = False
                print("geo mask dataset exists in %s" % mask_file)
            if IGRAM_MASK_DSET in f:
                print("igram mask dataset exists in %s" % mask_file)
                create_igram = False

    if geo_path is None:
        geo_path = utils.get_parent_dir(igram_path)

    # Used to shrink the .geo masks to save size as .int masks
    row_looks, col_looks = apertools.utils.find_looks_taken(igram_path, geo_path=geo_path)

    dem_rsc = sario.load(sario.find_rsc_file(directory=igram_path))
    _save_dem_to_h5(mask_file, dem_rsc, dset_name=DEM_RSC_DSET, overwrite=overwrite)

    if create_geos:
        save_geo_masks(
            geo_path,
            mask_file,
            dem_rsc=dem_rsc,
            row_looks=row_looks,
            col_looks=col_looks,
            overwrite=overwrite,
        )

    if not create_igram:
        return mask_file

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
    _create_dset(mask_file, dset_name, overwrite, shape=shape, dtype=bool)

    with h5py.File(mask_file) as f:
        dset = f[dset_name]
        for idx, geo_fname in enumerate(geo_file_list):
            g = sario.load(geo_fname, looks=(row_looks, col_looks))
            # ipdb.set_trace()
            print('Saving %s to stack' % geo_fname)
            dset[idx] = _get_geo_mask(g)


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
    _create_dset(mask_file, dset_name, overwrite, shape=shape, dtype=bool)

    with h5py.File(mask_file) as f:
        geo_mask_stack = f[GEO_MASK_DSET]
        int_mask_dset = f[dset_name]
        for idx, (early, late) in enumerate(int_date_list):
            early_idx = geo_date_list.index(early)
            late_idx = geo_date_list.index(late)
            early_mask = geo_mask_stack[early_idx]
            late_mask = geo_mask_stack[late_idx]

            int_mask_dset[idx] = np.logical_or(early_mask, late_mask)


def create_hdf5_stack(filename=None,
                      directory=None,
                      file_ext=None,
                      create_mean=True,
                      save_rsc=True,
                      overwrite=False,
                      **kwargs):
    """Make stack as hdf5 file from a list of files

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
    file_list = sario.find_files(directory=directory, search_term="*" + file_ext)

    stack_dset = "stack"
    stack_mean_dset = "mean_stack"
    if os.path.exists(filename):
        print("{} already exists".format(filename))
        if overwrite:
            print("Overwriting file")
        else:
            print("Skipping")
            return

    with h5py.File(filename, "w") as hf:
        stack = sario.load_stack(file_list=file_list)
        hf.create_dataset(
            stack_dset,
            data=stack,
        )
        hf[stack_dset].attrs["filenames"] = file_list

    if save_rsc:
        dem_rsc = sario.load(sario.find_rsc_file(directory=directory))
        _save_dem_to_h5(filename, dem_rsc, dset_name=DEM_RSC_DSET, overwrite=overwrite)

    if create_mean:
        with h5py.File(filename, "a") as hf:
            hf.create_dataset(
                stack_mean_dset,
                data=np.mean(stack, axis=0),
            )

    return filename


# TODO: Process the correlation, mask very bad corr pixels in the igrams
def _create_dset(h5file, dset_name, overwrite, shape, dtype=bool):
    with h5py.File(h5file, "a") as f:
        if dset_name in f:
            if overwrite:
                del f[dset_name]
            else:
                print("{dset} already exists in {file}: skipping".format(dset=dset_name,
                                                                         file=h5file))
                return

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


def _load_dem_from_h5(h5file, dset="dem_rsc"):
    with h5py.File(h5file, "a") as f:
        return json.loads(f['dem_rsc'][()])


def _save_dem_to_h5(h5file, dem_rsc, dset_name="dem_rsc", overwrite=True):
    with h5py.File(h5file, "a") as f:
        if dset_name in f:
            if overwrite:
                del f[dset_name]
            else:
                print("{dset} already exists in {file}: skipping".format(dset=dset_name,
                                                                         file=h5file))
                return
        f[dset_name] = json.dumps(dem_rsc)


def _geolist_to_str(geo_date_list):
    return [d.strftime(DATE_FMT) for d in geo_date_list]


def _intlist_to_str(int_date_list):
    return [(a.strftime(DATE_FMT), b.strftime(DATE_FMT)) for a, b in int_date_list]


def pick_reference(igram_path):
    # find ref on based on GPS availability and mean correlation
    # Make a latlon image to check for gps data containment
    # TODO: maybe i need to search for masks? dont wanna pick a garbage one by accident
    with h5py.File(unw_stack_file) as f:
        latlon_image = latlon.LatlonImage(
            data=unw_stack[0],
            # TODO: attach dem rsc to .hdf5
            dem_rsc_file=os.path.join(igram_path, 'dem.rsc'))
    ref_row, ref_col = find_reference_location(latlon_image, igram_path, mask_stack, gps_dir=None)


def load_geolist_intlist(directory, geolist_ignore_file=None, parse=True):
    """Load the geo_date_list and int_date_list from a directory with igrams"""
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


def find_time_diffs(geo_date_list):
    """Finds the number of days between successive .geo files

    Args:
        geo_date_list (list[date]): dates of the .geo SAR acquisitions

    Returns:
        np.array: days between each datetime in geo_date_list
            dtype=int, length is a len(geo_date_list) - 1"""
    return np.array([difference.days for difference in np.diff(geo_date_list)])


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


def pick_reference():
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


def shift(ref_row, ref_col):
    logger.info("Starting shift_stack: using %s, %s as ref_row, ref_col", ref_row, ref_col)
    # unw_stack = shift_stack(unw_stack, ref_row, ref_col, window=window)
    logger.info("Shifting stack complete")


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
            f["geo_date_list"] = _geolist_to_str(geo_date_list)
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


def load_unw_masked_stack(igram_path,
                          num_timediffs=None,
                          unw_ext='.unw',
                          deramp=True,
                          deramp_order=1,
                          masking=True):

    int_file_list = sario.find_igrams(igram_path, parse=False)
    if deramp:
        # Deramp each .unw file
        # TODO: do I ever really want 2nd order?
        # # For larger areas, use quadratic ramp. Otherwise, linear
        # max_linear = 20  # km
        # order = 1 if (width < max_linear or height < max_linear) else 2

        width, height = latlon.grid_size(**load_dem_rsc(os.path.join(igram_path, 'dem.rsc')))
        logger.info("Dem size %.2f by %.2f km: using order %s surface to deramp", width, height,
                    deramp_order)
        unw_stack = deramp_stack(int_file_list, unw_ext, order=deramp_order)
    else:
        unw_file_names = [f.replace('.int', unw_ext) for f in int_file_list]
        unw_stack = sario.load_stack(file_list=unw_file_names)

    unw_stack = unw_stack.view(np.ma.MaskedArray)

    if masking:
        int_mask_file_names = [n + '.mask.npy' for n in int_file_list]
        if not all(os.path.exists(f) for f in int_mask_file_names):
            logger.info("Creating and saving igram masks")
            row_looks, col_looks = utils.find_looks_taken(igram_path)
            create_igram_masks(igram_path, row_looks=row_looks, col_looks=col_looks)

        # Note: using .geo masks as well as .int masks since valid data in one
        # .geo produces non-zero igram, but it is still garbage

        # need to specify file_list or the order is different from find_igrams
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


def deramp_stack(unw_stack_file, order=1):
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
    stack_dset = "stack"
    deramp_dset = "stack_deramped"

    if not os.path.exists(unw_stack_file):
        with h5py.File(unw_stack_file, "r") as f:
            if stack_dset not in f:
                raise ValueError("unw stack file/ dataset doesn't exist at %s" % unw_stack_file)

    with h5py.File(unw_stack_file, "r") as f:
        if deramp_dset in f:
            logger.info("Deramped dataset exist.")
            return

    logger.info("Deramped dataset doesn't exist: Creating dataset %s in %s" %
                (deramp_dset, unw_stack_file))

    with h5py.File(unw_stack_file, "a") as f:
        nlayers, nrows, ncols = f[deramp_dset].shape
        out_stack = np.empty((nlayers, nrows, ncols), dtype=sario.FLOAT_32_LE)
        f[deramp_dset] = out_stack
        # Shape of sario.load_stack with return_amp is (nlayers, 2, nrows, ncols)
        for layer in f[stack_dset]:
            f[deramp_dset] = remove_ramp(layer, order=order)


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
