"""timeseries.py
Functions for performing time series analysis of unwrapped interferograms

files in the igrams folder:
    geolist, intlist, sbas_list
scott@lidar igrams]$ head geolist
../S1A_IW_SLC__1SDV_20180420T043026_20180420T043054_021546_025211_81BE.SAFE.geo
../S1A_IW_SLC__1SDV_20180502T043026_20180502T043054_021721_025793_5C18.SAFE.geo
[scott@lidar igrams]$ head sbas_list
../S1A_IW_SLC__1SDV_20180420T043026_20180420T043054_021546_025211_81BE.SAFE.geo \
        ../S1A_IW_SLC__1SDV_20180502T043026_20180502T043054_021721_025793_5C18.SAFE.geo 12.0   \
        -16.733327776024169
[scott@lidar igrams]$ head intlist
20180420_20180502.int

"""
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
from apertools.log import get_log, log_runtime

SENTINEL_WAVELENGTH = 5.5465763  # cm
PHASE_TO_CM = SENTINEL_WAVELENGTH / (-4 * np.pi)

logger = get_log()


@log_runtime
def run_inversion(igram_path,
                  reference=(None, None),
                  window=None,
                  constant_vel=False,
                  alpha=0,
                  difference=False,
                  deramp=True,
                  deramp_order=1,
                  masking=True,
                  geolist_ignore_file="geolist_missing.txt",
                  verbose=False):
    """Runs SBAS inversion on all unwrapped igrams

    Args:
        igram_path (str): path to the directory containing `intlist`,
            the .int filenames, the .unw files, and the dem.rsc file
        reference (tuple[int, int]): row and col index of the reference pixel to subtract
        window (int): size of the group around ref pixel to avg for reference.
            if window=1 or None, only the single pixel used to shift the group.
        constant_vel (bool): force solution to have constant velocity
            mutually exclusive with `alpha` option
        alpha (float): nonnegative Tikhonov regularization parameter.
            See https://en.wikipedia.org/wiki/Tikhonov_regularization
        difference (bool): for regularization, penalize differences in velocity
            Used to make a smoother final solution
        deramp (bool): Fits plane to each igram and subtracts (to remove orbital error)
        deramp_order (int): order of polynomial to use when removing phase
            from unwrapped igram
        geolist_ignore_file (str): text file with list of .geo files to ignore
            Removes the .geo and and igrams with these date
        masking (bool): flag to load stack of .int.mask files to mask invalid areas
        verbose (bool): print extra timing and debug info

    Returns:
        geolist (list[datetime]): dates of each SAR acquisition from read_geolist
        phi_arr (ndarray): absolute phases of every pixel at each time
        deformation (ndarray): matrix of deformations at each pixel and time
    """
    if verbose:
        logger.setLevel(10)  # DEBUG

    geolist, intlist = load_geolist_intlist(igram_path, parse=True)

    # Prepare B matrix and timediffs used for each pixel inversion
    B = build_B_matrix(geolist, intlist)
    timediffs = find_time_diffs(geolist)
    if B.shape[1] != len(timediffs):
        raise ValueError("Shapes of B {} and timediffs {} not compatible".format(
            B.shape, timediffs.shape))

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

    # Possible todo: process as blocks with view_as_blocks(stack, (num_stack, 4, 4))
    # from skimage.util.shape import view_as_blocks
    # Might need to save as separate blocks to get loading right

    dphi_columns = stack_to_cols(unw_stack)

    phi_arr_list = []
    max_bytes = 500e6
    num_patches = int(np.ceil(dphi_columns.nbytes / max_bytes)) + 1
    geo_mask_patches = np.array_split(geo_mask_columns, num_patches, axis=1)
    for idx, columns in enumerate(np.array_split(dphi_columns, num_patches, axis=1)):
        logger.info("Inverting patch %s out of %s" % (idx + 1, num_patches))
        geo_mask_patch = geo_mask_patches[idx]
        varr = invert_sbas(
            columns,
            B,
            geo_mask_columns=geo_mask_patch,
            constant_vel=constant_vel,
            alpha=alpha,
            difference=difference,
        )
        phi_arr_list.append(integrate_velocities(varr, timediffs))

    phi_arr = np.ma.hstack(phi_arr_list)

    # Multiple by wavelength ratio to go from phase to cm
    deformation = PHASE_TO_CM * phi_arr

    num_ints, rows, cols = unw_stack.shape
    # Now reshape all outputs that should be in stack form
    phi_arr = cols_to_stack(phi_arr, rows, cols)
    deformation = cols_to_stack(deformation, rows, cols).filled(np.NaN)
    return (geolist, phi_arr, deformation)


def _parse(datestr):
    return datetime.datetime.strptime(datestr, "%Y%m%d").date()


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
        geolist, intlist = ignore_geo_dates(geolist, intlist, ignore_file=ignore_filepath)
    return geolist, intlist


def ignore_geo_dates(geolist, intlist, ignore_file="geolist_missing.txt"):
    """Read extra file to ignore certain dates of interferograms"""
    ignore_geos = set(read_geolist(ignore_file))
    logger.info("Ignoreing the following .geo dates:")
    logger.info(sorted(ignore_geos))
    valid_geos = [g for g in geolist if g not in ignore_geos]
    valid_igrams = [i for i in intlist if i[0] not in ignore_geos and i[1] not in ignore_geos]
    return valid_geos, valid_igrams


def build_A_matrix(geolist, intlist):
    """Takes the list of igram dates and builds the SBAS A matrix

    Args:
        geolist (list[date]): datetimes of the .geo acquisitions
        intlist (list[tuple(date, date)])

    Returns:
        np.array 2D: the incident-like matrix from the SBAS paper: A*phi = dphi
            Each row corresponds to an igram, each column to a .geo
            value will be -1 on the early (slave) igrams, +1 on later (master)
    """
    # We take the first .geo to be time 0, leave out of matrix
    # Match on date (not time) to find indices
    geolist = geolist[1:]
    M = len(intlist)  # Number of igrams, number of rows
    N = len(geolist)
    A = np.zeros((M, N))
    for j in range(M):
        early_igram, late_igram = intlist[j]

        try:
            idx_early = geolist.index(early_igram)
            A[j, idx_early] = -1
        except ValueError:  # The first SLC will not be in the matrix
            pass

        idx_late = geolist.index(late_igram)
        A[j, idx_late] = 1

    return A


def find_time_diffs(geolist):
    """Finds the number of days between successive .geo files

    Args:
        geolist (list[date]): dates of the .geo SAR acquisitions

    Returns:
        np.array: days between each datetime in geolist
            dtype=int, length is a len(geolist) - 1"""
    return np.array([difference.days for difference in np.diff(geolist)])


def build_B_matrix(geolist, intlist):
    """Takes the list of igram dates and builds the SBAS B (velocity coeff) matrix

    Args:
        geolist (list[date]): dates of the .geo SAR acquisitions
        intlist (list[tuple(date, date)])

    Returns:
        np.array: 2D array of the velocity coefficient matrix from the SBAS paper:
                Bv = dphi
            Each row corresponds to an igram, each column to a .geo
            value will be t_k+1 - t_k for columns after the -1 in A,
            up to and including the +1 entry
    """
    # TODO: get rid of A matrix building first
    timediffs = find_time_diffs(geolist)

    A = build_A_matrix(geolist, intlist)
    B = np.zeros_like(A)

    for j, row in enumerate(A):
        # if no -1 entry, start at index 0. Otherwise, add 1 to exclude the -1 index
        start_idx = list(row).index(-1) + 1 if (-1 in row) else 0
        # End index is inclusive of the +1
        end_idx = np.where(row == 1)[0][0] + 1  # +1 will always exist in row

        # Now only fill in the time diffs in the range from the early igram index
        # to the later igram index
        B[j][start_idx:end_idx] = timediffs[start_idx:end_idx]

    return B


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


def _create_diff_matrix(n, order=1):
    """Creates n x n matrix subtracting adjacent vector elements

    Example:
        >>> print(_create_diff_matrix(4, order=1))
        [[ 1 -1  0  0]
         [ 0  1 -1  0]
         [ 0  0  1 -1]]

        >>> print(_create_diff_matrix(4, order=2))
        [[ 1 -1  0  0]
         [-1  2 -1  0]
         [ 0 -1  2 -1]
         [ 0  0 -1  1]]

    """
    if order == 1:
        diff_matrix = -1 * np.diag(np.ones(n - 1), k=1).astype('int')
        np.fill_diagonal(diff_matrix, 1)
        diff_matrix = diff_matrix[:-1, :]
    elif order == 2:
        diff_matrix = -1 * np.diag(np.ones(n - 1), k=1).astype('int')
        diff_matrix = diff_matrix + -1 * np.diag(np.ones(n - 1), k=-1).astype('int')
        np.fill_diagonal(diff_matrix, 2)
        diff_matrix[-1, -1] = 1
        diff_matrix[0, 0] = 1

    return diff_matrix


def invert_sbas(delta_phis, B, geo_mask_columns=None, constant_vel=False, alpha=0,
                difference=False):
    """Performs and SBAS inversion on each pixel of unw_stack to find deformation

    Solves the least squares equation Bv = dphi

    Args:
        delta_phis (ndarray): columns of unwrapped phases (delta phis)
            Each col is 1 pixel of load_stack along 3rd axis
        B (ndarray): output of build_B_matrix for current set of igrams
        geo_mask_columns (ndarray[bool]): .geo file masks, reshaped to columns
        constant_vel (bool): force solution to have constant velocity
            mutually exclusive with `alpha` option
        alpha (float): nonnegative Tikhonov regularization parameter.
            If alpha > 0, then the equation is instead to minimize
            ||B*v - dphi||^2 + ||alpha*I*v||^2
            See https://en.wikipedia.org/wiki/Tikhonov_regularization
        difference (bool): for regularization, penalize differences in velocity
            Used to make a smoother final solution

    Returns:
        ndarray: solution velocity arrary
    """

    def _augment_matrices(B, delta_phis, alpha):
        reg_matrix = _create_diff_matrix(B.shape[1]) if difference else np.eye(B.shape[1])
        B = np.vstack((B, alpha * reg_matrix))
        # Now make num rows match
        zeros_shape = (B.shape[0] - delta_phis.shape[0], delta_phis.shape[1])
        delta_phis = np.vstack((delta_phis, np.zeros(zeros_shape)))
        return B, delta_phis

    if B.shape[0] != delta_phis.shape[0]:
        raise ValueError("Shapes of B {} and delta_phis {} not compatible".format(
            B.shape, delta_phis.shape))
    elif alpha < 0:
        raise ValueError("alpha cannot be negative")

    # Adjustments to solution:
    # Force velocity constant across time
    if constant_vel is True:
        logger.info("Using a constant velocity for inversion solutions.")
        B = np.expand_dims(np.sum(B, axis=1), axis=1)
    # Add regularization to the solution
    elif alpha > 0:
        logger.info("Using regularization with alpha=%s, difference=%s", alpha, difference)
        # Augment only if regularization requested
        B, delta_phis = _augment_matrices(B, delta_phis, alpha)

    # Velocity will be result of the inversion
    # velocity_array, _, rank_B, sing_vals_B = np.linalg.lstsq(B, delta_phis, rcond=None)
    velocity_array = mask.masked_lstsq(B, delta_phis, geo_mask_columns)

    # velocity array entries: v_j = (phi_j - phi_j-1)/(t_j - t_j-1)
    if velocity_array.ndim == 1:
        velocity_array = np.expand_dims(velocity_array, axis=-1)

    return velocity_array


def integrate_velocities(velocity_array, timediffs):
    """Takes SBAS velocity output and finds phases

    Args:
        velocity_array (ndarray): output of invert_sbas, velocities at
            each point in time
        timediffs (np.array): dtype=int, days between each SAR acquisitions
            length will be 1 less than num SAR acquisitions

    Returns:
        ndarray: integrated phase array

    """
    # multiply each column of vel array: each col is a separate solution
    phi_diffs = timediffs.reshape((-1, 1)) * velocity_array

    # Now the final phase results are the cumulative sum of delta phis
    phi_arr = np.ma.cumsum(phi_diffs, axis=0)
    # Add 0 as first entry of phase array to match geolist length on each col
    phi_arr = np.ma.vstack((np.zeros(phi_arr.shape[1]), phi_arr))

    return phi_arr


def stack_to_cols(stacked):
    """Takes a 3D array, makes vectors along the 3D axes into cols

    The reverse function of cols_to_stack

    Args:
        stacked (ndarray): 3D array, each [idx, :, :] is an array of interest

    Returns:
        ndarray: a 2D array where each of the stacked[:, i, j] is
            now a column

    Raises:
        ValueError: if input shape is not 3D

    Example:
        >>> a = np.arange(18).reshape((2, 3, 3))
        >>> cols = stack_to_cols(a)
        >>> print(cols)
        [[ 0  1  2  3  4  5  6  7  8]
         [ 9 10 11 12 13 14 15 16 17]]
    """
    if len(stacked.shape) != 3:
        raise ValueError("Must be a 3D ndarray")

    num_stacks = stacked.shape[0]
    return stacked.reshape((num_stacks, -1))


def cols_to_stack(columns, rows, cols):
    """Takes a 2D array of columns, reshapes to cols along 3rd axis

    The reverse function of stack_to_cols

    Args:
        stacked (ndarray): 2D array of columns of data
        rows (int): number of rows of original stack
        cols (int): number of rows of original stack

    Returns:
        ndarray: a 2D array where each output[idx, :, :] was column idx

    Raises:
        ValueError: if input shape is not 2D

    Example:
        >>> a = np.arange(18).reshape((2, 3, 3))
        >>> cols = stack_to_cols(a)
        >>> print(cols)
        [[ 0  1  2  3  4  5  6  7  8]
         [ 9 10 11 12 13 14 15 16 17]]
        >>> print(np.all(cols_to_stack(cols, 3, 3) == a))
        True
    """
    if len(columns.shape) != 2:
        raise ValueError("Must be a 2D ndarray")

    return columns.reshape((-1, rows, cols))


def save_deformation(igram_path,
                     deformation,
                     geolist,
                     defo_name='deformation.npy',
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
        logger.info("Reading geoload masks into columns")
        geo_file_names = read_geolist(filepath=igram_path, parse=False)
        geo_mask_file_names = [n + '.mask.npy' for n in geo_file_names]
        geo_masks = np.ma.array(sario.load_stack(file_list=geo_mask_file_names))
        geo_mask_columns = stack_to_cols(geo_masks)
        del geo_masks

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
