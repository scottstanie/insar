"""Functions for performing time series analysis of unwrapped interferograms

files in the igrams folder:
    geolist, intlist, sbas_list
scott@lidar igrams]$ head geolist
../S1A_IW_SLC__1SDV_20180420T043026_20180420T043054_021546_025211_81BE.SAFE.geo
../S1A_IW_SLC__1SDV_20180502T043026_20180502T043054_021721_025793_5C18.SAFE.geo
[scott@lidar igrams]$ head sbas_list
../S1A_IW_SLC__1SDV_20180420T043026_20180420T043054_021546_025211_81BE.SAFE.geo ../S1A_IW_SLC__1SDV_20180502T043026_20180502T043054_021721_025793_5C18.SAFE.geo 12.0   -16.733327776024169
[scott@lidar igrams]$ head intlist
20180420_20180502.int

"""
import os
import datetime
import numpy as np

from insar.parsers import Sentinel
from insar import sario
from insar import utils
from insar.log import get_log, log_runtime

SENTINEL_WAVELENGTH = 5.5465763  # cm
PHASE_TO_CM = SENTINEL_WAVELENGTH / (-4 * np.pi)

logger = get_log()


def read_geolist(filepath="./geolist"):
    """Reads in the list of .geo files used, in time order

    Args:
        filepath (str): path to the intlist file

    Returns:
        list[date]: the parse dates of each .geo used, in date order

    """
    with open(filepath) as f:
        geolist = [os.path.split(geoname)[1] for geoname in f.read().splitlines()]
    return sorted([Sentinel(geo).start_time().date() for geo in geolist])


def read_intlist(filepath="./intlist", parse=True):
    """Reads the list of igrams to return dates of images as a tuple

    Args:
        filepath (str): path to the intlist file
        parse (bool): output the intlist as parsed datetime tuples

    Returns:
        tuple(date, date) of master, slave dates for all igrams (if parse=True)
            if parse=False: returns list[str], filenames of the igrams

    """

    def _parse(datestr):
        return datetime.datetime.strptime(datestr, "%Y%m%d").date()

    with open(filepath) as f:
        intlist = f.read().splitlines()

    if parse:
        intlist = [intname.strip('.int').split('_') for intname in intlist]
        return [(_parse(master), _parse(slave)) for master, slave in intlist]
    else:
        dirname = os.path.dirname(filepath)
        return [os.path.join(dirname, igram) for igram in intlist]


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
    timediffs = find_time_diffs(geolist)

    A = build_A_matrix(geolist, intlist)
    B = np.zeros_like(A)

    for j, row in enumerate(A):
        # if no -1 entry, start at index 0. Otherwise, add 1 so exclude the -1 index
        start_idx = list(row).index(-1) + 1 if (-1 in row) else 0
        # End index is inclusive of the +1
        end_idx = np.where(row == 1)[0][0] + 1  # +1 will always exist in row

        # Now only fill in the time diffs in the range from the early igram index
        # to the later igram index
        B[j][start_idx:end_idx] = timediffs[start_idx:end_idx]

    return B


def read_stack(directory, file_ext):
    """Reads a set of images into a 3D ndarray

    Args:
        directory (str): path to a dir containing all files
        file_ext (str): ending type of files to read (e.g. '.unw')

    Returns:
        ndarray: 3D array of each file stacked
            1st dim is the index of the image: stack[0, :, :]
    """
    all_file_names = sorted(sario.find_files(directory, "*" + file_ext))
    all_files = [sario.load_file(filename) for filename in all_file_names]
    return np.stack(all_files, axis=0)


''' TODO: may not need this after all
def find_stack_max(stack):
    """Gets the row, col of the max value for the mean of the stack

    Args:
        stack (ndarray): 3D array of images, stacked along axis=0

    Returns:
        tuple[int, int]: row, col of the mean for the stack_mean
    """
    stack_mean = np.mean(stack, axis=0)
    # Argmax gives the flattened indices, so we need to convert back to row, col
    max_row, max_col = np.unravel_index(np.argmax(stack_mean), stack_mean.shape)
'''


def shift_stack(stack, ref_row, ref_col, window=3):
    """Subtracts reference pixel group from each layer

    Args:
        stack (ndarray): 3D array of images, stacked along axis=0
        ref_row (int): row index of the reference pixel to subtract
        ref_col (int): col index of the reference pixel to subtract
        window (int): size of the group around ref pixel to avg for reference.
            if window=1 or None, only the single pixel used to shift the group.

    Raises:
        ValueError: if window is not a positive int, or if ref pixel out of bounds
    """
    window = window or 1
    if not isinstance(window, int) or window < 1:
        raise ValueError("Invalid window %s: must be odd positive int" % window)
    elif ref_row > stack.shape[1] or ref_col > stack.shape[2]:
        raise ValueError("(%s, %s) out of bounds reference for stack size %s" % (ref_row, ref_col,
                                                                                 stack.shape))

    if window % 2 == 0:
        window -= 1
        logger.warning("Making window an odd number (%s) to get square window", window)

    win_size = window // 2
    shifted = np.empty_like(stack)
    for idx in range(stack.shape[0]):
        cur_layer = stack[idx, :, :]
        ref_group = cur_layer[ref_row - win_size:ref_row + win_size + 1,
                              ref_col - win_size:ref_col + win_size + 1]  # yapf: disable
        shifted[idx, :, :] = cur_layer - np.mean(ref_group)

    return shifted


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

    # print(diff_matrix)
    return diff_matrix


def invert_sbas(delta_phis, timediffs, B, alpha=0, difference=False):
    """Performs and SBAS inversion on each pixel of unw_stack to find deformation

    Solves the least squares equation Bv = dphi

    Args:
        delta_phis (ndarray): 1D array of unwrapped phases (delta phis)
            comes from 1 pixel of read_stack along 3rd axis
        B (ndarray): output of build_B_matrix for current set of igrams
        timediffs (np.array): dtype=int, days between each SAR acquisitions
            length will be equal to B.shape[1], 1 less than num SAR acquisitions
        alpha (float): nonnegative Tikhonov regularization parameter.
            If alpha > 0, then the equation is instead to minimize
            ||B*v - dphi||^2 + ||alpha*I*v||^2
            See https://en.wikipedia.org/wiki/Tikhonov_regularization
        difference (bool): for regularization, penalize differences in velocity
            Used to make a smoother final solution
            TODO: difference giving wonky results

    Returns:
        tuple[ndarray, ndarray]: solution velocity array, and integrated phase array

    """

    def _augment_matrices(B, delta_phis, alpha):
        reg_matrix = _create_diff_matrix(B.shape[1]) if difference else np.eye(B.shape[1])
        B = np.vstack((B, alpha * reg_matrix))
        # Now make num rows match
        zeros_shape = (B.shape[0] - delta_phis.shape[0], delta_phis.shape[1])
        delta_phis = np.vstack((delta_phis, np.zeros(zeros_shape)))
        return B, delta_phis

    if B.shape[1] != len(timediffs):
        raise ValueError("Shapes of B {} and timediffs {} not compatible".format(
            B.shape, timediffs.shape))
    elif B.shape[0] != delta_phis.shape[0]:
        raise ValueError("Shapes of B {} and delta_phis {} not compatible".format(
            B.shape, delta_phis.shape))
    elif alpha < 0:
        raise ValueError("alpha cannot be negative")

    # Augment only if regularization requested
    if alpha > 0:
        logger.info("Using regularization with alpha=%s, difference=%s", alpha, difference)
        B, delta_phis = _augment_matrices(B, delta_phis, alpha)

    # Velocity will be result of the inversion
    velocity_array, _, rank_B, sing_vals_B = np.linalg.lstsq(B, delta_phis, rcond=None)

    # velocity array entries: v_j = (phi_j - phi_j-1)/(t_j - t_j-1)
    if velocity_array.ndim == 1:
        velocity_array = np.expand_dims(velocity_array, axis=-1)

    # Now integrate to get back to phases
    # multiple each column of vel array: each col is a separate solution
    phi_diffs = timediffs.reshape((-1, 1)) * velocity_array

    # Now the final phase results are the cumulative sum of delta phis
    phi_arr = np.cumsum(phi_diffs, axis=0)
    # Add 0 as first entry of phase array to match geolist length on each col
    phi_arr = np.insert(phi_arr, 0, 0, axis=0)

    return velocity_array, phi_arr


def stack_to_cols(stacked, reverse=False):
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


@log_runtime
def run_inversion(igram_path,
                  reference=(None, None),
                  window=None,
                  alpha=0,
                  difference=False,
                  verbose=False):
    """Runs SBAS inversion on all unwrapped igrams

    Args:
        igram_path (str): path to the directory containing `intlist`,
            the .int filenames, the .unw files, and the dem.rsc file
        reference (tuple[int, int]): row and col index of the reference pixel to subtract
        window (int): size of the group around ref pixel to avg for reference.
            if window=1 or None, only the single pixel used to shift the group.
        alpha (float): nonnegative Tikhonov regularization parameter.
            See https://en.wikipedia.org/wiki/Tikhonov_regularization
        difference (bool): for regularization, penalize differences in velocity
            Used to make a smoother final solution
        verbose (bool): print extra timing and debug info

    Returns:
        geolist (list[datetime]): dates of each SAR acquisition from read_geolist
        phi_arr (ndarray): absolute phases of every pixel at each time
        deformation (ndarray): matrix of deformations at each pixel and time
        varr (ndarray): array of volocities solved for from SBAS inversion
        unw_stack (ndarray): output of read_unw_stack (if desired)
    """
    if verbose:
        logger.setLevel(10)  # DEBUG

    intlist_path = os.path.join(igram_path, 'intlist')
    geolist_path = os.path.join(igram_path, 'geolist')

    intlist = read_intlist(filepath=intlist_path)
    geolist = read_geolist(filepath=geolist_path)

    logger.debug("Reading stack")
    # unw_stack = read_unw_stack(igram_path, *reference)
    unw_stack = read_stack(igram_path, ".unw")
    # logger.debug("Reading stack complete")
    ref_row, ref_col = reference
    unw_stack = shift_stack(unw_stack, ref_row, ref_col, window=window)
    logger.debug("Shifting stack complete")

    # Prepare B matrix and timediffs used for each pixel inversion
    B = build_B_matrix(geolist, intlist)
    timediffs = find_time_diffs(geolist)

    # Save shape for end
    num_ints, rows, cols = unw_stack.shape
    phi_columns = stack_to_cols(unw_stack)

    varr, phi_arr = invert_sbas(phi_columns, timediffs, B, alpha=alpha, difference=difference)
    # Multiple by wavelength ratio to go from phase to cm
    deformation = PHASE_TO_CM * phi_arr

    # Now reshape all outputs that should be in stack form
    phi_arr = cols_to_stack(phi_arr, rows, cols)
    deformation = cols_to_stack(deformation, rows, cols)
    varr = cols_to_stack(varr, rows, cols)
    return (geolist, phi_arr, deformation, varr, unw_stack)


def save_deformation(igram_path, deformation, geolist):
    """Saves deformation ndarray and geolist dates as .npy file"""
    np.save(os.path.join(igram_path, 'deformation.npy'), deformation)
    np.save(os.path.join(igram_path, 'geolist.npy'), geolist)


def load_deformation(igram_path, ref_row=None, ref_col=None, alpha=0, difference=False):
    try:
        deformation = np.load(os.path.join(igram_path, 'deformation.npy'))
        # geolist is a list of datetimes: encoding must be bytes
        geolist = np.load(os.path.join(igram_path, 'geolist.npy'), encoding='bytes')

    except (IOError, OSError):
        if not ref_col and not ref_col:
            logger.error("deformation.npy or geolist.npy not found in path %s", igram_path)
            logger.error("Need ref_row, ref_col to run inversion and create files")
            return None, None
        else:
            logger.warning("No deformation.npy detected: running inversion")

        geolist, phi_arr, deformation, varr, unw_stack = run_inversion(
            igram_path, reference=(ref_row, ref_col), alpha=alpha, difference=difference)
        save_deformation(igram_path, deformation, geolist)

    return geolist, deformation


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
    # Note: rows == ys, cols are xs
    yidxs, xidxs = matrix_indices(z.shape, flatten=True)
    # c_ stacks 1D arrays as columns into a 2D array
    if order == 1:
        A = np.c_[xidxs, yidxs, np.ones(xidxs.shape)]
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
        a, b, c = coeffs
        y_block, x_block = matrix_indices(z.shape, flatten=False)
        return z - (a * x_block + b * y_block + c)
    elif order == 2:
        yy, xx = matrix_indices(z.shape, flatten=True)
        idx_matrix = np.c_[np.ones(xx.shape), xx, yy, xx * yy, xx**2, yy**2]
        z_fit = np.dot(idx_matrix, coeffs).reshape(z.shape)
        return z - z_fit
    else:
        raise NotImplementedError("Order only implemented for 1 and 2")
