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
    all_file_names = sario.find_files(directory, "*" + file_ext)
    all_files = [sario.load_file(filename) for filename in all_file_names]
    return np.stack(all_files, axis=0)


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


def read_unw_stack(igram_path, ref_row, ref_col):
    """Reads all unwrapped phase .unw files into unw_stack

    Uses ref_row, ref_col as the normalizing point (subtracts
        that pixels value from all others in each .unw file)

    Args:
        igram_path (str): path to the directory containing `intlist`,
            the .int filenames, the .unw files, and the dem.rsc file
        ref_row (int): row index of the reference pixel to subtract
        ref_col (int): col index of the reference pixel to subtract

    Returns:
        ndarray: 3D array of each unw file stacked
            1st dim is the index of the igram: unw_stack[0, :, :]

    """

    def _allocate_stack(igram_path, num_ints):
        # Get igram file size data to pre-allocate space for 3D unw stack
        rsc_path = os.path.join(igram_path, 'dem.rsc')
        rsc_data = sario.load_file(rsc_path)
        rows = rsc_data['FILE_LENGTH']
        cols = rsc_data['WIDTH']
        return np.empty((num_ints, rows, cols), dtype='float32')

    intlist_path = os.path.join(igram_path, 'intlist')
    igram_files = read_intlist(intlist_path, parse=False)
    num_ints = len(igram_files)

    unw_stack = _allocate_stack(igram_path, num_ints)

    for idx, igram_file in enumerate(igram_files):
        unw_file = igram_file.replace('.int', '.unw')
        cur_unw = sario.load_file(unw_file)
        try:
            unw_stack[idx, :, :] = cur_unw - cur_unw[ref_row, ref_col]
        except IndexError:
            logger.error("Reference pixel (%s, %s) is out of bounds for unw shape %s", ref_row,
                         ref_col, unw_stack.shape[1:])
            raise
    return unw_stack


def invert_sbas(delta_phis, timediffs, B):
    """Performs and SBAS inversion on each pixel of unw_stack to find deformation

    Args:
        delta_phis (ndarray): 1D array of unwrapped phases (delta phis)
            comes from 1 pixel of read_unw_stack along 3rd axis
        B (ndarray): output of build_B_matrix for current set of igrams
        timediffs (np.array): dtype=int, days between each SAR acquisitions
            length will be equal to B.shape[1], 1 less than num SAR acquisitions

    return

    """
    assert B.shape[1] == len(timediffs)

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
def run_inversion(igram_path, reference=(None, None), verbose=False):
    """Runs SBAS inversion on all unwrapped igrams

    Args:
        igram_path (str): path to the directory containing `intlist`,
            the .int filenames, the .unw files, and the dem.rsc file
        reference (tuple[int, int]): row and col index of the reference pixel to subtract
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
    unw_stack = read_unw_stack(igram_path, *reference)
    logger.debug("Reading stack complete")

    # Prepare B matrix and timediffs used for each pixel inversion
    B = build_B_matrix(geolist, intlist)
    timediffs = find_time_diffs(geolist)

    # Save shape for end
    num_ints, rows, cols = unw_stack.shape
    phi_columns = stack_to_cols(unw_stack)

    varr, phi_arr = invert_sbas(phi_columns, timediffs, B)
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


def load_deformation(igram_path, ref_row=None, ref_col=None):
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
            igram_path, reference=(ref_row, ref_col))
        save_deformation(igram_path, deformation, geolist)

    return geolist, deformation
