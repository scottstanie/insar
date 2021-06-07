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
import numpy as np

from apertools import sario, latlon
from apertools.log import get_log, log_runtime

SENTINEL_WAVELENGTH = 5.5465763  # cm
PHASE_TO_CM = SENTINEL_WAVELENGTH / (4 * np.pi)

logger = get_log()


@log_runtime
def run_inversion(
    igram_path,
    reference=(None, None),
    window=None,
    constant_velocity=False,
    alpha=0,
    difference=False,
    deramp=True,
    deramp_order=1,
    masking=True,
    geolist_ignore_file="geolist_missing.txt",
    verbose=False,
):
    """Runs SBAS inversion on all unwrapped igrams

    Args:
        igram_path (str): path to the directory containing `intlist`,
            the .int filenames, the .unw files, and the dem.rsc file
        reference (tuple[int, int]): row and col index of the reference pixel to subtract
        window (int): size of the group around ref pixel to avg for reference.
            if window=1 or None, only the single pixel used to shift the group.
        constant_velocity (bool): force solution to have constant velocity
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
        geolist (list[datetime]): dates of each SAR acquisition from find_geos
        phi_arr (ndarray): absolute phases of every pixel at each time
        deformation (ndarray): matrix of deformations at each pixel and time
    """
    if verbose:
        logger.setLevel(10)  # DEBUG

    geolist, intlist = load_geolist_intlist(
        igram_path, geolist_ignore_file=geolist_ignore_file, parse=True
    )

    # Prepare B matrix and timediffs used for each pixel inversion
    B = build_B_matrix(geolist, intlist)
    timediffs = find_time_diffs(geolist)
    if B.shape[1] != len(timediffs):
        raise ValueError(
            "Shapes of B {} and timediffs {} not compatible".format(
                B.shape, timediffs.shape
            )
        )

    logger.debug("Reading unw stack")
    unw_stack, mask_stack, geo_mask_columns = load_unw_masked_stack(
        igram_path,
        num_timediffs=len(timediffs),
        unw_ext=".unw",
        deramp=deramp,
        deramp_order=deramp_order,
        masking=masking,
    )

    # TODO: Process the correlation, mask very bad corr pixels in the igrams

    # Use the given reference, or find one on based on max correlation
    if any(r is None for r in reference):
        # Make a latlon image to check for gps data containment
        # TODO: maybe i need to search for masks? dont wanna pick a garbage one by accident
        latlon_image = latlon.LatlonImage(
            data=unw_stack[0], dem_rsc_file=os.path.join(igram_path, "dem.rsc")
        )
        ref_row, ref_col = find_reference_location(
            latlon_image, igram_path, mask_stack, gps_dir=None
        )
    else:
        ref_row, ref_col = reference

    logger.info(
        "Starting shift_stack: using %s, %s as ref_row, ref_col", ref_row, ref_col
    )
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
            constant_velocity=constant_velocity,
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


def find_time_diffs(date_list):
    """Finds the number of days between successive files

    Args:
        date_list (list[date]): dates of the SAR acquisitions

    Returns:
        np.array: days between each datetime in date_list
            dtype=int, length is a len(date_list) - 1"""
    return np.array([d.days for d in np.diff(date_list)])


def load_geolist_intlist(filepath, geolist_ignore_file=None, parse=True):
    """Load the geolist and intlist from a directory with igrams"""
    geolist = sario.find_geos(filepath, parse=parse)
    intlist = sario.find_igrams(filepath, parse=parse)
    if geolist_ignore_file is not None:
        ignore_filepath = os.path.join(filepath, geolist_ignore_file)
        geolist, intlist = ignore_geo_dates(
            geolist, intlist, ignore_file=ignore_filepath
        )
    return geolist, intlist


def ignore_geo_dates(geolist, intlist, ignore_file="geolist_missing.txt"):
    """Read extra file to ignore certain dates of interferograms"""
    ignore_geos = set(sario.find_geos(ignore_file))
    logger.info("Ignoreing the following .geo dates:")
    logger.info(sorted(ignore_geos))
    valid_geos = [g for g in geolist if g not in ignore_geos]
    valid_igrams = [
        i for i in intlist if i[0] not in ignore_geos and i[1] not in ignore_geos
    ]
    return valid_geos, valid_igrams


def build_A_matrix(sar_date_list, ifg_date_list):
    """Takes the list of igram dates and builds the SBAS A matrix

    Args:
        sar_date_list (list[date]): datetimes of the acquisitions
        ifg_date_list (list[tuple(date, date)])

    Returns:
        np.array 2D: the incident-like matrix from the SBAS paper: A*phi = dphi
            Each row corresponds to an igram, each column to a SAR date
            value will be -1 on the early (reference) igrams, +1 on later (secondary)
    """
    # We take the first .geo to be time 0, leave out of matrix
    # Match on date (not time) to find indices
    sar_date_list = sar_date_list[1:]
    M = len(ifg_date_list)  # Number of igrams, number of rows
    N = len(sar_date_list)
    A = np.zeros((M, N))
    for j in range(M):
        early_igram, late_igram = ifg_date_list[j]

        try:
            idx_early = sar_date_list.index(early_igram)
            A[j, idx_early] = -1
        except ValueError:  # The first SLC will not be in the matrix
            pass

        idx_late = sar_date_list.index(late_igram)
        A[j, idx_late] = 1

    return A


def build_B_matrix(sar_date_list, ifg_date_list):
    """Takes the list of igram dates and builds the SBAS B (velocity coeff) matrix

    Args:
        sar_date_list (list[date]): dates of the SAR acquisitions
        ifg_date_list (list[tuple(date, date)])

    Returns:
        np.array: 2D array of the velocity coefficient matrix from the SBAS paper:
                Bv = dphi
            Each row corresponds to an igram, each column to a SAR date
            value will be t_k+1 - t_k for columns after the -1 in A,
            up to and including the +1 entry
    """
    timediffs = find_time_diffs(sar_date_list)

    A = build_A_matrix(sar_date_list, ifg_date_list)
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
        patch = layer[
            ref_row - win : ref_row + win + 1, ref_col - win : ref_col + win + 1
        ]
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
        diff_matrix = -1 * np.diag(np.ones(n - 1), k=1).astype("int")
        np.fill_diagonal(diff_matrix, 1)
        diff_matrix = diff_matrix[:-1, :]
    elif order == 2:
        diff_matrix = -1 * np.diag(np.ones(n - 1), k=1).astype("int")
        diff_matrix = diff_matrix + -1 * np.diag(np.ones(n - 1), k=-1).astype("int")
        np.fill_diagonal(diff_matrix, 2)
        diff_matrix[-1, -1] = 1
        diff_matrix[0, 0] = 1

    return diff_matrix


def _augment_matrices(B, delta_phis, alpha, difference=False):
    reg_matrix = _create_diff_matrix(B.shape[1]) if difference else np.eye(B.shape[1])
    B = np.vstack((B, alpha * reg_matrix))
    # Now make num rows match
    zeros_shape = (B.shape[0] - delta_phis.shape[0], delta_phis.shape[1])
    delta_phis = np.vstack((delta_phis, np.zeros(zeros_shape)))
    return B, delta_phis


def _augment_zeros(B, delta_phis):
    try:
        r, c = delta_phis.shape
    except ValueError:
        delta_phis = delta_phis.reshape((-1, 1))
        r, c = delta_phis.shape
    zeros_shape = (B.shape[0] - r, c)
    delta_phis = np.vstack((delta_phis, np.zeros(zeros_shape)))
    return delta_phis


def prepB(geolist, intlist, constant_velocity=False, alpha=0, difference=False):
    """TODO: transfer this to the "invert_sbas"? this is from julia"""
    B = build_B_matrix(geolist, intlist)
    # Adjustments to solution:
    # Force velocity constant across time
    if constant_velocity is True:
        logger.info("Using a constant velocity for inversion solutions.")
        B = np.expand_dims(np.sum(B, axis=1), axis=1)
    # Add regularization to the solution
    elif alpha > 0:
        logger.info(
            "Using regularization with alpha=%s, difference=%s", alpha, difference
        )
        # Augment only if regularization requested
        reg_matrix = (
            _create_diff_matrix(B.shape[1]) if difference else np.eye(B.shape[1])
        )
        B = np.vstack((B, alpha * reg_matrix))
    return B


def invert_sbas(
    delta_phis,
    B,
    geo_mask_columns=None,
    constant_velocity=False,
    alpha=0,
    difference=False,
):
    """Performs and SBAS inversion on each pixel of unw_stack to find deformation

    Solves the least squares equation Bv = dphi

    Args:
        delta_phis (ndarray): columns of unwrapped phases (delta phis)
            Each col is 1 pixel of load_stack along 3rd axis
        B (ndarray): output of build_B_matrix for current set of igrams
        geo_mask_columns (ndarray[bool]): .geo file masks, reshaped to columns
        constant_velocity (bool): force solution to have constant velocity
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

    if B.shape[0] != delta_phis.shape[0]:
        raise ValueError(
            "Shapes of B {} and delta_phis {} not compatible".format(
                B.shape, delta_phis.shape
            )
        )
    elif alpha < 0:
        raise ValueError("alpha cannot be negative")

    # Adjustments to solution:
    # Force velocity constant across time
    if constant_velocity is True:
        logger.info("Using a constant velocity for inversion solutions.")
        B = np.expand_dims(np.sum(B, axis=1), axis=1)
    # Add regularization to the solution
    elif alpha > 0:
        logger.info(
            "Using regularization with alpha=%s, difference=%s", alpha, difference
        )
        # Augment only if regularization requested
        B, delta_phis = _augment_matrices(B, delta_phis, alpha)

    # Velocity will be result of the inversion
    # velocity_array, _, rank_B, sing_vals_B = np.linalg.lstsq(B, delta_phis, rcond=None)
    # velocity_array = mask.masked_lstsq(B, delta_phis, geo_mask_columns)
    raise ValueError()

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
