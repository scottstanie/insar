"""timeseries.py
Functions for performing time series analysis of unwrapped interferograms

files in the igrams folder:
    slclist, ifglist, sbas_list
scott@lidar igrams]$ head slclist
../S1A_IW_SLC__1SDV_20180420T043026_20180420T043054_021546_025211_81BE.SAFE.geo
../S1A_IW_SLC__1SDV_20180502T043026_20180502T043054_021721_025793_5C18.SAFE.geo
[scott@lidar igrams]$ head sbas_list
../S1A_IW_SLC__1SDV_20180420T043026_20180420T043054_021546_025211_81BE.SAFE.geo \
        ../S1A_IW_SLC__1SDV_20180502T043026_20180502T043054_021721_025793_5C18.SAFE.geo 12.0   \
        -16.733327776024169
[scott@lidar igrams]$ head ifglist
20180420_20180502.int

"""
import os
import hdf5plugin
import h5py
import numpy as np

from apertools import sario, latlon, utils, gps
from apertools.log import get_log, log_runtime
from .prepare import create_dset
from . import constants

SENTINEL_WAVELENGTH = 5.5465763  # cm
PHASE_TO_CM = SENTINEL_WAVELENGTH / (4 * np.pi)

logger = get_log()


@log_runtime
def run_inversion(
    unw_stack_file="unw_stack.h5",
    input_dset=constants.STACK_FLAT_SHIFTED_DSET,
    outfile="deformation_stack.h5",
    overwrite=False,
    dem_rsc_file="dem.rsc",
    min_date=None,
    max_date=None,
    stack_average=False,
    constant_velocity=False,
    max_temporal_baseline=800,
    max_temporal_bandwidth=None,  # TODO
    outlier_sigma=0,  # TODO: outlier outlier_sigma. Use trodi
    alpha=0,
    # L1=False, # TODO
    # difference=False,
    slclist_ignore_file="slclist_ignore.txt",
    verbose=False,
):
    """Runs SBAS inversion on all unwrapped igrams

    Args:
        unw_stack_file (str): path to the directory containing `unw_stack`,
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
        slclist_ignore_file (str): text file with list of .geo files to ignore
            Removes the .geo and and igrams with these date
        masking (bool): flag to load stack of .int.mask files to mask invalid areas
        verbose (bool): print extra timing and debug info

    Returns:
        slclist (list[datetime]): dates of each SAR acquisition from find_geos
        phi_arr (ndarray): absolute phases of every pixel at each time
        deformation (ndarray): matrix of deformations at each pixel and time
    """
    if verbose:
        logger.setLevel(10)  # DEBUG

    # averaging or linear means output will is 3D array (not just map of velocities)
    is_3d = not (stack_average or constant_velocity)
    output_dset = "stack" if is_3d else "velos"
    rsc_data = sario.load(dem_rsc_file)

    slclist, ifglist = sario.load_slclist_ifglist(
        h5file=unw_stack_file,
        slclist_ignore_file=slclist_ignore_file,
    )

    slclist, ifglist, valid_ifg_idxs = utils.filter_slclist_ifglist(
        ifg_date_list=ifglist,
        min_date=min_date,
        max_date=max_date,
        max_temporal_baseline=max_temporal_baseline,
        max_bandwidth=max_temporal_bandwidth,
    )

    with h5py.File(unw_stack_file) as hf:
        full_shape = hf[input_dset].shape
        nstack, nrows, ncols = full_shape
        nbytes = hf[input_dset].dtype.itemsize
        chunk_size = list(hf[input_dset].chunks) or [nstack, 10, 10]
        chunk_size[0] = nstack  # always load a full depth slice at once

    # Figure out how much to load at 1 type
    block_shape = _get_block_shape(
        full_shape, chunk_size, block_size_max=1e9, nbytes=nbytes
    )
    if constant_velocity:
        # proc_func = proc_pixel_linear
        output_shape = (nrows, ncols)
    else:
        # proc_func = proc_pixel_daily
        output_shape = (nrows, ncols, len(slclist))

    paramfile = "{}_run_params".format(outfile).replace(".", "_") + ".toml"
    # Saves all desried run variables and objects into a .toml file
    _record_run_params(
        paramfile,
        outfile=outfile,
        output_dset=output_dset,
        unw_stack_file=unw_stack_file,
        input_dset=input_dset,
        min_date=min_date,
        max_date=max_date,
        max_temporal_baseline=max_temporal_baseline,
        max_bandwidth=max_temporal_bandwidth,
        outlier_sigma=outlier_sigma,
        alpha=alpha,
        # L1=False,
        # difference=difference,
        slclist_ignore=open(slclist_ignore_file).read().splitlines(),
        block_shape=block_shape,
    )

    # if os.path.exists(outfile):
    sario.check_dset(outfile, output_dset, overwrite)
    create_dset(
        outfile, output_dset, output_shape, np.float32, chunks=True, compress=True
    )

    velo_file_out = run_sbas(
        unw_stack_file,
        input_dset,
        outfile,
        output_dset,
        block_shape,
        slclist,
        ifglist,
        valid_ifg_idxs,
        constant_velocity,
        alpha,
        # L1,
        outlier_sigma,
    )

    # # Multiple by wavelength ratio to go from phase to cm
    # deformation = PHASE_TO_CM * phi_arr

    # num_ints, rows, cols = unw_stack.shape
    # # Now reshape all outputs that should be in stack form
    # phi_arr = cols_to_stack(phi_arr, rows, cols)
    # deformation = cols_to_stack(deformation, rows, cols).filled(np.NaN)
    # return (slclist, phi_arr, deformation)


def _get_block_shape(full_shape, chunk_size, block_size_max=1e9, nbytes=4):
    import copy

    chunks_per_block = block_size_max / (np.prod(chunk_size) * nbytes)
    row_chunks, col_chunks = 1, 1
    cur_block_shape = copy.copy(chunk_size)
    while chunks_per_block > 1:
        if row_chunks * chunk_size[1] < full_shape[1]:
            row_chunks += 1
            cur_block_shape[1] = row_chunks * chunk_size[1]
        elif col_chunks * chunk_size[2] < full_shape[2]:
            col_chunks += 1
            cur_block_shape[2] = col_chunks * chunk_size[2]
        else:
            break
        chunks_per_block = block_size_max / (np.prod(cur_block_shape) * nbytes)
    return cur_block_shape


def run_sbas(
    unw_stack_file,
    input_dset,
    outfile,
    output_dset,
    block_shape,
    slclist,
    ifglist,
    valid_ifg_i,
    constant_velocity,
    alpha,
    # L1,
    outlier_sigma=0,
):
    """Performs and SBAS inversion on each pixel of unw_stack to find deformation

    Solves the least squares equation Bv = dphi

    Args:

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

    if alpha < 0:
        raise ValueError("alpha cannot be negative")

    with h5py.File(unw_stack_file) as hf:
        nstack, nrows, ncols = hf[input_dset].shape

    print(nrows, ncols, block_shape)
    blk_slices = utils.block_iterator((nrows, ncols), block_shape[-2:], overlaps=(0, 0))
    for (rows, cols) in blk_slices:
        with h5py.File(unw_stack_file) as hf:
            nstack, nrows, ncols = hf[input_dset].shape
            unw_chunk = hf[input_dset][:, rows[0] : rows[1], cols[0] : cols[1]]
            out_chunk = calc_soln(unw_chunk)
            write_out_chunk(out_chunk, outfile, output_dset)

    # if alpha > 0:
    #     logger.info(
    #         "Using regularization with alpha=%s, difference=%s", alpha, difference
    #     )
    #     # Augment only if regularization requested
    #     B, delta_phis = _augment_matrices(B, delta_phis, alpha)


def calc_soln(
    unw_chunk,
    slclist,
    ifglist,
    alpha,
    constant_velocity,
    # L1 = True,
    cor_pixel=None,
    cor_thresh=0.0,
    prune_outliers=True,
    outlier_sigma=4,
):
    slcs_clean, ifglist_clean, unw_clean = slclist, ifglist, unw_chunk
    # if outlier_sigma > 0:
    #     slc_clean, ifglist_clean, unw_clean = remove_outliers(
    #         slc_clean, ifglist_clean, unw_clean, mean_sigma_cutoff=sigma
    #     )
    igram_count = len(unw_clean)

    # Last, pad with zeros if doing Tikh. regularization
    # unw_final = alpha > 0 ? augment_zeros(B, unw_clean) : unw_clean

    # Prepare B matrix and timediffs used for each pixel inversion
    # B = prepB(slc_clean, ifglist_clean, constant_velocity, alpha)
    B = build_B_matrix(slcs_clean, ifglist_clean, model="linear" if constant_velocity else None)
    timediffs = find_time_diffs(slclist)



def find_time_diffs(date_list):
    """Finds the number of days between successive files

    Args:
        date_list (list[date]): dates of the SAR acquisitions

    Returns:
        np.array: days between each datetime in date_list
            dtype=int, length is a len(date_list) - 1"""
    return np.array([d.days for d in np.diff(date_list)])


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


def build_B_matrix(sar_dates, ifg_date_list, model=None):
    """Takes the list of igram dates and builds the SBAS B (velocity coeff) matrix

    Args:
        sar_date_list (list[date]): dates of the SAR acquisitions
        ifg_date_list (list[tuple(date, date)])
        model (str): If 'linear', creates the M x 1 matrix for linear velo model

    Returns:
        np.array: 2D array of the velocity coefficient matrix from the SBAS paper:
                Bv = dphi
            Each row corresponds to an igram, each column to a SAR date
            value will be t_k+1 - t_k for columns after the -1 in A,
            up to and including the +1 entry
    """
    try:
        sar_dates = sar_dates.date
    except AttributeError:
        pass
    timediffs = np.array([difference.days for difference in np.diff(sar_dates)])

    A = build_A_matrix(sar_dates, ifg_date_list)
    B = np.zeros_like(A)

    for j, row in enumerate(A):
        # if no -1 entry, start at index 0. Otherwise, add 1 to exclude the -1 index
        start_idx = list(row).index(-1) + 1 if (-1 in row) else 0
        # End index is inclusive of the +1
        end_idx = np.where(row == 1)[0][0] + 1  # +1 will always exist in row

        # Now only fill in the time diffs in the range from the early igram index
        # to the later igram index
        B[j][start_idx:end_idx] = timediffs[start_idx:end_idx]

    if model == "linear":
        return B.sum(axis=1, keepdims=True)
    else:
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


def prepB(slclist, ifglist, constant_velocity=False, alpha=0, difference=False):
    """TODO: transfer this to the "run_sbas"? this is from julia"""
    B = build_B_matrix(slclist, ifglist)
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


def _record_run_params(paramfile, **kwargs):
    import toml

    with open(paramfile, "w") as f:
        toml.dump(kwargs, f)


def integrate_velocities(velocity_array, timediffs):
    """Takes SBAS velocity output and finds phases

    Args:
        velocity_array (ndarray): output of run_sbas, velocities at
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
    # Add 0 as first entry of phase array to match slclist length on each col
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
