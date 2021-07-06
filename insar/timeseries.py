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
import math
import hdf5plugin
import h5py
import numpy as np

from matplotlib.dates import date2num
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

    # Figure out how much to load at 1 time, staying at ~`block_size_max` bytes of RAM
    block_shape = _get_block_shape(
        full_shape, chunk_size, block_size_max=100e6, nbytes=nbytes
    )
    if constant_velocity:
        # proc_func = proc_pixel_linear
        output_shape = (nrows, ncols)
    else:
        # proc_func = proc_pixel_daily
        output_shape = (len(slclist), nrows, ncols)

    paramfile = "{}_run_params".format(outfile).replace(".", "_") + ".yml"
    # Saves all desried run variables and objects into a yaml file
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

    if sario.check_dset(outfile, output_dset, overwrite):
        create_dset(
            outfile, output_dset, output_shape, np.float32, chunks=True, compress=True
        )
    else:
        raise ValueError(f"{outfile}:/{output_dset} exists, {overwrite = }")

    run_sbas(
        unw_stack_file,
        input_dset,
        valid_ifg_idxs,
        outfile,
        output_dset,
        block_shape,
        date2num(slclist),
        date2num(ifglist),
        constant_velocity,
        alpha,
        # L1,
        outlier_sigma,
    )
    sario.save_slclist_to_h5(
        out_file=outfile, slc_date_list=slclist, dset_name=output_dset
    )
    dem_rsc = sario.load_dem_from_h5("unw_stack.h5")
    sario.save_dem_to_h5(outfile, dem_rsc)


from concurrent.futures import ProcessPoolExecutor, as_completed


def run_sbas(
    unw_stack_file,
    input_dset,
    valid_ifg_idxs,
    outfile,
    output_dset,
    block_shape,
    slclist,
    ifglist,
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
    # blk_slices = list(blk_slices)[:6]

    with ProcessPoolExecutor(max_workers=4) as executor:
        # for (rows, cols) in blk_slices:
        future_to_block = {
            executor.submit(
                _run_and_save,
                blk,
                unw_stack_file,
                input_dset,
                valid_ifg_idxs,
                slclist,
                ifglist,
                constant_velocity,
            ): blk
            for blk in blk_slices
        }
        for future in as_completed(future_to_block):
            blk = future_to_block[future]
            out_chunk = future.result()
            rows, cols = blk
            write_out_chunk(out_chunk, outfile, output_dset, rows, cols)


def _run_and_save(
    blk, unw_stack_file, input_dset, valid_ifg_idxs, slclist, ifglist, constant_velocity
):
    rows, cols = blk
    with h5py.File(unw_stack_file) as hf:
        nstack, nrows, ncols = hf[input_dset].shape
        logger.info(f"Loading chunk {rows}, {cols}")
        unw_chunk = hf[input_dset][valid_ifg_idxs, rows[0] : rows[1], cols[0] : cols[1]]
        out_chunk = calc_soln(
            # out_chunk = calc_soln(
            unw_chunk,
            slclist,
            ifglist,
            # alpha,
            constant_velocity,
        )
        return out_chunk


def write_out_chunk(chunk, outfile, output_dset, rows=None, cols=None):
    rows = rows or [0, None]
    cols = cols or [0, None]
    logger.info(f"Writing out ({rows = }, {cols = }) chunk to {outfile}:/{output_dset}")
    with h5py.File(outfile, "r+") as hf:
        hf[output_dset][:, rows[0] : rows[1], cols[0] : cols[1]] = chunk


try:
    import numba
    from .ts_numba import build_B_matrix, build_A_matrix

    deco = numba.njit

except:
    from .ts_utils import build_B_matrix

    # Identity decorator if the numba.jit ones fail
    deco = lambda func: func


@deco
def calc_soln(
    unw_chunk,
    slclist,
    ifglist,
    # alpha,
    constant_velocity,
    # L1 = True,
    # outlier_sigma=4,
):
    slcs_clean, ifglist_clean, unw_clean = slclist, ifglist, unw_chunk
    # if outlier_sigma > 0:
    #     slc_clean, ifglist_clean, unw_clean = remove_outliers(
    #         slc_clean, ifglist_clean, unw_clean, mean_sigma_cutoff=sigma
    #     )
    # igram_count = len(unw_clean)

    # Last, pad with zeros if doing Tikh. regularization
    # unw_final = alpha > 0 ? augment_zeros(B, unw_clean) : unw_clean

    # # Prepare B matrix and timediffs used for each pixel inversion
    # # B = prepB(slc_clean, ifglist_clean, constant_velocity, alpha)
    # B = build_B_matrix(
    #     slcs_clean, ifglist_clean, model="linear" if constant_velocity else None
    # )
    # timediffs = np.array([d.days for d in np.diff(slclist)])
    A = build_A_matrix(slcs_clean, ifglist_clean)
    pA = np.linalg.pinv(A).astype(unw_clean.dtype)
    nstack, nrow, ncol = unw_clean.shape
    # stack = cols_to_stack(pA @ stack_to_cols(unw_subset), *unw_subset.shape[1:])
    # equiv:
    stack = (pA @ unw_clean.reshape((nstack, -1))).reshape((-1, nrow, ncol))

    # Add a 0 image for the first date
    stack = np.concatenate((np.zeros((1, nrow, ncol)), stack), axis=0)
    stack *= PHASE_TO_CM
    return stack


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


def _get_block_shape(full_shape, chunk_size, block_size_max=10e6, nbytes=4):
    """Find a size of a data cube less than `block_size_max` in increments of `chunk_size`"""
    import copy

    chunks_per_block = block_size_max / (np.prod(chunk_size) * nbytes)
    row_chunks, col_chunks = 1, 1
    cur_block_shape = copy.copy(chunk_size)
    while chunks_per_block > 1:
        # First keep incrementing the number of rows we grab at once time
        if row_chunks * chunk_size[1] < full_shape[1]:
            row_chunks += 1
            cur_block_shape[1] = min(row_chunks * chunk_size[1], full_shape[1])
        # Then increase the column size if still haven't hit `block_size_max`
        elif col_chunks * chunk_size[2] < full_shape[2]:
            col_chunks += 1
            cur_block_shape[2] = min(col_chunks * chunk_size[2], full_shape[2])
        else:
            break
        chunks_per_block = block_size_max / (np.prod(cur_block_shape) * nbytes)
    return cur_block_shape


def _record_run_params(paramfile, **kwargs):
    from ruamel.yaml import YAML
    yaml = YAML()

    with open(paramfile, "w") as f:
        yaml.dump(kwargs, f)


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
