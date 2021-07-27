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
from concurrent.futures import ProcessPoolExecutor, as_completed
import hdf5plugin  # noqa
import h5py
import numpy as np
from matplotlib.dates import date2num

from apertools import sario, utils
from apertools.log import get_log, log_runtime
from .prepare import create_dset
from . import constants

logger = get_log()

# Import numba if available; otherwise, just use python-only version
try:
    import numba
    from .ts_numba import build_A_matrix

    jit_decorator = numba.njit

except:
    logger.info("Numba not avialable, falling back to python-only")
    from .ts_utils import build_A_matrix

    # Identity decorator if the numba.jit ones fail
    def jit_decorator(func):
        return func


@log_runtime
def run_inversion(
    unw_stack_file=constants.UNW_FILENAME,
    input_dset=constants.STACK_FLAT_SHIFTED_DSET,
    outfile=constants.DEFORMATION_FILENAME,
    output_dset=constants.STACK_DSET,
    overwrite=False,
    min_date=None,
    max_date=None,
    stack_average=False,
    # constant_velocity=False,
    max_temporal_baseline=800,
    max_temporal_bandwidth=None,  # TODO
    outlier_sigma=0,  # TODO: outlier outlier_sigma. Use trodi
    alpha=0,
    # L1=False, # TODO
    # difference=False,
    slclist_ignore_file="slclist_ignore.txt",
    save_as_netcdf=True,
    verbose=False,
):
    """Runs SBAS inversion on all unwrapped igrams

    Args:
        unw_stack_file (str): path to the directory containing `unw_stack`,
            the .int filenames, the .unw files, and the dem.rsc file
        input_dset (str): input dataset in the `unw_stack_file`
            default: `constants.UNW_FILENAME`
        outfile (str): Name of HDF5 output file
        output_dset (str): name of dataset within `outfile` to store data
        overwrite (bool): If True, clobber `outfile:/output_dset`
        min_date (datetime.date): only take ifgs from `unw_stack_file` *after* this date
        max_date (datetime.date): only take ifgs from `unw_stack_file` *before* this date
        max_temporal_baseline (int): limit ifgs from `unw_stack_file` to be
            shorter in temporal baseline
        alpha (float): nonnegative Tikhonov regularization parameter.
            See https://en.wikipedia.org/wiki/Tikhonov_regularization
        difference (bool): for regularization, penalize differences in velocity
            Used to make a smoother final solution
        slclist_ignore_file (str): text file with list of .geo files to ignore
            Removes the .geo and and igrams with these date
        save_as_netcdf (bool): if true, also save the `outfile` as `outfile`.nc for
            easier manipulation with xarray
        verbose (bool): print extra timing and debug info

    Returns:
        slclist (list[datetime]): dates of each SAR acquisition from find_geos
        phi_arr (ndarray): absolute phases of every pixel at each time
        deformation (ndarray): matrix of deformations at each pixel and time
    """
    if verbose:
        logger.setLevel(10)  # DEBUG

    # averaging or linear means output will is 3D array (not just map of velocities)
    # is_3d = not (stack_average or constant_velocity)
    # output_dset = "stack" if is_3d else "velos"

    slclist, ifglist = sario.load_slclist_ifglist(h5file=unw_stack_file)

    slclist, ifglist, valid_ifg_idxs = utils.filter_slclist_ifglist(
        ifg_date_list=ifglist,
        min_date=min_date,
        max_date=max_date,
        slclist_ignore_file=slclist_ignore_file,
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

    # if constant_velocity:
    # proc_func = proc_pixel_linear
    # output_shape = (nrows, ncols)
    # else:
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
        # constant_velocity,
        alpha,
        # L1,
        outlier_sigma,
    )
    sario.save_slclist_to_h5(
        out_file=outfile, slc_date_list=slclist, dset_name=output_dset
    )
    dem_rsc = sario.load_dem_from_h5(unw_stack_file)
    sario.save_dem_to_h5(outfile, dem_rsc)
    if save_as_netcdf:
        from apertools import netcdf

        netcdf.hdf5_to_netcdf(
            outfile,
            outname=constants.DEFORMATION_FILENAME_NC,
            dset_name=output_dset,
            stack_dim="date",
        )


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
    # blk_slices = list(blk_slices)[:6]  # Test small area

    with ProcessPoolExecutor(max_workers=4) as executor:
        # for (rows, cols) in blk_slices:
        future_to_block = {
            executor.submit(
                _load_and_run,
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


def _load_and_run(
    blk, unw_stack_file, input_dset, valid_ifg_idxs, slclist, ifglist, constant_velocity
):
    rows, cols = blk
    with h5py.File(unw_stack_file) as hf:
        logger.info(f"Loading chunk {rows}, {cols}")
        unw_chunk = hf[input_dset][valid_ifg_idxs, rows[0] : rows[1], cols[0] : cols[1]]
        # TODO: get rid of nan pixels at edge! dont let it ruin the whole chunk
        out_chunk = calc_soln(
            # out_chunk = calc_soln_pixelwise(
            unw_chunk,
            slclist,
            ifglist,
            # alpha,
            # constant_velocity,
        )
        return out_chunk


def write_out_chunk(chunk, outfile, output_dset, rows=None, cols=None):
    rows = rows or [0, None]
    cols = cols or [0, None]
    logger.info(f"Writing out ({rows = }, {cols = }) chunk to {outfile}:/{output_dset}")
    with h5py.File(outfile, "r+") as hf:
        hf[output_dset][:, rows[0] : rows[1], cols[0] : cols[1]] = chunk


@jit_decorator
def calc_soln(
    unw_chunk,
    slclist,
    ifglist,
    # alpha,
    # constant_velocity,
    # L1 = True,
    # outlier_sigma=4,
):
    # TODO: this is where i'd get rid of specific dates/ifgs
    slcs_clean, ifglist_clean, unw_clean = slclist, ifglist, unw_chunk
    dtype = unw_clean.dtype

    nstack, nrow, ncol = unw_clean.shape
    unw_cols = unw_clean.reshape((nstack, -1))
    nan_idxs = np.isnan(unw_cols)
    unw_cols_nonan = np.where(nan_idxs, 0, unw_cols).astype(dtype)
    # skip any all 0 blocks:
    if unw_cols_nonan.sum() == 0:
        return np.zeros((len(slcs_clean), nrow, ncol), dtype=dtype)

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
    pA = np.linalg.pinv(A).astype(dtype)
    # stack = cols_to_stack(pA @ stack_to_cols(unw_subset), *unw_subset.shape[1:])
    # equiv:
    stack = (pA @ unw_cols_nonan).reshape((-1, nrow, ncol)).astype(dtype)

    # Add a 0 image for the first date
    stack = np.concatenate((np.zeros((1, nrow, ncol), dtype=dtype), stack), axis=0)
    stack *= constants.PHASE_TO_CM
    return stack


# @jit_decorator
@numba.njit(fastmath=True, parallel=True, cache=True, nogil=True)
def calc_soln_pixelwise(
    unw_chunk,
    slclist,
    ifglist,
    # alpha,
    # constant_velocity,
    # L1 = True,
    # outlier_sigma=4,
):
    slcs_clean, ifglist_clean, unw_clean = slclist, ifglist, unw_chunk

    nsar = len(slclist)
    _, nrow, ncol = unw_clean.shape

    stack = np.zeros((nsar, nrow, ncol))

    for idx in range(nrow):
        for jdx in range(ncol):
            A = build_A_matrix(slcs_clean, ifglist_clean)
            pA = np.linalg.pinv(A).astype(unw_clean.dtype)
            # the slice would not be contiguous, which makes @ slower
            cur_pixel = np.ascontiguousarray(unw_clean[:, idx, jdx])
            cur_soln = pA @ cur_pixel
            # first date is 0
            stack[1:, idx, jdx] = cur_soln

    # stack = (pA @ unw_clean.reshape((nstack, -1))).reshape((-1, nrow, ncol))

    stack *= constants.PHASE_TO_CM
    return stack


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


def fit_poly_to_stack(
    stack_fname=constants.DEFORMATION_FILENAME_NC,
    stack_dset=constants.STACK_DSET,
    degree=1,
    save=True,
    overwrite=True,
):
    import xarray as xr

    ds = xr.open_dataset(stack_fname)
    stack_da = ds[stack_dset]
    stack_poly = stack_da.polyfit("date", deg=degree)
    print(stack_da.max())
    print(xr.polyval(stack_da.date, stack_poly.polyfit_coefficients)[-1].max())

    # stack_poly["polyfit_coefficients"][0] is the offset/y-intercept
    # stack_poly["polyfit_coefficients"][1] is the rate
    velocities_cm_per_ns = stack_poly["polyfit_coefficients"][1]
    print(stack_poly["polyfit_coefficients"][0].max())
    print(stack_poly["polyfit_coefficients"][1].max())
    # Note: xarray converts the dates to "nanoseconds sicne 1970"
    # https://github.com/pydata/xarray/blob/main/xarray/core/missing.py#L273-L280
    # So the rate is "radians / nanosecond" (pecos is +/- 75 ish)
    velocities_cm_per_year = velocities_cm_per_ns * 1e-9 * 86400 * 365.25
    ds.close()
    if save:
        # ds[constants.VELOCITIES_DSET] = velocities_cm_per_year
        # ds[constants.VELOCITIES_DSET].units = "cm per year"
        dsname = constants.VELOCITIES_DSET
        if sario.check_dset(stack_fname, dsname, overwrite):
            logger.info("Saving linear velocities to %s", dsname)
            velo_ds = velocities_cm_per_year.to_dataset(name=dsname)
            velo_ds.attrs['units'] = "cm per year"
            velo_ds.to_netcdf(stack_fname, mode="a")

        dsname = constants.CUMULATIVE_LINEAR_DEFO_DSET
        if sario.check_dset(stack_fname, dsname, overwrite):
            logger.info("Saving cumulative linear velocity to %s", dsname)
            cumulative = xr.polyval(stack_da.date, stack_poly.polyfit_coefficients)[-1]
            cumulative.attrs['units'] = "cm"
            cum_ds = cumulative.to_dataset(name=dsname)
            cum_ds.to_netcdf(stack_fname, mode="a")
        # ds[constants.CUMULATIVE_LINEAR_DEFO_DSET] = cumulative
        # ds[constants.CUMULATIVE_LINEAR_DEFO_DSET].units = "cm"

    # For an easy cumulative:
    return velocities_cm_per_year