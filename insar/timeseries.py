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
from concurrent.futures import ProcessPoolExecutor, as_completed

# try:
#     import hdf5plugin  # noqa : for the possiblity of HDF5 blosc filter
# except ImportError:
#     print("Cant import hdf5plugin. wont read Blosc compressed files")

import h5py
import xarray as xr
import numpy as np
from matplotlib.dates import date2num

from apertools import sario, utils
from apertools.log import get_log, log_runtime
from .prepare import create_dset, detect_rdr_coordinates
from . import constants
from apertools.constants import PHASE_TO_CM_MAP, WAVELENGTH_MAP

# from .ts_utils import get_cor_mean, ptp_by_date, ptp_by_date_pct
# from insar import ts_utils
from .ts_utils import ptp_by_date_pct, DummyExecutor

PARALLEL = True  # With false, uses dummy Executor (for debugging)
MAX_WORKERS = 6
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
    unw_stack_file=sario.UNW_FILENAME,
    input_dset=sario.STACK_FLAT_SHIFTED_DSET,
    outfile=constants.DEFO_FILENAME,
    output_dset=constants.DEFO_NOISY_DSET,
    cor_stack_file=sario.COR_FILENAME,
    cor_dset=constants.COR_MEAN_DSET,
    los_file=constants.LOS_ENU_FILENAME,
    overwrite=False,
    min_date=None,
    max_date=None,
    stack_average=False,
    # constant_velocity=False,
    max_temporal_baseline=None,
    max_temporal_bandwidth=None,  # TODO
    min_temporal_bandwidth=None,  # TODO
    include_annual=False,
    # outlier_sigma=0,  # TODO: outlier outlier_sigma. Use trodi
    alpha=0,
    # L1=False, # TODO
    # difference=False,
    slclist_ignore_file="slclist_ignore.txt",
    save_as_netcdf=True,
    coordinates=None,  # geo, rdr
    platform="s1",
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

    Returns:
        slclist (list[datetime]): dates of each SAR acquisition from find_geos
        phi_arr (ndarray): absolute phases of every pixel at each time
        deformation (ndarray): matrix of deformations at each pixel and time
    """

    # averaging or linear means output will is 3D array (not just map of velocities)
    # is_3d = not (stack_average or constant_velocity)
    # output_dset = "stack" if is_3d else "velos"
    if coordinates is None:
        coordinates = detect_rdr_coordinates(os.path.dirname(unw_stack_file))

    slclist, ifglist = sario.load_slclist_ifglist(h5file=unw_stack_file)
    if not os.path.exists(slclist_ignore_file):
        logger.info("%s does not exist. Creating empty file", slclist_ignore_file)
        open(slclist_ignore_file, "w").close()

    slclist, ifglist, valid_ifg_idxs = utils.filter_slclist_ifglist(
        ifg_date_list=ifglist,
        min_date=min_date,
        max_date=max_date,
        slclist_ignore_file=slclist_ignore_file,
        max_temporal_baseline=max_temporal_baseline,
        max_bandwidth=max_temporal_bandwidth,
        min_bandwidth=min_temporal_bandwidth,
        include_annual=include_annual,
    )

    # Gather all unw stack metadata for processing, and for copying over (reference,...)
    with h5py.File(unw_stack_file) as hf:
        full_shape = hf[input_dset].shape
        nstack, nrows, ncols = full_shape
        nbytes = hf[input_dset].dtype.itemsize
        try:
            chunk_size = list(hf[input_dset].chunks) or [nstack, 10, 10]
            chunk_size[0] = nstack  # always load a full depth slice at once
        except:
            chunk_size = [nstack, 100, 100]
        # The attrs set by HDF5/xarray have keys that are capitalized. skip those
        # Note: .tolist() needed to convert numpy objects into lists/ints
        attrs_to_copy = {
            k: v for k, v in hf[input_dset].attrs.items() if k != k.upper()
        }

    keys_to_remove = []
    for k, v in attrs_to_copy.items():
        if k == "coordinates":
            keys_to_remove.append(k)
        elif hasattr(v, "__array_interface__"):
            # elif isinstance(v, np.ndarray):
            attrs_to_copy[k] = v.tolist()
    for k in keys_to_remove:
        attrs_to_copy.pop(k)

    # Figure out how much to load at 1 time, staying at ~`block_size_max` bytes of RAM
    block_shape = _get_block_shape(
        full_shape, chunk_size, block_size_max=100e6, nbytes=nbytes
    )

    # TODO: platform = standardize(..)
    phase_to_cm = PHASE_TO_CM_MAP[platform]

    # if constant_velocity:
    # proc_func = proc_pixel_linear
    # output_shape = (nrows, ncols)
    # else:
    # proc_func = proc_pixel_daily
    output_shape = (len(slclist), nrows, ncols)

    paramfile = (
        "{}_{}_run_params".format(outfile, output_dset).replace(".", "_") + ".yml"
    )
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
        alpha=alpha,
        slclist_ignore=open(slclist_ignore_file).read().splitlines(),
        block_shape=block_shape,
        platform=platform,
        wavelength=WAVELENGTH_MAP[platform],
        coordinates=coordinates,
        **attrs_to_copy,
    )

    if sario.check_dset(outfile, output_dset, overwrite) is False:
        raise ValueError(f"{outfile}:/{output_dset} exists, {overwrite = }")

    create_dset(outfile, output_dset, output_shape, np.float32, chunks=True)

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
        phase_to_cm,
        # outlier_sigma,
    )

    # Now save the ifg/slc information
    sario.save_slclist_to_h5(out_file=outfile, slc_date_list=slclist, alt_name="date")
    sario.save_ifglist_to_h5(out_file=outfile, ifg_date_list=ifglist)
    # Add the mean correlation of interferograms used in this network
    # Also copy over the metadata from the unw stack

    if cor_stack_file:
        logger.info(
            "Saving correlation from %s to %s/%s",
            cor_stack_file,
            outfile,
            cor_dset,
        )
        # cor_mean = ts_utils.get_cor_mean(defo_fname=outfile, cor_fname=cor_stack_file)
        cor_mean = get_cor_mean(
            valid_ifg_idxs,
            cor_fname=cor_stack_file,
            cor_dset=sario.STACK_DSET,
        )
        with h5py.File(outfile, "a") as hf:
            if cor_dset not in hf:
                hf[cor_dset] = cor_mean
                for k, v in attrs_to_copy.items():
                    hf[output_dset].attrs[k] = v
    else:
        cor_mean, cor_dset = None, None

    if los_file:
        los_dset = constants.LOS_ENU_DSET
        logger.info("Saving line of sight to %s/%s", outfile, los_dset)
        los_map = sario.load(los_file)
        with h5py.File(outfile, "a") as hf:
            if los_dset not in hf:
                hf[los_dset] = los_map
    else:
        los_map, los_dset = None, None

    if sario.attach_latlon:
        with h5py.File(unw_stack_file) as hf:
            lat = hf["lat"][()]
            lon = hf["lon"][()]
        if coordinates == "geo":
            sario.save_latlon_to_h5(outfile, lat=lat, lon=lon, overwrite=overwrite)
            sario.attach_latlon(outfile, output_dset, depth_dim="date")
            if cor_mean is not None:
                sario.attach_latlon(outfile, cor_dset)
            if los_map is not None:
                sario.attach_latlon(outfile, los_dset)
        else:
            sario.save_latlon_2d_to_h5(outfile, lat=lat, lon=lon, overwrite=overwrite)
            sario.attach_latlon_2d(outfile, output_dset, depth_dim="date")
            if cor_mean is not None:
                sario.attach_latlon_2d(outfile, cor_dset)

    # TODO: just use the h5?
    if save_as_netcdf:
        # TODO: any downside of the xr version instead of mine?
        outfile_nc = outfile.replace(".h5", ".nc")
        logger.info("Rewriting output to %s", outfile_nc)
        with xr.open_dataset(outfile) as ds:
            ds.to_netcdf(outfile_nc, engine="h5netcdf")
        # from apertools import netcdf
        # netcdf.hdf5_to_netcdf(
        #     outfile,
        #     dset_name=output_dset,
        #     stack_dim="date",
        #     data_units="cm",
        # )


def run_sbas(
    unw_stack_file,
    input_dset,
    valid_ifg_idxs,
    outfile,
    output_dset,
    block_shape,
    slclist,
    ifglist,
    # constant_velocity,
    alpha,
    phase_to_cm,
    # L1,
    # outlier_sigma=0,
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
        # print(nrows, ncols, block_shape)

    blk_slices = utils.block_iterator((nrows, ncols), block_shape[-2:], overlaps=(0, 0))
    # blk_slices = list(blk_slices)[:6]  # Test small area

    ExecutorClass = ProcessPoolExecutor if PARALLEL else DummyExecutor
    with ExecutorClass(max_workers=MAX_WORKERS) as executor:
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
                # constant_velocity,
                phase_to_cm,
            ): blk
            for blk in blk_slices
        }
        for future in as_completed(future_to_block):
            blk = future_to_block[future]
            out_chunk = future.result()
            rows, cols = blk
            write_out_chunk(out_chunk, outfile, output_dset, rows, cols)


def _load_and_run(
    blk,
    unw_stack_file,
    input_dset,
    valid_ifg_idxs,
    slclist,
    ifglist,
    # constant_velocity,
    phase_to_cm,
):
    rows, cols = blk
    with h5py.File(unw_stack_file) as hf:
        logger.info(f"Loading chunk {rows}, {cols}")
        unw_chunk = hf[input_dset][valid_ifg_idxs, rows[0] : rows[1], cols[0] : cols[1]]
        # TODO: get rid of nan pixels at edge! dont let it ruin the whole chunk
        out_chunk = _calc_soln(
            # out_chunk = _calc_soln_pixelwise(
            unw_chunk,
            slclist,
            ifglist,
            # alpha,
            # constant_velocity,
            phase_to_cm,
        )
        return out_chunk


def write_out_chunk(chunk, outfile, output_dset, rows=None, cols=None):
    rows = rows or [0, None]
    cols = cols or [0, None]
    logger.info(f"Writing out ({rows = }, {cols = }) chunk to {outfile}:/{output_dset}")
    with h5py.File(outfile, "r+") as hf:
        hf[output_dset][:, rows[0] : rows[1], cols[0] : cols[1]] = chunk


@jit_decorator
def _calc_soln(
    unw_chunk,
    slclist,
    ifglist,
    # alpha,
    # constant_velocity,
    # L1 = True,
    # outlier_sigma=4,
    phase_to_cm,
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
    stack *= phase_to_cm
    return stack


# @jit_decorator
@numba.njit(fastmath=True, parallel=True, cache=True, nogil=True)
def _calc_soln_pixelwise(
    unw_chunk,
    slclist,
    ifglist,
    # alpha,
    # constant_velocity,
    # L1 = True,
    # outlier_sigma=4,
    phase_to_cm,
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

    stack *= phase_to_cm
    return stack


def _get_block_shape(full_shape, chunk_size, block_size_max=10e6, nbytes=4):
    """Find a size of a data cube less than `block_size_max` in increments of `chunk_size`"""
    import copy

    if chunk_size is None:
        chunk_size = full_shape

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


def get_cor_mean(idxs, cor_fname=sario.COR_FILENAME, cor_dset=sario.STACK_DSET):
    """Get the mean correlation from a subset of images specified by `idxs`"""
    with h5py.File(cor_fname, "r") as hf:
        logger.info(f"Getting mean correlation from {len(idxs)} images")
        return np.mean(hf[cor_dset][idxs], axis=0)


@log_runtime
def calc_model_fit_deformation(
    defo_fname=constants.DEFO_FILENAME_NC,
    orig_dset=constants.DEFO_NOISY_DSET,
    degree=2,
    remove_day1_atmo=True,
    reweight_by_atmo_var=True,
    save_linear_fit=True,
    linear_velo_dset=constants.LINEAR_VELO_DSET,
    outname=None,
    overwrite=False,
):
    """Calculate a cumulative deformation by fitting a model to noisy timseries per-pixel

    Args:
        defo_fname (str): Name of the .nc file (default=`constants.DEFO_FILENAME_NC`)
        orig_dset (str): Name of dataset within `defo_fname` containing cumulative
            deformation+(atmospheric noise) timeseries (default=`constants.DEFO_NOISY_DSET`)
        degree (int): Polynomial degree to fit to each pixel's timeseries to model
            the deformation. This fit is removed to estimate the day 1 atmosphere
        remove_day1_atmo (bool): default True. Estimates and removes the first date's
            atmospheric phase screen. See `Notes` for details.
        reweight_by_atmo_var (bool): default True. Performs weighted least squares
            to refit model from residual variances. See `Notes` for details.
        outname (str): Name of dataset to save atmo estimattion within `defo_fname`
            (default= `constants.ATMO_DAY1_DSET`)
        overwrite (bool): If True, delete (if exists) the output

    Returns:
        avg_atmo (xr.DataArray): 2D array with the estiamted first day's atmospheric phase

    Notes:
    `remove_day1_atmo` gives the option to estimate atmospheric phase on the SAR first date.
    To find the first date's atmosphere, uses the (model-removed) daily phase timeseries,
    and recomputes the difference between each day and day1, then averages.
    Since the differences have been converted (through `run_inversion`) into phases
    on each date (consisting of (atmospheric delay + deformation)), we can just
    average each date's image after removing the linear trend.

    `reweight_by_atmo_var` first performs ordinary least squares to fit the model to
    the timeseries, then finds the residuals on each date. This will be mostly the
    atmospher noise on each date (though not perfect, as some deformation/other noises
    will be mixed in). Then the total variance of each date's image is used as the "sigma**2"
    value to perform weighted least squares.
    This will generally help ignore very noisy atmospheric days.

    """
    model_str = "polynomial_deg{}".format(degree)
    if outname is None:
        outname = constants.MODEL_DEFO_DSET.format(model=model_str)
        # polyfit_outname =
    _confirm_closed(defo_fname)

    if sario.check_dset(defo_fname, outname, overwrite) is False:  # already exists:
        with xr.open_dataset(defo_fname) as ds:
            # TODO: save the poly, also load that
            return ds[outname]

    with xr.open_dataset(defo_fname) as ds:
        noisy_da = ds[orig_dset]

        logger.info(
            "Fitting degree %s polynomial to %s/%s", degree, defo_fname, orig_dset
        )
        # Fit a polynomial along the "date" dimension (1 per pixel)
        polyfit_ds = noisy_da.polyfit("date", deg=degree)
        # This is the "modeled" deformation

        # Get expected ifg deformation phase from the polynomial velocity fit
        model_defo = xr.polyval(noisy_da.date, polyfit_ds.polyfit_coefficients)

        if remove_day1_atmo:
            logger.info("Compensating day1 atmosphere")
            # take difference of `linear_ifgs` and SBAS cumulative
            cum_detrend = model_defo - noisy_da
            # print(ptp_by_date_pct(cum_detrend, 0.02, 0.98)[:3])
            # Then reconstruct the ifgs containing the day 0
            reconstructed_ifgs = cum_detrend[1:] - cum_detrend[0]
            # print(reconstructed_ifgs.max(), reconstructed_ifgs.min(), reconstructed_ifgs.mean())
            # add -1 so that it has same sign as the timeseries, which
            # makes the compensation is (noisy_da - avg_atmo)
            avg_atmo = -1 * reconstructed_ifgs.mean(dim="date")
            avg_atmo.attrs["units"] = "cm"
            avg_atmo = avg_atmo.astype("float32")
            # print(avg_atmo.max(), avg_atmo.min(), avg_atmo.mean())

            # model_defo = model_defo - avg_atmo
            # # Still first the first day to 0
            # model_defo[0] = 0

        if reweight_by_atmo_var:
            logger.info("Refitting polynomial model using variances as weights")

            resids = model_defo - noisy_da
            # print(ptp_by_date_pct(resids, 0.02, 0.98)[:3])
            # polyfit wants to have the std dev. of variances, if known
            # atmo_stddevs = resids.std(dim=("lat", "lon"))
            # weights = 1 / atmo_stddevs
            # To more heavily beat down the noisy days, square these values
            # weights = (1 / atmo_stddevs) ** 2
            # atmo_ptps = ptp_by_date(resids)
            # atmo_ptp_qt = ptp_by_date_pct(resids, 0.05, 0.95)
            atmo_ptp_qt = ptp_by_date_pct(resids, 0.02, 0.98)
            weights = 1 / atmo_ptp_qt
            # return atmo_stddevs, atmo_ptps, atmo_ptp_qt, atmo_ptp_qt2

            # print(np.min(weights), weights[0])
            # if remove_day1_atmo:  # Make sure the avg_atmo variable is defined
            # weights[0] = 1 / np.var(avg_atmo)
            # weights[0] = 1 / ptp_by_date_pct(avg_atmo)
            # weights[0] = 1
            # else:
            # weights[0] = 1
            # Deep copy is done cuz of this: https://github.com/pydata/xarray/issues/5644
            polyfit_ds = (noisy_da.copy(True)).polyfit(
                "date",
                deg=degree,
                w=weights,
                # cov="unscaled",
                cov=True,
            )
            model_defo = xr.polyval(noisy_da.date, polyfit_ds.polyfit_coefficients)

        if remove_day1_atmo:
            logger.info("Compensating day1 atmosphere")
            # # take difference of `linear_ifgs` and SBAS cumulative
            # cum_detrend = model_defo - noisy_da
            # # Then reconstruct the ifgs containing the day 0
            # reconstructed_ifgs = cum_detrend[1:] - cum_detrend[0]
            # print(reconstructed_ifgs.max(), reconstructed_ifgs.min(), reconstructed_ifgs.mean())
            # # add -1 so that it has same sign as the timeseries, which
            # # makes the compensation is (noisy_da - avg_atmo)
            # avg_atmo = -1 * reconstructed_ifgs.mean(dim="date")
            # avg_atmo.attrs["units"] = "cm"
            # avg_atmo = avg_atmo.astype("float32")
            # print(avg_atmo.max(), avg_atmo.min(), avg_atmo.mean())

            model_defo = model_defo - avg_atmo
            # Still first the first day to 0
            # model_defo[0] = 0

        if save_linear_fit:
            logger.info("Finding linear velocity estimate using deg 1 polynomial")
            if not reweight_by_atmo_var:
                weights = None
            nda = noisy_da.copy(True)
            # print(nda.date[-1], (nda.date[-1] - nda.date[0]).dt.days)
            polyfit_lin = nda.polyfit(
                "date",
                deg=1,
                w=weights,
                # cov=True
                cov="unscaled",
            )
            velocities_cm_per_ns = polyfit_lin["polyfit_coefficients"][-2]
            velocities = velocities_cm_per_ns * constants.NS_PER_YEAR
            velocities = velocities.drop_vars("degree").astype("float32")
            velocities.attrs["units"] = "cm per year"

            logger.info("Uncertainty results from linear poly fit:")
            sigma_velo_cm_ns = np.sqrt(polyfit_lin["polyfit_covariance"][0, 0])
            sigma_velo_cm_yr = float(sigma_velo_cm_ns) * constants.NS_PER_YEAR
            logger.info("%.2f cm / year", sigma_velo_cm_yr)

    model_defo.attrs["units"] = "cm"
    model_defo = model_defo.astype("float32")
    logger.info("Saving cumulative model-fit deformation to %s", outname)
    model_ds = model_defo.to_dataset(name=outname)

    _confirm_closed(defo_fname)
    model_ds.to_netcdf(defo_fname, mode="a", engine="h5netcdf")

    if remove_day1_atmo:
        out = constants.ATMO_DAY1_DSET
        logger.info("Saving day1 atmo estimation to %s", out)
        avg_atmo.to_dataset(name=out).to_netcdf(
            defo_fname,
            mode="a",
            engine="h5netcdf",
        )

    if save_linear_fit:
        # out = constants.ATMO_DAY1_DSET
        logger.info("Saving linear velocity fit to %s", linear_velo_dset)
        if sario.check_dset(defo_fname, linear_velo_dset, overwrite):
            velocities.to_dataset(name=linear_velo_dset).to_netcdf(
                defo_fname,
                mode="a",
                engine="h5netcdf",
            )

    group = "polyfit_results"
    logger.info("Saving polyfit results to %s:/%s", defo_fname, group)
    if sario.check_dset(defo_fname, group, overwrite):
        polyfit_ds.to_netcdf(defo_fname, group=group, mode="a", engine="h5netcdf")

    group = "polyfit_lin_results"
    logger.info("Saving polyfit results to %s:/%s", defo_fname, group)
    if sario.check_dset(defo_fname, group, overwrite):
        polyfit_lin.to_netcdf(defo_fname, group=group, mode="a", engine="h5netcdf")

    return model_defo


def _record_run_params(paramfile, **kwargs):
    from ruamel.yaml import YAML

    print(kwargs)

    yaml = YAML()

    with open(paramfile, "w") as f:
        yaml.dump(kwargs, f)


def _confirm_closed(fname):
    """Weird hack to make sure file handles are closed
    https://github.com/h5py/h5py/issues/1090#issuecomment-608485873"""
    xr.open_dataset(fname, engine="h5netcdf").close()
    with h5py.File(fname, "r") as f:
        pass


@log_runtime
def lowess(
    defo_fname=constants.DEFO_FILENAME_NC,
    orig_dset=constants.DEFO_NOISY_DSET,
    out_fname=None,
    out_dset=constants.DEFO_LOWESS_DSET,
    frac=0.7,
    n_iter=2,
    overwrite=False,
):
    import apertools.lowess

    if not out_fname:
        out_fname = defo_fname

    _confirm_closed(defo_fname)
    if sario.check_dset(defo_fname, out_dset, overwrite) is False:  # already exists:
        with xr.open_dataset(defo_fname, engine="h5netcdf") as ds:
            return ds[out_dset]

    with xr.open_dataset(defo_fname, engine="h5netcdf") as ds:
        # ts = date2num(ds["date"].values)
        noisy_da = ds[orig_dset]
        # TODO: for when it's too big to run in memory
        # nstack, nrows, ncols = noisy_da.shape
        # nbytes = noisy_da.dtype.itemsize
        # chunk_size = noisy_da.chunks or [nstack, min(100, nrows), min(100, ncols)]
        # block_shape = _get_block_shape(
        # noisy_da.shape, chunk_size, block_size_max=100e6, nbytes=nbytes
        # )

        logger.info("Running lowess on %s/%s", defo_fname, orig_dset)
        # Run the "lowess" on each pixel separately
        x = date2num(noisy_da["date"].values)
        stack = noisy_da.values
    out_stack = apertools.lowess.lowess_stack(stack, x, frac=frac, n_iter=2)
    out_da = xr.DataArray(out_stack, coords=noisy_da.coords, dims=noisy_da.dims)

    # x = date2num(noisy_da['date'].values)
    # stack = noisy_da.values
    # blk_slices = utils.block_iterator((nrows, ncols), block_shape[-2:], overlaps=(0, 0))
    # out_stack = np.zeros_like(noisy_da.values)
    # for (rows, cols) in blk_slices:
    #     cur_block = noisy_da[:, rows, cols]
    #     cur_out = lowess.lowess_stack(noisy_da.values, x, frac=frac, n_iter=n_iter)
    #     out_stack[:, rows[0] : rows[1], cols[0] : cols[1]] = cur_out

    # out_da = apertools.lowess.lowess_xr( noisy_da, x_dset="date", frac=frac, n_iter=n_iter)
    # The first date will be a good estimate of that day's atmo
    day1_atmo = out_da.isel(date=0)
    out_da = out_da - day1_atmo

    out_ds = out_da.to_dataset(name=out_dset)
    atmo_ds = day1_atmo.to_dataset(name=constants.ATMO_DAY1_DSET)

    # save to a tmp file in case if fails due to dumb locking problems
    tmp_file = defo_fname.replace(".nc", ".tmp.nc")
    # Write first so that date gets overwritten for full date set
    atmo_ds.to_netcdf(
        tmp_file,
        mode="a",
        engine="h5netcdf",
    )
    out_ds.to_netcdf(tmp_file, engine="h5netcdf")

    # Now save the first day atmosphere/ full smoothed deformation to same file
    _confirm_closed(defo_fname)
    logger.info("Saving day1 atmo estimation to %s", constants.ATMO_DAY1_DSET)
    atmo_ds.to_netcdf(
        defo_fname,
        mode="a",
        engine="h5netcdf",
    )
    _confirm_closed(defo_fname)
    logger.info("Saving lowess-smoothed deformation to %s/%s", defo_fname, out_dset)
    out_ds.to_netcdf(defo_fname, mode="a", engine="h5netcdf")
    return out_da
