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

import h5py
try:
    import hdf5plugin  # noqa
except ImportError:
    blosc_id = 32001
    if not h5py.h5z.filter_avail(blosc_id):
        print("Failed to load hdf5plugin: may not save/load using blosc")

import numpy as np
import xarray as xr
from apertools import sario, utils
from apertools.constants import PHASE_TO_CM_MAP, WAVELENGTH_MAP
from apertools.log import get_log, log_runtime
from matplotlib.dates import date2num

from . import constants
from .prepare import create_dset, detect_rdr_coordinates
# from .ts_utils import get_cor_mean, ptp_by_date, ptp_by_date_pct
# from insar import ts_utils
from .ts_utils import DummyExecutor, ptp_by_date_pct

logger = get_log()

# Import numba if available; otherwise, just use python-only version
try:
    import numba

    from .ts_numba import build_A_matrix, build_B_matrix, integrate_velocities

    jit_decorator = numba.njit

except:
    logger.warning("Numba not avialable, falling back to python-only", exc_info=True)
    from .ts_utils import build_A_matrix, build_B_matrix, integrate_velocities

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
    cor_stack_dset=sario.STACK_DSET,
    los_file=constants.LOS_ENU_FILENAME,
    overwrite=False,
    min_date=None,
    max_date=None,
    # stack_average=False,
    # constant_velocity=False,
    max_temporal_baseline=None,
    max_temporal_bandwidth=None,  # TODO
    min_temporal_bandwidth=None,  # TODO
    include_annual=False,
    weight_by_cor=False,
    cor_thresh=0.15,
    # outlier_sigma=0,  # TODO: outlier outlier_sigma. Use trodi
    alpha=0,
    # L1=False, # TODO
    # difference=False,
    slclist_ignore_file="slclist_ignore.txt",
    save_as_netcdf=True,
    coordinates=None,  # geo, rdr
    platform="s1",
    max_workers=3,
    use_B_matrix=False,
    weight_by_temp_baseline=False,
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
        cor_thresh (float): threshold for setting all correlation values below to 0.
            Used since the correlation estimates are biased, meaning ~0 correlation
            can show up as 0.05-0.15 sometimes.
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
        full_shape, chunk_size, block_size_max=10e6, nbytes=nbytes
    )

    # TODO: platform = standardize(..)
    phase_to_cm = PHASE_TO_CM_MAP[platform]

    # if constant_velocity:
    # proc_func = proc_pixel_linear
    # output_shape = (nrows, ncols)
    # else:
    # proc_func = proc_pixel_daily
    output_shape = (len(slclist), nrows, ncols)
    logger.info("Target output shape: %s", output_shape)

    paramfile = (
        "{}_{}_run_params".format(outfile, output_dset).replace(".", "_") + ".yml"
    )
    # Saves all desried run variables and objects into a yaml file
    utils.record_params_as_yaml(
        paramfile,
        outfile=outfile,
        output_dset=output_dset,
        unw_stack_file=unw_stack_file,
        input_dset=input_dset,
        min_date=min_date,
        max_date=max_date,
        max_temporal_baseline=max_temporal_baseline,
        max_bandwidth=max_temporal_bandwidth,
        cor_thresh=cor_thresh,
        alpha=alpha,
        slclist_ignore=open(slclist_ignore_file).read().splitlines(),
        block_shape=block_shape,
        platform=platform,
        wavelength=WAVELENGTH_MAP[platform],
        coordinates=coordinates,
        max_workers=max_workers,
        use_B_matrix=use_B_matrix,
        weight_by_temp_baseline=weight_by_temp_baseline,
        **attrs_to_copy,
    )

    if sario.check_dset(outfile, output_dset, overwrite) is False:
        raise ValueError(f"{outfile}:/{output_dset} exists, {overwrite = }")

    create_dset(outfile, output_dset, output_shape, np.float32, chunks=True)
    create_dset(
        outfile, constants.COR_MEAN_DSET, output_shape[1:], np.float32, chunks=True
    )
    create_dset(
        outfile, constants.COR_STD_DSET, output_shape[1:], np.float32, chunks=True
    )
    create_dset(
        outfile, constants.TEMP_COH_DSET, output_shape[1:], np.float32, chunks=True
    )

    run_sbas(
        unw_stack_file,
        input_dset,
        valid_ifg_idxs,
        cor_stack_file,
        cor_stack_dset,
        outfile,
        output_dset,
        block_shape,
        date2num(slclist),
        date2num(ifglist),
        # constant_velocity,
        alpha,
        # L1,
        weight_by_cor,
        cor_thresh,
        # outlier_sigma,
        phase_to_cm,
        max_workers=max_workers,
        use_B_matrix=use_B_matrix,
        weight_by_temp_baseline=weight_by_temp_baseline,
    )

    # Now save the ifg/slc information
    sario.save_slclist_to_h5(out_file=outfile, slc_date_list=slclist, alt_name="date")
    sario.save_ifglist_to_h5(out_file=outfile, ifg_date_list=ifglist)
    # Add the mean correlation of interferograms used in this network
    # Also copy over the metadata from the unw stack

    # if cor_stack_file:
    #     logger.info(
    #         "Saving correlation from %s to %s/%s",
    #         cor_stack_file,
    #         outfile,
    #         cor_mean_dset,
    #     )
    #     # cor_mean = ts_utils.get_cor_mean(defo_fname=outfile, cor_fname=cor_stack_file)
    #     cor_mean = get_cor_mean(
    #         valid_ifg_idxs,
    #         cor_fname=cor_stack_file,
    #         cor_dset=cor_stack_dset,
    #     )
    with h5py.File(outfile, "a") as hf:
        # if cor_mean_dset not in hf:
        # hf[cor_mean_dset] = cor_mean
        for k, v in attrs_to_copy.items():
            hf[output_dset].attrs[k] = v
    # else:
    # cor_mean, cor_mean_dset = None, None

    if los_file and os.path.exists(los_file):
        los_dset = constants.LOS_ENU_DSET
        logger.info("Saving line of sight to %s/%s", outfile, los_dset)
        los_map = sario.load(los_file)
        with h5py.File(outfile, "a") as hf:
            if los_dset not in hf:
                hf[los_dset] = los_map
    else:
        logger.info("Not saving line of sight file: %s", los_file)
        los_map, los_dset = None, None

    with h5py.File(unw_stack_file) as hf:
        lat = hf["lat"][()]
        lon = hf["lon"][()]
    if coordinates == "geo":
        sario.save_latlon_to_h5(outfile, lat=lat, lon=lon, overwrite=overwrite)
        sario.attach_latlon(outfile, output_dset, depth_dim="date")
        # if cor_mean is not None:
        sario.attach_latlon(outfile, constants.COR_MEAN_DSET)
        sario.attach_latlon(outfile, constants.COR_STD_DSET)
        sario.attach_latlon(outfile, constants.TEMP_COH_DSET)
        if los_map is not None:
            sario.attach_latlon(outfile, los_dset)
    else:
        sario.save_latlon_2d_to_h5(outfile, lat=lat, lon=lon, overwrite=overwrite)
        sario.attach_latlon_2d(outfile, output_dset, depth_dim="date")
        # if cor_mean is not None:
        sario.attach_latlon_2d(outfile, constants.COR_MEAN_DSET)
        sario.attach_latlon_2d(outfile, constants.COR_STD_DSET)
        sario.attach_latlon_2d(outfile, constants.TEMP_COH_DSET)

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
    cor_stack_file,
    cor_stack_dset,
    outfile,
    output_dset,
    block_shape,
    slclist,
    ifglist,
    alpha,
    weight_by_cor,
    cor_thresh,
    phase_to_cm,
    max_workers,
    use_B_matrix=False,
    weight_by_temp_baseline=False,
):
    """Performs and SBAS inversion on each pixel of unw_stack to find deformation

    Returns:
        ndarray: solution velocity arrary
    """

    if alpha < 0:
        raise ValueError("alpha cannot be negative")

    with h5py.File(unw_stack_file) as hf:
        nstack, nrows, ncols = hf[input_dset].shape
        # print(nrows, ncols, block_shape)

    blk_slices = utils.block_slices((nrows, ncols), block_shape[-2:], overlaps=(0, 0))
    # TESTING: small area
    # blk_slices = list(blk_slices)[:25]

    if weight_by_temp_baseline:
        weights = _temp_baseline_weights(ifglist)
    else:
        weights = np.ones(len(ifglist))

    ExecutorClass = ProcessPoolExecutor if max_workers > 1 else DummyExecutor
    with ExecutorClass(max_workers=max_workers) as executor:
        # for (rows, cols) in blk_slices:
        future_to_block = {
            executor.submit(
                _load_and_run,
                blk,
                unw_stack_file,
                input_dset,
                valid_ifg_idxs,
                cor_stack_file,
                cor_stack_dset,
                slclist,
                ifglist,
                # constant_velocity,
                weight_by_cor,
                cor_thresh,
                phase_to_cm,
                use_B_matrix=use_B_matrix,
                weights=weights,
                # weight_by_temp_baseline=weight_by_temp_baseline,
            ): blk
            for blk in blk_slices
        }
        for future in as_completed(future_to_block):
            blk = future_to_block[future]
            out_chunk, cor_mean, cor_std, temp_coh = future.result()
            rows, cols = blk
            write_out_chunk(out_chunk, outfile, output_dset, rows, cols)
            write_out_chunk(
                cor_mean, outfile, constants.COR_MEAN_DSET, rows, cols, verbose=False
            )
            write_out_chunk(
                cor_std, outfile, constants.COR_STD_DSET, rows, cols, verbose=False
            )
            write_out_chunk(
                temp_coh, outfile, constants.TEMP_COH_DSET, rows, cols, verbose=False
            )


def _load_and_run(
    blk,
    unw_stack_file,
    input_dset,
    valid_ifg_idxs,
    cor_stack_file,
    cor_stack_dset,
    slclist,
    ifglist,
    # constant_velocity,
    weight_by_cor,
    cor_thresh,
    phase_to_cm,
    use_B_matrix=False,
    weights=None,
    # weight_by_temp_baseline=False,
):
    rows, cols = blk
    with h5py.File(unw_stack_file) as hf, h5py.File(cor_stack_file) as hf_c:
        logger.info(f"Loading chunk {rows}, {cols}")
        unw_chunk = hf[input_dset][valid_ifg_idxs, rows[0] : rows[1], cols[0] : cols[1]]
        cor_chunk = hf_c[cor_stack_dset][
            valid_ifg_idxs, rows[0] : rows[1], cols[0] : cols[1]
        ]

        if weight_by_cor and cor_thresh is not None:
            zero_idxs = cor_chunk < cor_thresh
            cor_chunk[zero_idxs] = 0.0
            unw_chunk[zero_idxs] = 0.0
            # Only count nonzero- we'll ignore those
            cor_mean = cor_chunk[~zero_idxs].mean(axis=0)
            cor_std = cor_chunk[~zero_idxs].std(axis=0)
        else:
            cor_mean = np.average(cor_chunk, weights=weights, axis=0)
            cor_std = cor_chunk.std(axis=0)

        if weight_by_cor:
            out_chunk = _calc_soln_cor_weighted(
                unw_chunk,
                cor_chunk,
                slclist,
                ifglist,
                phase_to_cm,
            )
        else:
            out_chunk, temp_coh = _calc_soln(
                # out_chunk = _calc_soln_pixelwise(
                unw_chunk,
                slclist,
                ifglist,
                # alpha,
                # constant_velocity,
                phase_to_cm,
                weights=weights,
                use_B_matrix=use_B_matrix,
                # weight_by_temp_baseline=weight_by_temp_baseline,
            )
        return out_chunk, cor_mean, cor_std, temp_coh


def write_out_chunk(chunk, outfile, output_dset, rows=None, cols=None, verbose=True):
    rows = rows or [0, None]
    cols = cols or [0, None]
    if verbose:
        logger.info(
            f"Writing out ({rows = }, {cols = }) chunk to {outfile}:/{output_dset}"
        )
    with h5py.File(outfile, "r+") as hf:
        dset = hf[output_dset]
        if dset.ndim == 3:
            dset[:, rows[0] : rows[1], cols[0] : cols[1]] = chunk
        else:
            dset[rows[0] : rows[1], cols[0] : cols[1]] = chunk


def _temp_baseline_weights(ifglist):
    temp_baselines = np.array([ifg[1] - ifg[0] for ifg in ifglist])
    # Use square root of temp baseline as weighting,
    # assuming that the variance of each ifg is proportional to the baseline
    weights = 1 / np.sqrt(temp_baselines)
    # make closer to 1, so that the weights are not too small (numerical issues)
    weights /= np.max(weights)
    return weights


@jit_decorator
def _calc_soln(
    unw_chunk,
    slclist,
    ifglist,
    # alpha,
    # constant_velocity,
    # outlier_sigma=4,
    phase_to_cm,
    use_B_matrix=False,
    # L1=False,
    weights=None,
    # weight_by_temp_baseline=False,
):
    dtype = unw_chunk.dtype

    nstack, nrow, ncol = unw_chunk.shape
    unw_cols = unw_chunk.reshape((nstack, -1))
    nan_idxs = np.isnan(unw_cols)
    unw_cols_nonan = np.where(nan_idxs, 0, unw_cols).astype(dtype)
    # skip any all 0 blocks:
    if unw_cols_nonan.sum() == 0:
        out = np.zeros((len(slclist), nrow, ncol), dtype=dtype)
        temp_coh = np.zeros((nrow, ncol), dtype=dtype)
        return out, temp_coh

    if use_B_matrix:
        G = build_B_matrix(slclist, ifglist)
    else:
        G = build_A_matrix(slclist, ifglist)
    G = G.astype(dtype)

    if weights is not None:
        weights = weights.reshape((-1, 1))
        unw_cols_nonan *= weights
        G *= weights

    pG = np.linalg.pinv(G)
    # Each column will be one pixel's solution
    soln_cols = pG @ unw_cols_nonan
    residual_cols = unw_cols_nonan - G @ soln_cols

    if use_B_matrix:
        # Version with B (velo diffs)
        timediffs = np.diff(slclist)
        # zero for first date added in integration
        phi_cols = integrate_velocities(soln_cols, timediffs).astype(dtype)
    else:
        # Add a 0 image for the first date
        phi_cols = np.vstack(
            (np.zeros((1, soln_cols.shape[1]), dtype=dtype), soln_cols)
        )
        phi_cols = phi_cols.astype(dtype)

    stack = phi_cols.reshape((-1, nrow, ncol))

    # Compute the temporal coherence as one solution quality metric
    temp_coh_cols = (
        np.abs(np.sum(np.exp(1j * residual_cols), axis=0)) / residual_cols.shape[0]
    )
    temp_coh_cols[np.sum(unw_cols_nonan, axis=0) == 0] = 0.0
    temp_coh = temp_coh_cols.reshape((nrow, ncol)).astype(dtype)

    stack *= phase_to_cm

    return stack, temp_coh


# @jit_decorator
def _calc_soln_cor_weighted(
    unw_chunk,
    cor_chunk,
    slclist,
    ifglist,
    phase_to_cm,
):
    dtype = unw_chunk.dtype

    nifg, nrow, ncol = unw_chunk.shape
    npixels = nrow * ncol
    unw_cols = unw_chunk.reshape((nifg, npixels))  # Shape: (nifg, npixels)
    nan_idxs = np.isnan(unw_cols)
    unw_cols_nonan = np.where(nan_idxs, 0, unw_cols).astype(dtype)
    # skip any all 0 blocks:
    if unw_cols_nonan.sum() == 0:
        return np.zeros((len(slclist), nrow, ncol), dtype=dtype)

    # To solve all, with different A matrix for each pixel:
    # a.shape: npixels, nifg, nsar (aka nsar = ndates)
    # b.shape: npixels, nifg
    # In [116]: a.shape, bb.shape
    # Out[116]: ((4, 3, 2), (4, 3))
    # pa = np.linalg.pinv(a)
    # In [119]: np.squeeze(pa @ bb[:, :, None]).shape
    # Out[119]: (4, 2)

    # igram_count = len(unw_chunk)
    A = build_A_matrix(slclist, ifglist)  # shape: (nifg, nsar)
    nsar = A.shape[1]
    # Weight by correlation for each pixel
    cor_cols = cor_chunk.reshape((nifg, npixels))  # shape: (nifg, npixels)
    # Force any nans to 0
    nan_idxs = np.isnan(cor_cols)
    cor_cols = np.where(nan_idxs, 0, cor_cols).astype(dtype)
    # # Note that we need to take the square root for the squared residuals to have
    # # weights equal to the correlation.
    # cor_cols = np.sqrt(cor_cols)

    # TODO
    # %time np.linalg.pinv(np.transpose(A, axes=(0, 2, 1)) @ A)
    # CPU times: user 12min 53s, sys: 18min 33s, total: 31min 27s
    # Wall time: 31.9 s

    # weight the b vector by the correlation
    b_weighted = unw_cols_nonan * cor_cols
    # Need the transpose of the b vector before reshaping
    b2 = b_weighted.T.reshape(npixels, nifg, 1)
    # b2 = np.ascontiguousarray(b_weighted.T.reshape(npixels, nifg, 1))

    # Version with the A matrix
    # Need the 3D A matrix to be (npixels, nifg, nsar)
    # A = A[None, :, :]  # shape: (1, nifg, nsar)
    # cor_cols.T[:, :, None].shape: (npixels, nifg, 1)
    # A_weighted = A[None, :, :] * cor_cols.T[:, :, None]
    A_weighted = A.reshape((1, nifg, nsar)) * cor_cols.T.reshape((npixels, nifg, 1))
    # A_weighted = np.ascontiguousarray(A_weighted)
    # pA = np.linalg.pinv(A_weighted).astype(dtype)  # shape: (npixels, nsar, nifg)
    AT = np.transpose(A_weighted, axes=(0, 2, 1))
    # AT = np.ascontiguousarray(np.transpose(A_weighted, axes=(0, 2, 1)))
    AtA = AT @ A_weighted
    Atb = AT @ b2
    # print("inverting with pinv")
    pAtA = np.linalg.pinv(AtA)
    # print(pAtA.shape)
    # b_weighted[:, :, None].shape: (npixels, nifg, 1)
    # soln_cols = np.squeeze(pA @ b_weighted[:, :, None])
    soln_cols = np.squeeze(pAtA @ Atb)
    # Need to transpose the solution before reshaping
    stack = soln_cols.T.reshape((-1, nrow, ncol)).astype(dtype)
    # Add a 0 image for the first date
    stack = np.concatenate((np.zeros((1, nrow, ncol), dtype=dtype), stack), axis=0)
    stack *= phase_to_cm

    # # Version with B (velo diffs)
    # B = build_B_matrix(slclist, ifglist)
    # B_weighted = B.reshape((1, nifg, nsar)) * cor_cols.T.reshape((npixels, nifg, 1))
    # BT = np.transpose(B_weighted, axes=(0, 2, 1))
    # BtB = BT @ B_weighted
    # timediffs = np.diff(slclist)
    # Btb = BT @ b2
    # pBtB = np.linalg.pinv(BtB)
    # soln_cols = np.squeeze(pBtB @ Btb)
    # phi_soln = integrate_velocities(soln_cols.T, timediffs)
    # stack = phi_soln.reshape((-1, nrow, ncol))
    # stack *= phase_to_cm

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
    """Find size of data cube less than `block_size_max` in increments of `chunk_size`"""
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
    min_days_weighted=2. * 365.25,
    frac=None,
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
        # Convert days to days, using matplotlib's functions
        x = date2num(noisy_da["date"].values)
        stack = noisy_da.values

    # Run the "lowess" on each pixel separately
    out_stack = apertools.lowess.lowess_stack(
        stack, x, frac=frac, min_x_weighted=min_days_weighted, n_iter=n_iter
    )
    out_da = xr.DataArray(out_stack, coords=noisy_da.coords, dims=noisy_da.dims)

    # x = date2num(noisy_da['date'].values)
    # stack = noisy_da.values
    # blk_slices = utils.block_slices((nrows, ncols), block_shape[-2:], overlaps=(0, 0))
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

    # # save to a tmp file in case if fails due to dumb locking problems
    # tmp_file = defo_fname.replace(".nc", ".tmp.nc")
    # # Write first so that date gets overwritten for full date set
    # atmo_ds.to_netcdf(
    #     tmp_file,
    #     mode="a",
    #     engine="h5netcdf",
    # )
    # out_ds.to_netcdf(tmp_file, engine="h5netcdf")

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
