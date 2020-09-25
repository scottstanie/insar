"""prepare.py

Preprocessing insar data for timeseries analysis

Forms stacks as .h5 files for easy access to depth-wise slices
"""
from datetime import datetime
import h5py
import hdf5plugin
import os
import numpy as np
from scipy.ndimage.morphology import binary_opening
import rasterio as rio

from apertools import sario, utils  # , latlon
import apertools.gps
from apertools.log import get_log, log_runtime

from .constants import (
    MASK_FILENAME,
    UNW_FILENAME,
    # CC_FILENAME,
    STACK_FLAT_SHIFTED_DSET,
    GEO_MASK_DSET,
    GEO_MASK_SUM_DSET,
    IGRAM_MASK_DSET,
    IGRAM_MASK_SUM_DSET,
    DEM_RSC_DSET,
)

logger = get_log()


@log_runtime
def prepare_stacks(
    igram_path,
    ref_row=None,
    ref_col=None,
    ref_station=None,
    deramp_order=2,
    window=5,
    overwrite=False,
):
    # cc_stack_file = os.path.join(igram_path, CC_FILENAME)
    # mask_stack_file = os.path.join(igram_path, MASK_FILENAME)
    unw_stack_file = os.path.join(igram_path, UNW_FILENAME)

    create_mask_stacks(igram_path, overwrite=overwrite)

    if ref_station is not None:
        rsc_data = sario.load(os.path.join(igram_path, "dem.rsc"))
        ref_row, ref_col = apertools.gps.station_rowcol(
            station_name=ref_station,
            rsc_data=rsc_data,
        )
    if ref_row is None or ref_col is None:
        # ref_row, ref_col, ref_station = find_reference_location(
        raise ValueError("Need ref_row, ref_col or ref_station")

    deramp_and_shift_unws(
        ref_row,
        ref_col,
        unw_stack_file=unw_stack_file,
        dset_name=STACK_FLAT_SHIFTED_DSET,
        directory=igram_path,
        deramp_order=deramp_order,
        window=window,
        overwrite=overwrite,
    )
    # Now record attrs of the dataset
    with h5py.File(unw_stack_file, "r+") as f:
        f[STACK_FLAT_SHIFTED_DSET].attrs["deramp_order"] = deramp_order
        f[STACK_FLAT_SHIFTED_DSET].attrs["reference"] = [ref_row, ref_col]
        if ref_station is not None:
            f[STACK_FLAT_SHIFTED_DSET].attrs["reference_station"] = ref_station


def create_dset(h5file, dset_name, shape, dtype, chunks=True, compress=True):
    comp_dict = hdf5plugin.Blosc() if compress else dict()
    with h5py.File(h5file, "a") as f:
        f.create_dataset(
            dset_name, shape=shape, dtype=dtype, chunks=chunks, **comp_dict
        )


def temporal_baseline(filename):
    fmt = "%Y%m%d"
    fname = os.path.split(filename)[1]
    datestrs = os.path.splitext(fname)[0].split("_")
    igram = [datetime.strptime(t, fmt) for t in datestrs]
    return (igram[1] - igram[0]).days


@log_runtime
def deramp_and_shift_unws(
    ref_row,
    ref_col,
    unw_stack_file=UNW_FILENAME,
    dset_name=STACK_FLAT_SHIFTED_DSET,
    directory=".",
    deramp_order=2,
    window=5,
    overwrite=False,
    stack_fname="stackavg.tif",
):

    if not sario.check_dset(unw_stack_file, dset_name, overwrite):
        return
    logger.info(f"Deramping with reference ({ref_row},{ref_col})")
    # First make the empty dataset and save aux info
    in_ext = ".unw"
    file_list = sario.find_files(directory=directory, search_term="*" + in_ext)
    band = 2

    with rio.open(file_list[0]) as src:
        rows, cols = src.shape
        # bshape = src.block_shapes[band-1]  # TODO: use?
        dtype = src.dtypes[band - 1]

    shape = (len(file_list), rows, cols)
    create_dset(unw_stack_file, dset_name, shape, dtype, chunks=True, compress=True)

    # Save the extra files too
    rsc_data = sario.load(os.path.join(directory, "dem.rsc"))
    sario.save_dem_to_h5(
        unw_stack_file, rsc_data, dset_name=DEM_RSC_DSET, overwrite=overwrite
    )
    sario.save_geolist_to_h5(
        igram_path=directory, out_file=unw_stack_file, overwrite=overwrite
    )
    sario.save_intlist_to_h5(
        igram_path=directory, out_file=unw_stack_file, overwrite=overwrite
    )

    with h5py.File(unw_stack_file, "r+") as f:
        chunk_shape = f[dset_name].chunks
        chunk_depth, chunk_rows, chunk_cols = chunk_shape
        # n = n or chunk_size[0]

    # While we're iterating, save a stacked average
    stackavg = np.zeros((rows, cols), dtype="float32")

    buf = np.empty((chunk_depth, rows, cols), dtype=dtype)
    win = window // 2
    lastidx = 0
    for idx, in_fname in enumerate(file_list):
        if idx % 100 == 0:
            logger.info(f"Processing {in_fname} -> {idx+1} out of {len(file_list)}")

        if idx % chunk_depth == 0 and idx > 0:
            logger.info(f"Writing {lastidx}:{lastidx+chunk_depth}")
            with h5py.File(unw_stack_file, "r+") as f:
                f[dset_name][lastidx : lastidx + chunk_depth, :, :] = buf

            lastidx = idx

        with rio.open(in_fname, driver="ROI_PAC") as inf:
            mask = _read_mask_by_idx(idx)
            # amp = inf.read(1)
            phase = inf.read(2)
            deramped_phase = remove_ramp(phase, deramp_order=deramp_order, mask=mask)

            # Now center it on the shift window
            patch = deramped_phase[
                ref_row - win : ref_row + win + 1, ref_col - win : ref_col + win + 1
            ]
            if not np.all(np.isnan(patch)):
                deramped_phase -= np.nanmean(patch)
            else:
                # Do I actually just want to ignore this one and give 0s?
                logger.debug(f"Patch is all nan for {ref_row},{ref_col}")
                deramped_phase -= np.nanmean(deramped_phase)

            # now store this in the buffer until emptied
            curidx = idx % chunk_depth
            buf[curidx, :, :] = deramped_phase

            # sum for the stack, only use non-masked data
            stackavg[~mask] += deramped_phase[~mask] / temporal_baseline(in_fname)

    # Get the projection information to use to write as gtiff
    with rio.open(file_list[0], driver="ROI_PAC") as ds:
        transform = ds.transform
        crs = ds.crs

    with rio.open(
        stack_fname,
        "w",
        crs=crs,
        transform=transform,
        driver="GTiff",
        height=stackavg.shape[0],
        width=stackavg.shape[1],
        count=1,
        nodata=0,
        dtype=stackavg.dtype,
    ) as dst:
        dst.write(stackavg, 1)


@log_runtime
def create_mask_stacks(igram_path, mask_filename=None, geo_path=None, overwrite=False):
    """Create mask stacks for areas in .geo and .int

    Uses .geo dead areas as well as correlation
    """
    if mask_filename is None:
        mask_file = os.path.join(igram_path, MASK_FILENAME)

    if geo_path is None:
        geo_path = utils.get_parent_dir(igram_path)

    # Used to shrink the .geo masks to save size as .int masks
    row_looks, col_looks = apertools.sario.find_looks_taken(
        igram_path, geo_path=geo_path
    )

    rsc_data = sario.load(sario.find_rsc_file(os.path.join(igram_path, "dem.rsc")))
    sario.save_dem_to_h5(
        mask_file, rsc_data, dset_name=DEM_RSC_DSET, overwrite=overwrite
    )
    sario.save_geolist_to_h5(
        igram_path=igram_path, out_file=mask_file, overwrite=overwrite
    )
    sario.save_intlist_to_h5(
        igram_path=igram_path, out_file=mask_file, overwrite=overwrite
    )

    save_geo_masks(
        geo_path,
        mask_file,
        dem_rsc=rsc_data,
        row_looks=row_looks,
        col_looks=col_looks,
        overwrite=overwrite,
    )

    compute_int_masks(
        mask_file=mask_file,
        igram_path=igram_path,
        geo_path=geo_path,
        row_looks=row_looks,
        col_looks=col_looks,
        dem_rsc=rsc_data,
        overwrite=overwrite,
    )
    # TODO: now add the correlation check
    return mask_file


def save_geo_masks(
    directory,
    mask_file=MASK_FILENAME,
    dem_rsc=None,
    dset_name=GEO_MASK_DSET,
    row_looks=1,
    col_looks=1,
    overwrite=False,
):
    """Creates .mask files for geos where zeros occur

    Makes look arguments are to create arrays the same size as the igrams
    Args:
        overwrite (bool): erase the dataset from the file if it exists and recreate
    """

    def _get_geo_mask(geo_arr):
        # Uses for removing single mask pixels from nearest neighbor resample
        m = binary_opening(np.abs(geo_arr) == 0, structure=np.ones((3, 3)))
        return np.ma.make_mask(m, shrink=False)

    # Make the empty stack, or delete if exists
    if not sario.check_dset(mask_file, dset_name, overwrite):
        return
    if not sario.check_dset(mask_file, GEO_MASK_SUM_DSET, overwrite):
        return

    # rsc_geo = sario.load(sario.find_rsc_file(directory=directory))
    rsc_geo = sario.load(os.path.join(directory, "elevation.dem.rsc"))
    gshape = (rsc_geo["file_length"], rsc_geo["width"])
    geo_file_list = sario.find_files(directory=directory, search_term="*.geo")
    shape = _find_file_shape(
        dem_rsc=dem_rsc,
        file_list=geo_file_list,
        row_looks=row_looks,
        col_looks=col_looks,
    )

    create_dset(mask_file, dset_name, shape=shape, dtype=bool)

    with h5py.File(mask_file, "a") as f:
        dset = f[dset_name]
        for idx, geo_fname in enumerate(geo_file_list):
            # save as an individual file too
            mask_name = os.path.split(geo_fname)[1] + ".mask"
            if not os.path.exists(mask_name):
                # g = sario.load(geo_fname, looks=(row_looks, col_looks))
                gmap = np.memmap(
                    geo_fname,
                    dtype="complex64",
                    mode="r",
                    shape=gshape,
                )
                g_subsample = gmap[
                    (row_looks - 1) :: row_looks, (col_looks - 1) :: col_looks
                ]
                # ipdb.set_trace()
                logger.info(f"Saving {geo_fname} to stack")
                cur_mask = _get_geo_mask(g_subsample)
                sario.save(mask_name, cur_mask)
            else:
                cur_mask = sario.load(mask_name, rsc_file="dem.rsc")
            dset[idx] = cur_mask

        # Also add a composite mask depthwise
        f[GEO_MASK_SUM_DSET] = np.sum(dset, axis=0)

    # Now stack all these together


def compute_int_masks(
    mask_file=None,
    igram_path=None,
    geo_path=None,
    row_looks=None,
    col_looks=None,
    dem_rsc=None,
    dset_name=IGRAM_MASK_DSET,
    overwrite=False,
):
    """Creates igram masks by taking the logical-or of the two .geo files

    Assumes save_geo_masks already run
    """
    if not sario.check_dset(mask_file, dset_name, overwrite):
        return
    if not sario.check_dset(mask_file, IGRAM_MASK_SUM_DSET, overwrite):
        return

    int_date_list = sario.find_igrams(directory=igram_path)
    int_file_list = sario.find_igrams(directory=igram_path, parse=False)

    geo_date_list = sario.find_geos(directory=geo_path)

    # Make the empty stack, or delete if exists
    shape = _find_file_shape(dem_rsc=dem_rsc, file_list=int_file_list)
    create_dset(mask_file, dset_name, shape=shape, dtype=bool)

    with h5py.File(mask_file, "a") as f:
        geo_mask_stack = f[GEO_MASK_DSET]
        int_mask_dset = f[dset_name]
        for idx, (early, late) in enumerate(int_date_list):
            early_idx = geo_date_list.index(early)
            late_idx = geo_date_list.index(late)
            early_mask = geo_mask_stack[early_idx]
            late_mask = geo_mask_stack[late_idx]

            int_mask_dset[idx] = np.logical_or(early_mask, late_mask)

        # Also create one image of the total masks
        f[IGRAM_MASK_SUM_DSET] = np.sum(int_mask_dset, axis=0)


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


def _read_mask_by_idx(idx, fname="masks.h5", dset=IGRAM_MASK_DSET):
    with h5py.File(fname, "r") as f:
        return f[dset][idx, :, :]


def remove_ramp(z, deramp_order=1, mask=np.ma.nomask, copy=False):
    """Estimates a linear plane through data and subtracts to flatten

    Used to remove noise artifacts from unwrapped interferograms

    Args:
        z (ndarray): 2D array, interpreted as heights
        deramp_order (int): degree of surface estimation
            deramp_order = 1 removes linear ramp, deramp_order = 2 fits quadratic surface

    Returns:
        ndarray: flattened 2D array with estimated surface removed
    """
    z_masked = z.copy() if copy else z
    # Make a version of the image with nans in masked places
    z_masked[mask] = np.nan
    # Use this constrained version to find the plane fit
    z_fit = estimate_ramp(z_masked, deramp_order)
    # Then use the non-masked as return value
    return z - z_fit


def estimate_ramp(z, deramp_order):
    """Takes a 2D array an fits a linear plane to the data

    Ignores pixels that have nan values

    Args:
        z (ndarray): 2D array, interpreted as heights
        deramp_order (int): degree of surface estimation
            deramp_order = 1 removes linear ramp, deramp_order = 2 fits quadratic surface
        deramp_order (int)

    Returns:
        ndarray: the estimated coefficients of the surface
            For deramp_order = 1, it will be 3 numbers, a, b, c from
                 ax + by + c = z
            For deramp_order = 2, it will be 6:
                f + ax + by + cxy + dx^2 + ey^2
    """
    if deramp_order > 2:
        raise ValueError("Order only implemented for 1 and 2")
    # Note: rows == ys, cols are xs
    yidxs, xidxs = matrix_indices(z.shape, flatten=True)
    # c_ stacks 1D arrays as columns into a 2D array
    zflat = z.flatten()
    good_idxs = ~np.isnan(zflat)
    if deramp_order == 1:
        A = np.c_[np.ones(xidxs.shape), xidxs, yidxs]
        coeffs, _, _, _ = np.linalg.lstsq(A[good_idxs], zflat[good_idxs], rcond=None)
        # coeffs will be a, b, c in the equation z = ax + by + c
        c, a, b = coeffs
        # We want full blocks, as opposed to matrix_index flattened
        y_block, x_block = matrix_indices(z.shape, flatten=False)
        z_fit = a * x_block + b * y_block + c

    elif deramp_order == 2:
        A = np.c_[
            np.ones(xidxs.shape), xidxs, yidxs, xidxs * yidxs, xidxs ** 2, yidxs ** 2
        ]
        # coeffs will be 6 elements for the quadratic
        coeffs, _, _, _ = np.linalg.lstsq(A[good_idxs], zflat[good_idxs], rcond=None)
        yy, xx = matrix_indices(z.shape, flatten=True)
        idx_matrix = np.c_[np.ones(xx.shape), xx, yy, xx * yy, xx ** 2, yy ** 2]
        z_fit = np.dot(idx_matrix, coeffs).reshape(z.shape)

    return z_fit
