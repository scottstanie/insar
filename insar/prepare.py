"""prepare.py

Preprocessing insar data for timeseries analysis

Forms stacks as .h5 files for easy access to depth-wise slices
"""
from datetime import datetime
import os
import itertools
import h5py
import hdf5plugin
import numpy as np
from scipy.ndimage.morphology import binary_opening
import rasterio as rio

from apertools import sario, utils  # , latlon
import apertools.gps
from apertools.log import get_log, log_runtime
import apertools.deramp as deramp

from .constants import (
    MASK_FILENAME,
    UNW_FILENAME,
    # COR_FILENAME,
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
    # cc_stack_file = os.path.join(igram_path, COR_FILENAME)
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
    mask_fname=MASK_FILENAME,
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

    int_date_list = sario.save_intlist_to_h5(
        igram_path=directory,
        out_file=unw_stack_file,
        overwrite=overwrite,
        igram_ext=".unw",
    )

    # Only keep the SAR dates which have an interferogram where we're looking
    geo_date_list = list(sorted(set(itertools.chain.from_iterable(int_date_list))))

    _ = sario.save_geolist_to_h5(
        geo_date_list=geo_date_list,
        out_file=unw_stack_file,
        overwrite=overwrite,
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
    cur_chunk_size = 0
    for idx, in_fname in enumerate(file_list):
        if idx % 100 == 0:
            logger.info(f"Processing {in_fname} -> {idx+1} out of {len(file_list)}")

        if idx % chunk_depth == 0 and idx > 0:
            logger.info(f"Writing {lastidx}:{lastidx+chunk_depth}")
            assert cur_chunk_size <= chunk_depth
            with h5py.File(unw_stack_file, "r+") as f:
                f[dset_name][lastidx : lastidx + cur_chunk_size, :, :] = buf

            lastidx = idx
            cur_chunk_size = 0

        driver = "ROI_PAC" if in_fname.endswith(".unw") else None  # let gdal guess
        with rio.open(in_fname, driver=driver) as inf:
            mask = _read_mask_by_idx(idx, fname=mask_fname).astype(bool)
            # amp = inf.read(1)
            phase = inf.read(2)
            deramped_phase = deramp.remove_ramp(
                phase, deramp_order=deramp_order, mask=mask
            )

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
            cur_chunk_size += 1

            # sum for the stack, only use non-masked data
            stackavg[~mask] += deramped_phase[~mask] / temporal_baseline(in_fname)

    if cur_chunk_size > 0:
        # Write the final part of the buffer:
        with h5py.File(unw_stack_file, "r+") as f:
            f[dset_name][lastidx : lastidx + cur_chunk_size, :, :] = buf[
                :cur_chunk_size
            ]

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
def create_mask_stacks(
    igram_path,
    mask_filename=None,
    geo_path=None,
    overwrite=False,
    compute_from_geos=True,
):
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

    int_date_list = sario.save_intlist_to_h5(
        igram_path=igram_path,
        out_file=mask_file,
        overwrite=overwrite,
        igram_ext=".unw",
    )

    # Only keep the SAR dates which have an interferogram where we're looking
    geo_date_list = list(sorted(set(itertools.chain.from_iterable(int_date_list))))

    _ = sario.save_geolist_to_h5(
        geo_date_list=geo_date_list,
        out_file=mask_file,
        overwrite=overwrite,
    )

    all_geo_files = sario.find_geos(directory=geo_path, parse=False)
    all_geo_dates = sario.find_geos(directory=geo_path)
    geo_file_list = [
        gf for gf, gd in zip(all_geo_files, all_geo_dates) if gd in geo_date_list
    ]

    if compute_from_geos:
        save_geo_masks(
            geo_path,
            mask_file,
            dem_rsc=rsc_data,
            geo_file_list=geo_file_list,
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
        geo_date_list=geo_date_list,
        overwrite=overwrite,
        compute_from_geos=compute_from_geos,
    )
    # TODO: now add the correlation check
    return mask_file


def save_geo_masks(
    directory,
    mask_file=MASK_FILENAME,
    dem_rsc=None,
    dset_name=GEO_MASK_DSET,
    geo_file_list=None,
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
    if geo_file_list is None:
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
            if idx % 100 == 0:
                print(f"Done with {idx} out of {len(geo_file_list)}")

        # Also add a composite mask depthwise
        f[GEO_MASK_SUM_DSET] = np.sum(dset, axis=0)


def compute_int_masks(
    mask_file=None,
    igram_path=None,
    geo_path=None,
    row_looks=None,
    col_looks=None,
    dem_rsc=None,
    igram_ext=".unw",
    geo_date_list=None,
    dset_name=IGRAM_MASK_DSET,
    overwrite=False,
    compute_from_geos=True,  # TODO: combine these
    mask_dem=True,
    dem_filename="elevation_looked.dem",
):
    """Creates igram masks by taking the logical-or of the two .geo files

    Assumes save_geo_masks already run
    """
    if not sario.check_dset(mask_file, dset_name, overwrite):
        return
    if not sario.check_dset(mask_file, IGRAM_MASK_SUM_DSET, overwrite):
        return

    int_date_list = sario.find_igrams(directory=igram_path, ext=igram_ext)
    int_file_list = sario.find_igrams(directory=igram_path, ext=igram_ext, parse=False)

    if geo_date_list is None:
        geo_date_list = sario.find_geos(directory=geo_path)

    # Make the empty stack, or delete if exists
    shape = _find_file_shape(dem_rsc=dem_rsc, file_list=int_file_list)
    create_dset(mask_file, dset_name, shape=shape, dtype=bool)

    if mask_dem:
        dem_mask = sario.load(dem_filename) == 0

    with h5py.File(mask_file, "a") as f:
        geo_mask_stack = f[GEO_MASK_DSET]
        int_mask_dset = f[dset_name]
        for idx, (early, late) in enumerate(int_date_list):
            if compute_from_geos:
                early_idx = geo_date_list.index(early)
                late_idx = geo_date_list.index(late)
                early_mask = geo_mask_stack[early_idx]
                late_mask = geo_mask_stack[late_idx]
                int_mask_dset[idx] = np.logical_or(early_mask, late_mask)
            elif mask_dem:
                int_mask_dset[idx] = dem_mask
            else:
                print("Not masking")
                # int_mask_dset[idx] = np.ma.make_mask(dem_mask, shrink=False)

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
        m = f[dset][idx, :, :]
    # if fname.endswith(".nc"):  #
    # return m[::-1, :]
    # else:
    return m


# TODO: for mask subsetting...
# sario.save("dem.rsc", latlon.from_grid(mds.lon.values, mds.lat.values, sparse=True))
# time gdal_translate -of netCDF -co "FORMAT=NC4" -co "COMPRESS=DEFLATE" -co "ZLEVEL=4"
# -projwin -103.85 31.6 -102.8 30.8 masks_igram4.nc masks_subset4.nc
# sario.hdf5_to_netcdf("../igrams_looked_18/masks.h5", stack_dset_list=['igram', 'geo'],
# stack_dim_list=['idx', 'date'], outname="masks_igram3.nc")
# sario.save_geolist_to_h5(out_file="masks_subset4.nc",
# geo_date_list=sario.load_geolist_from_h5("../igrams_looked_18/masks.h5"))
# sario.save_intlist_to_h5(out_file="masks_subset5.nc",
# int_date_list=sario.load_intlist_from_h5("../igrams_looked_18/masks.h5"))