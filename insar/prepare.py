"""prepare.py

Preprocessing insar data for timeseries analysis

Forms stacks as .h5 files for easy access to depth-wise slices
"""
import os
import glob
import itertools
import h5py
from tqdm import tqdm

try:
    import hdf5plugin  # noqa
except ImportError:
    print("Failed to load hdf5plugin: may not save/load using blosc")
import numpy as np
from scipy.ndimage.morphology import binary_opening
import rasterio as rio

from apertools import sario, utils  # , latlon
import apertools.gps
from apertools.log import get_log, log_runtime
import apertools.deramp as deramp

from apertools.sario import (
    MASK_FILENAME,
    UNW_FILENAME,
    COR_FILENAME,
    IFG_FILENAME,
    STACK_DSET,
    STACK_MEAN_DSET,
    STACK_FLAT_SHIFTED_DSET,
    SLC_MASK_DSET,
    SLC_MASK_SUM_DSET,
    IFG_MASK_DSET,
    IFG_MASK_SUM_DSET,
    ISCE_GEOM_DIR,
    ISCE_SLC_DIR,
)
from apertools.constants import COORDINATES_CHOICES

logger = get_log()


@log_runtime
def prepare_stacks(
    igram_path,
    unw_filename=UNW_FILENAME,
    cor_filename=COR_FILENAME,
    mask_filename=MASK_FILENAME,
    ref_row=None,
    ref_col=None,
    ref_lat=None,
    ref_lon=None,
    ref_station=None,
    deramp_order=2,
    window=5,
    search_term="*.unw",
    cor_search_term="*.cc",
    float_cor=False,
    row_looks=None,
    col_looks=None,
    overwrite=False,
    coordinates="geo",
    geom_dir=ISCE_GEOM_DIR,
    mask_from_slcs=True,
    compute_water_mask=False,
    mask_dem=True,
):
    import apertools.latlon
    if coordinates is None:
        coordinates = detect_rdr_coordinates(igram_path)
    if coordinates not in COORDINATES_CHOICES:
        raise ValueError("coordinates must be in {}".format(COORDINATES_CHOICES))

    ifg_date_list = sario.find_ifgs(directory=igram_path, search_term=search_term)
    ifg_file_list = sario.find_ifgs(
        directory=igram_path, search_term=search_term, parse=False
    )
    slc_date_list = list(sorted(set(itertools.chain.from_iterable(ifg_date_list))))

    mask_file = os.path.join(igram_path, mask_filename)
    cor_stack_file = os.path.join(igram_path, cor_filename)
    unw_stack_file = os.path.join(igram_path, unw_filename)
    for f in [mask_file, cor_stack_file, unw_stack_file]:
        sario.save_ifglist_to_h5(
            ifg_date_list=ifg_date_list, out_file=f, overwrite=overwrite
        )
        sario.save_slclist_to_h5(
            slc_date_list=slc_date_list, out_file=f, overwrite=overwrite
        )

    create_mask_stacks(
        igram_path,
        mask_file=mask_file,
        overwrite=overwrite,
        coordinates=coordinates,
        compute_water_mask=compute_water_mask,
        compute_from_slcs=mask_from_slcs,
        slc_date_list=slc_date_list,
        ifg_date_list=ifg_date_list,
        ifg_file_list=ifg_file_list,
        row_looks=row_looks,
        col_looks=col_looks,
        mask_dem=mask_dem,
        geom_dir=geom_dir,
    )

    create_cor_stack(
        cor_stack_file=cor_stack_file,
        dset_name=STACK_DSET,
        directory=igram_path,
        search_term=cor_search_term,
        overwrite=overwrite,
        coordinates=coordinates,
        float_cor=float_cor,
    )

    if ref_station is not None:
        rsc_data = sario.load(os.path.join(igram_path, "dem.rsc"))
        ref_row, ref_col = apertools.gps.station_rowcol(
            station_name=ref_station,
            rsc_data=rsc_data,
        )
    elif ref_lat is not None and ref_lon is not None:
        # TODO: getting this from radar coord?

        if coordinates == "geo":
            rsc_file = os.path.join(igram_path, "dem.rsc")
            if os.path.exists(rsc_file):
                rsc_data = sario.load(rsc_file)
                filename = None
            else:
                filename = mask_file
            ref_row, ref_col = apertools.latlon.latlon_to_rowcol(
                rsc_data=rsc_data,
                filename=filename,
            )
        else:
            ref_row, ref_col = apertools.latlon.latlon_to_rowcol_rdr(
                ref_lat, ref_lon, geom_dir=geom_dir
            )
            print(ref_row, ref_col)

    if ref_row is None or ref_col is None:
        # ref_row, ref_col, ref_station = find_reference_location(
        raise ValueError("Need ref_row, ref_col or ref_station")

    try:
        water_mask = sario.load(
            os.path.join(geom_dir, "waterMask.rdr"), use_gdal=True, band=1
        )
        water_mask = ~(water_mask.astype(bool))
    except:
        water_mask = None

    # Now create the unwrapped ifg stack
    deramp_and_shift_unws(
        ref_row,
        ref_col,
        unw_stack_file=unw_stack_file,
        dset_name=STACK_FLAT_SHIFTED_DSET,
        mask_file=mask_filename,
        directory=igram_path,
        deramp_order=deramp_order,
        window=window,
        search_term=search_term,
        coordinates=coordinates,
        water_mask=water_mask,
        overwrite=overwrite,
    )

    # Now record attrs of the dataset
    with h5py.File(unw_stack_file, "r+") as f:
        f[STACK_FLAT_SHIFTED_DSET].attrs["deramp_order"] = deramp_order
        f[STACK_FLAT_SHIFTED_DSET].attrs["reference"] = [ref_row, ref_col]
        f[STACK_FLAT_SHIFTED_DSET].attrs["reference_latlon"] = [ref_lat, ref_lon]
        f[STACK_FLAT_SHIFTED_DSET].attrs["reference_window"] = window
        if ref_station is not None:
            f[STACK_FLAT_SHIFTED_DSET].attrs["reference_station"] = ref_station


def create_dset(h5file, dset_name, shape, dtype, chunks=True, compress=True):
    # comp_dict = hdf5plugin.Blosc() if compress else dict()
    # comp_dict = dict(compression="gzip") if compress else dict()
    # TODO: gzip is super slow, but lfz and blosc can't be read by other stuff...
    # what to do
    comp_dict = dict()
    with h5py.File(h5file, "a") as f:
        f.create_dataset(
            dset_name, shape=shape, dtype=dtype, chunks=chunks, **comp_dict
        )


def temporal_baseline(filename):
    ifg = sario.parse_ifglist_strings(filename)
    return (ifg[1] - ifg[0]).days


@log_runtime
def deramp_and_shift_unws(
    ref_row,
    ref_col,
    unw_stack_file=UNW_FILENAME,
    mask_file=MASK_FILENAME,
    dset_name=STACK_FLAT_SHIFTED_DSET,
    directory=".",
    search_term="*.unw",
    deramp_order=2,
    window=5,
    overwrite=False,
    stack_fname="stackavg.tif",
    coordinates="geo",
    water_mask=None,
    geom_dir=ISCE_GEOM_DIR,
):

    if not sario.check_dset(unw_stack_file, dset_name, overwrite):
        return
    logger.info(f"Deramping with reference ({ref_row},{ref_col})")
    # First make the empty dataset and save aux info
    file_list = sario.find_ifgs(
        directory=directory, search_term=search_term, parse=False
    )
    band = 2

    with rio.open(file_list[0]) as src:
        rows, cols = src.shape
        dtype = src.dtypes[band - 1]

    shape = (len(file_list), rows, cols)
    create_dset(unw_stack_file, dset_name, shape, dtype, chunks=True, compress=True)

    with h5py.File(unw_stack_file, "r+") as f:
        chunk_shape = f[dset_name].chunks
        chunk_depth, chunk_rows, chunk_cols = chunk_shape
        # n = n or chunk_size[0]

    # Get the projection information to use to write as gtiff
    if os.path.exists(file_list[0] + ".rsc"):
        driver = "ROI_PAC"
    # elif os.path.exists(file_list[0] + ".xml") or os.path.exists(file_list[0] + ".vrt"):
    # driver = "ISCE"
    else:
        driver = None

    # While we're iterating, save a stacked average
    stack_sum = np.zeros((rows, cols), dtype="float32")
    stack_counts = np.zeros((rows, cols), dtype="float32")

    mask = np.zeros((rows, cols), dtype=bool)
    buf = np.empty((chunk_depth, rows, cols), dtype=dtype)
    win = window // 2
    lastidx = 0
    cur_chunk_size = 0
    for idx, in_fname in enumerate(tqdm(file_list)):
        if idx % chunk_depth == 0 and idx > 0:
            tqdm.write(f"Writing {lastidx}:{lastidx+chunk_depth}")
            assert cur_chunk_size <= chunk_depth
            with h5py.File(unw_stack_file, "r+") as f:
                f[dset_name][lastidx : lastidx + cur_chunk_size, :, :] = buf

            lastidx = idx
            cur_chunk_size = 0

        with rio.open(in_fname, driver=driver) as inf:
            # amp = inf.read(1)
            phase = inf.read(2)

            # TODO: this isn't well resolved for ISCE stuff
            if water_mask is not None:
                mask = water_mask
            elif mask_file:
                with h5py.File(mask_file, "r") as f:
                    mask = f[IFG_MASK_DSET][idx, :, :].astype(bool)
            else:
                mask = (mask * 0).astype(bool)

            deramped_phase = deramp.remove_ramp(
                phase, deramp_order=deramp_order, mask=mask
            )

            # Now center it on the shift window
            deramped_phase = _shift(deramped_phase, ref_row, ref_col, win)
            # now store this in the buffer until emptied
            curidx = idx % chunk_depth
            buf[curidx, :, :] = deramped_phase
            cur_chunk_size += 1

            # sum for the stack, only use non-masked data
            nan_mask = np.isnan(deramped_phase)
            stack_sum[~nan_mask] += deramped_phase[~nan_mask]
            stack_counts[~nan_mask] += 1

    if cur_chunk_size > 0:
        # Write the final part of the buffer:
        with h5py.File(unw_stack_file, "r+") as f:
            f[dset_name][lastidx : lastidx + cur_chunk_size, :, :] = buf[
                :cur_chunk_size
            ]

    # Save the stack average
    stackavg = stack_sum / (stack_counts + 1e-6)
    with rio.open(file_list[0], driver=driver) as ds:
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

    # Save the extra files too
    if coordinates == "geo":
        # TODO: how to not make it tied to ROI_PAC
        rsc_data = sario.load(file_list[0] + ".rsc")
        sario.save_latlon_to_h5(unw_stack_file, rsc_data=rsc_data)
        sario.attach_latlon(unw_stack_file, dset_name, depth_dim="ifg_idx")
    else:
        lat, lon = sario.load_rdr_latlon(geom_dir=geom_dir)
        sario.save_latlon_2d_to_h5(
            unw_stack_file, lat=lat, lon=lon, overwrite=overwrite
        )
        sario.attach_latlon_2d(unw_stack_file, dset_name, depth_dim="ifg_idx")


def _shift(deramped_phase, ref_row, ref_col, win):
    """Shift with 2d or 3d array to be zero in `win` around (ref_row, ref_col)"""
    patch = deramped_phase[
        ..., ref_row - win : ref_row + win + 1, ref_col - win : ref_col + win + 1
    ]
    return deramped_phase - np.nanmean(patch, axis=(-2, -1), keepdims=True)


def _get_load_func(in_fname, band=2):
    try:
        with rio.open(in_fname):
            pass

        def load_func(x):
            with rio.open(x) as inf:
                return inf.read(band)

    except rio.errors.RasterioIOError:

        def load_func(x):
            return sario.load(x)

    return load_func


@log_runtime
def create_cor_stack(
    cor_stack_file=COR_FILENAME,
    dset_name=STACK_DSET,
    directory=".",
    search_term="*.cc",
    overwrite=False,
    geom_dir=ISCE_GEOM_DIR,
    coordinates="geo",
    float_cor=False,
):

    if not sario.check_dset(cor_stack_file, dset_name, overwrite):
        return
    # First make the empty dataset and save aux info
    file_list = sario.find_ifgs(
        directory=directory, search_term=search_term, parse=False
    )
    logger.info(f"Creating {cor_stack_file} of {len(file_list)} correlation files")

    band = 1 if float_cor else 2
    load_func = _get_load_func(file_list[0], band=band)
    rows, cols = load_func(file_list[0]).shape
    dtype = np.float32

    shape = (len(file_list), rows, cols)
    create_dset(cor_stack_file, dset_name, shape, dtype, chunks=True, compress=True)

    with h5py.File(cor_stack_file, "r+") as f:
        chunk_shape = f[dset_name].chunks
        chunk_depth, chunk_rows, chunk_cols = chunk_shape
        # n = n or chunk_size[0]

    # While we're iterating, save a stacked average
    stack_sum = np.zeros((rows, cols), dtype="float32")
    stack_counts = np.zeros((rows, cols), dtype="float32")

    buf = np.empty((chunk_depth, rows, cols), dtype=dtype)
    lastidx = 0
    cur_chunk_size = 0
    for idx, in_fname in enumerate(tqdm(file_list)):
        if idx % chunk_depth == 0 and idx > 0:
            tqdm.write(f"Writing {lastidx}:{lastidx+chunk_depth}")
            assert cur_chunk_size <= chunk_depth
            with h5py.File(cor_stack_file, "r+") as f:
                f[dset_name][lastidx : lastidx + cur_chunk_size, :, :] = buf

            lastidx = idx
            cur_chunk_size = 0

        # driver = "ROI_PAC" if in_fname.endswith(".cor") else None  # let gdal guess
        # try
        cor = load_func(in_fname)
        # with rio.open(in_fname) as inf:
        # cor = inf.read(band)
        # store this in the buffer until emptied
        curidx = idx % chunk_depth
        buf[curidx, :, :] = cor
        cur_chunk_size += 1

        mask = np.isnan(cor)
        stack_sum[~mask] += cor[~mask]
        stack_counts[~mask] += 1

    if cur_chunk_size > 0:
        # Write the final part of the buffer:
        with h5py.File(cor_stack_file, "r+") as f:
            f[dset_name][lastidx : lastidx + cur_chunk_size, :, :] = buf[
                :cur_chunk_size
            ]
    # Also save the stack mean
    with h5py.File(cor_stack_file, "r+") as f:
        f[STACK_MEAN_DSET] = stack_sum / (stack_counts + 1e-6)

    # Save the lat/lon datasets, attach to main data files
    if coordinates == "geo":
        # TODO: best way to check for radar coords? wouldn't want any lat/lon...
        rsc_data = sario.load(os.path.join(directory, "dem.rsc"))
        sario.save_latlon_to_h5(cor_stack_file, rsc_data=rsc_data)
        sario.attach_latlon(cor_stack_file, dset_name, depth_dim="ifg_idx")
        sario.attach_latlon(cor_stack_file, STACK_MEAN_DSET, depth_dim=None)
    else:
        lat, lon = sario.load_rdr_latlon(geom_dir=geom_dir)
        sario.save_latlon_2d_to_h5(
            cor_stack_file, lat=lat, lon=lon, overwrite=overwrite
        )
        sario.attach_latlon_2d(cor_stack_file, dset_name, depth_dim="ifg_idx")
        sario.attach_latlon_2d(cor_stack_file, STACK_MEAN_DSET, depth_dim=None)


@log_runtime
def create_mask_stacks(
    igram_path,
    mask_file=MASK_FILENAME,
    slc_path=None,
    overwrite=False,
    compute_from_slcs=True,
    slc_date_list=None,
    ifg_date_list=None,
    ifg_file_list=None,
    coordinates="geo",
    compute_water_mask=False,
    mask_dem=True,
    row_looks=None,
    col_looks=None,
    geom_dir=ISCE_GEOM_DIR,
):
    """Create mask stacks for areas in .geo and .int

    Uses .geo dead areas as well as correlation
    """
    if coordinates == "geo":
        # Save the extra files too
        rsc_data = sario.load(sario.find_rsc_file(os.path.join(igram_path, "dem.rsc")))
        # .save_dem_to_h5( mask_file, rsc_data, dset_name=DEM_RSC_DSET, overwrite=overwrite
        if slc_path is None:
            slc_path = utils.get_parent_dir(igram_path)
    else:
        # TODO:
        rsc_data = None
        if slc_path is None:
            slc_path = os.path.join(igram_path, ISCE_SLC_DIR)

    # Used to shrink the .geo masks to save size as .int masks
    if row_looks is None or col_looks is None:
        row_looks, col_looks = apertools.sario.find_looks_taken(
            igram_path, slc_path=slc_path
        )

    if compute_from_slcs:
        all_slc_files = sario.find_slcs(directory=slc_path, parse=False)
        all_slc_dates = sario.find_slcs(directory=slc_path)
        slc_file_list = [
            f for f, d in zip(all_slc_files, all_slc_dates) if d in slc_date_list
        ]
        save_slc_masks(
            slc_path,
            mask_file,
            dem_rsc=rsc_data,
            slc_file_list=slc_file_list,
            row_looks=row_looks,
            col_looks=col_looks,
            overwrite=overwrite,
        )
    elif sario.check_dset(mask_file, SLC_MASK_DSET, overwrite):
        # Save empty SLC datasets
        fs = glob.glob(os.path.join(igram_path, ISCE_SLC_DIR, "**/*.slc"))
        with rio.open(fs[0]) as src:
            shape = (len(fs), *src.shape[-2:])
        print(SLC_MASK_SUM_DSET, shape)
        create_dset(mask_file, SLC_MASK_DSET, shape=shape, dtype=bool)
        create_dset(mask_file, SLC_MASK_SUM_DSET, shape=shape[1:], dtype=bool)
    if compute_water_mask:
        # TODO
        pass

    compute_int_masks(
        mask_file=mask_file,
        slc_path=slc_path,
        dem_rsc=rsc_data,
        slc_date_list=slc_date_list,
        ifg_date_list=ifg_date_list,
        ifg_file_list=ifg_file_list,
        overwrite=overwrite,
        compute_from_slcs=compute_from_slcs,
        mask_dem=mask_dem,
    )

    # Finally, attach the latitude/longitude datasets
    if coordinates == "geo":
        rsc_data = sario.load(sario.find_rsc_file(os.path.join(igram_path, "dem.rsc")))
        sario.save_latlon_to_h5(mask_file, rsc_data=rsc_data)
        attach_func = sario.attach_latlon
    else:
        lat, lon = sario.load_rdr_latlon(geom_dir=geom_dir)
        sario.save_latlon_2d_to_h5(mask_file, lat=lat, lon=lon, overwrite=overwrite)
        rsc_data = None
        attach_func = sario.attach_latlon_2d

    latlon_dsets = [
        SLC_MASK_DSET,
        SLC_MASK_SUM_DSET,
        IFG_MASK_DSET,
        IFG_MASK_SUM_DSET,
    ]
    depth_dims = ["slc_dates", None, "ifg_idx", None]
    for name, dim in zip(latlon_dsets, depth_dims):
        logger.info(f"attaching {dim} in {name} to depth")
        attach_func(mask_file, name, depth_dim=dim)


def save_slc_masks(
    directory,
    mask_file=MASK_FILENAME,
    dem_rsc=None,
    dset_name=SLC_MASK_DSET,
    slc_file_list=None,
    row_looks=1,
    col_looks=1,
    overwrite=False,
):
    """Creates .mask files for slc where zeros occur

    Makes look arguments are to create arrays the same size as the igrams
    Args:
        overwrite (bool): erase the dataset from the file if it exists and recreate
    """

    def _get_slc_mask(slc_arr):
        # Uses for removing single mask pixels from nearest neighbor resample
        m = binary_opening(np.abs(slc_arr) == 0, structure=np.ones((3, 3)))
        return np.ma.make_mask(m, shrink=False)

    # Make the empty stack, or delete if exists
    if not sario.check_dset(mask_file, dset_name, overwrite):
        return
    if not sario.check_dset(mask_file, SLC_MASK_SUM_DSET, overwrite):
        return

    # rsc_slc = sario.load(sario.find_rsc_file(directory=directory))
    rsc_slc = sario.load(os.path.join(directory, "elevation.dem.rsc"))
    big_shape = (rsc_slc["file_length"], rsc_slc["width"])
    if slc_file_list is None:
        slc_file_list = sario.find_slcs(directory=directory)

    shape = _find_file_shape(
        dem_rsc=dem_rsc,
        file_list=slc_file_list,
        row_looks=row_looks,
        col_looks=col_looks,
    )

    create_dset(mask_file, dset_name, shape=shape, dtype=bool)

    with h5py.File(mask_file, "a") as f:
        dset = f[dset_name]
        for idx, slc_fname in enumerate(tqdm(slc_file_list)):
            # save as an individual file too
            mask_name = os.path.split(slc_fname)[1] + ".mask"
            if not os.path.exists(mask_name):
                # g = sario.load(slc_fname, looks=(row_looks, col_looks))
                gmap = np.memmap(
                    slc_fname,
                    dtype="complex64",
                    mode="r",
                    shape=big_shape,
                )
                g_subsample = gmap[
                    (row_looks - 1) :: row_looks, (col_looks - 1) :: col_looks
                ]
                # ipdb.set_trace()
                tqdm.write(f"Saving {slc_fname} to stack")
                cur_mask = _get_slc_mask(g_subsample)
                sario.save(mask_name, cur_mask)
            else:
                cur_mask = sario.load(mask_name, rsc_file="dem.rsc")
            dset[idx] = cur_mask

        # Also add a composite mask depthwise
        f[SLC_MASK_SUM_DSET] = np.sum(dset, axis=0)


def compute_int_masks(
    mask_file=None,
    slc_path=None,
    dem_rsc=None,
    slc_date_list=None,
    ifg_date_list=None,
    ifg_file_list=None,
    dset_name=IFG_MASK_DSET,
    overwrite=False,
    compute_from_slcs=True,
    mask_dem=True,
    dem_filename="elevation_looked.dem",
):
    """Creates igram masks by taking the logical-or of the two .geo files

    Assumes save_slc_masks already run
    """
    if not sario.check_dset(mask_file, dset_name, overwrite):
        return
    if not sario.check_dset(mask_file, IFG_MASK_SUM_DSET, overwrite):
        return

    if slc_date_list is None and compute_from_slcs:
        slc_date_list = sario.find_slcs(directory=slc_path)

    # Make the empty stack, or delete if exists
    shape = _find_file_shape(dem_rsc=dem_rsc, file_list=ifg_file_list)
    create_dset(mask_file, dset_name, shape=shape, dtype=bool)

    if mask_dem:
        dem_mask = sario.load(dem_filename) == 0
    else:
        dem_mask = np.zeros(shape[-2:], dtype=bool)

    with h5py.File(mask_file, "a") as f:
        int_mask_dset = f[dset_name]
        # TODO: read coherence, also form that one
        for idx, (early, late) in enumerate(tqdm(ifg_date_list)):
            if compute_from_slcs:
                early_idx = slc_date_list.index(early)
                late_idx = slc_date_list.index(late)
                early_mask = f[SLC_MASK_DSET][early_idx]
                late_mask = f[SLC_MASK_DSET][late_idx]
                int_mask_dset[idx] = np.logical_or(early_mask, late_mask)
            elif mask_dem:
                int_mask_dset[idx] = dem_mask
            else:
                # print("Not masking")
                # int_mask_dset[idx] = np.ma.make_mask(dem_mask, shrink=False)
                # Already zeros... skip
                pass

        # Also create one image of the total masks
        f[IFG_MASK_SUM_DSET] = np.sum(int_mask_dset, axis=0)


def _find_file_shape(dem_rsc=None, file_list=None, row_looks=None, col_looks=None):
    if len(file_list) == 0:
        raise ValueError("No files found ")
    if not dem_rsc:
        if not row_looks and not col_looks:
            with rio.open(file_list[0]) as src:
                return (len(file_list), *src.shape[-2:])

        try:
            g = sario.load(file_list[0], looks=(row_looks, col_looks))
        except TypeError:
            raise ValueError("Need file_list if no dem_rsc")

        return (len(file_list), g.shape[0], g.shape[1])
    else:
        return (len(file_list), dem_rsc["file_length"], dem_rsc["width"])


def prepare_isce(
    project_dir=".",
    geom_dir=ISCE_GEOM_DIR,
    ref_row=None,
    ref_col=None,
    ref_lat=None,
    ref_lon=None,
    deramp_order=1,
    row_looks=5,
    col_looks=3,
    unw_filename=UNW_FILENAME,
    cor_filename=COR_FILENAME,
    mask_filename=MASK_FILENAME,
    search_term="Igrams/**/2*.unw",
    cor_search_term="Igrams/**/2*.cor",
    # For filtered version:
    # search_term="Igrams/**/filt*.unw",
    # cor_search_term="Igrams/**/filt*.cor",
):
    prepare_stacks(
        project_dir,
        ref_row=ref_row,
        ref_col=ref_col,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        unw_filename=unw_filename,
        cor_filename=cor_filename,
        mask_filename=mask_filename,
        deramp_order=deramp_order,
        search_term=search_term,
        cor_search_term=cor_search_term,
        row_looks=row_looks,
        col_looks=col_looks,
        coordinates="rdr",
        mask_from_slcs=False,
        geom_dir=os.path.join(project_dir, geom_dir),
        mask_dem=False,
        float_cor=True,
    )


def redo_deramp(
    unw_stack_file,
    ref_row=None,
    ref_col=None,
    deramp_order=2,
    dset_name=STACK_FLAT_SHIFTED_DSET,
    # cor_filename=COR_FILENAME,
    cor_filename="cor_stack_filt.h5",  # TODO: change
    chunk_layers=None,
    dem_file="elevation_looked.dem",
    window=5,
    cur_layer=0,
    subfactor=5,
    cor_thresh=0.4,
    coordinates="geo",
):
    # cur_layer = 0
    if coordinates == "geo":
        dem = sario.load(dem_file)
    else:
        dem = sario.load(dem_file, use_gdal=True, band=1)
    dem_sub = utils.take_looks(dem, subfactor, subfactor)

    # TODO
    # if ref_station is not None:
    #     rsc_data = sario.load(os.path.join(igram_path, "dem.rsc"))
    #     ref_row, ref_col = apertools.gps.station_rowcol(
    #         station_name=ref_station,
    #         rsc_data=rsc_data,
    #     )
    with h5py.File(cor_filename) as hf:
        cor_mean = hf[STACK_MEAN_DSET][()]

    # TODO: hardcode bad
    shadow = sario.load("geom_reference/shadowMask.rdr", use_gdal=True, band=1).astype(
        bool
    )
    water = ~sario.load("geom_reference/waterMask.rdr", use_gdal=True, band=1).astype(
        bool
    )
    cor = cor_mean < cor_thresh
    mask = np.any(np.stack((shadow, water, cor)), axis=0)
    mask_sub = utils.take_looks(mask, subfactor, subfactor).astype(bool)

    with h5py.File(unw_stack_file, "a") as hf:
        dset = hf[dset_name]
        total_layers, rows, cols = dset.shape
        if not chunk_layers:
            chunk_shape = hf[dset_name].chunks
            chunk_layers, _, _ = chunk_shape
        buf = np.zeros((chunk_layers, rows, cols), dtype=np.float32)

        while cur_layer < total_layers:
            if cur_layer + chunk_layers > total_layers:
                # over the edge for the final part
                cur_slice = np.s_[cur_layer:]
                dest_slice = np.s_[: total_layers - cur_layer]

            else:
                # normal operation
                cur_slice = np.s_[cur_layer : cur_layer + chunk_layers]
                dest_slice = np.s_[:chunk_layers]

            logger.info(f"Deramping {cur_slice}")
            buf *= 0
            dset.read_direct(buf, cur_slice, dest_slice)
            # out = deramp.remove_ramp(buf[dest_slice], deramp_order=deramp_order)
            out = remove_elevation(buf[dest_slice], dem, dem_sub, mask_sub, subfactor)

            # # Now center it on the shift window
            # win = window // 2
            # phase = _shift(phase, ref_row, ref_col, win)

            logger.info(f"Writing {cur_slice} back to dataset")
            dset.write_direct(out[dest_slice], dest_sel=cur_slice)

            cur_layer += chunk_layers

    # TODO: what do i want to record here...
    # # Now record attrs of the dataset
    # with h5py.File(unw_stack_file, "r+") as f:
    #     f[dset_name].attrs["deramp_order"] = deramp_order
    #     f[dset_name].attrs["reference"] = [ref_row, ref_col]
    #     # if ref_station is not None:
    #     # f[STACK_FLAT_SHIFTED_DSET].attrs["reference_station"] = ref_station

    # dirname = os.path.dirname(unw_stack_file)
    # if coordinates == "geo":
    #     rsc_data = sario.load(os.path.join(dirname, "dem.rsc"))
    #     sario.save_latlon_to_h5(unw_stack_file, rsc_data=rsc_data)
    #     sario.attach_latlon(unw_stack_file, dset_name, depth_dim="ifg_idx")
    # else:
    #     lat, lon = sario.load_rdr_latlon(geom_dir=os.path.join(dirname, ISCE_GEOM_DIR))
    #     sario.save_latlon_2d_to_h5(unw_stack_file, lat=lat, lon=lon)
    #     sario.attach_latlon_2d(unw_stack_file, dset_name, depth_dim="ifg_idx")


def redo_reference(
    unw_stack_file,
    ref_lat=None,
    ref_lon=None,
    ref_row=None,
    ref_col=None,
    ref_station=None,
    dset_name=STACK_FLAT_SHIFTED_DSET,
    geom_dir=ISCE_GEOM_DIR,
    coordinates="geo",
    start_layer=0,
    window=5,
):

    win = window // 2

    with h5py.File(unw_stack_file, "r+") as f:

        nimages, rows, cols = f[dset_name].shape
        chunk_shape = f[dset_name].chunks
        # chunk_layers, chunk_rows, chunk_cols = chunk_shape
        chunk_layers = chunk_shape[0]

    buf = np.zeros((chunk_layers, rows, cols), dtype=np.float32)
    ref_row, ref_col, ref_lat, ref_lon, ref_station = get_reference(
        ref_row,
        ref_col,
        ref_lat,
        ref_lon,
        ref_station,
        coordinates,
        unw_stack_file,
        geom_dir,
    )

    cur_layer = start_layer
    with h5py.File(unw_stack_file, "a") as hf:
        dset = hf[dset_name]
        total_layers, rows, cols = dset.shape

        while cur_layer < total_layers:
            if cur_layer + chunk_layers > total_layers:
                cur_slice = np.s_[cur_layer:]
                dest_slice = np.s_[: total_layers - cur_layer]

            else:
                cur_slice = np.s_[cur_layer : cur_layer + chunk_layers]
                dest_slice = np.s_[:chunk_layers]

            logger.info(f"Recentering {cur_slice}")
            buf *= 0
            dset.read_direct(buf, cur_slice, dest_slice)
            # Now center it on the shift window
            buf = _shift(buf, ref_row, ref_col, win)

            logger.info(f"Writing {cur_slice} back to dataset")
            dset.write_direct(buf[dest_slice], dest_sel=cur_slice)

            cur_layer += chunk_layers

    with h5py.File(unw_stack_file, "r+") as f:
        f[dset_name].attrs["reference"] = [ref_row, ref_col]
        f[dset_name].attrs["reference_latlon"] = [ref_lat, ref_lon]
        f[dset_name].attrs["reference_window"] = window
        f[dset_name].attrs["reference_station"] = ref_station or ""


def detect_rdr_coordinates(igram_path):
    if any(ISCE_GEOM_DIR in f for f in glob.glob(os.path.join(igram_path, "*"))):
        return "rdr"
    else:
        return "geo"


def get_reference(
    ref_row,
    ref_col,
    ref_lat,
    ref_lon,
    ref_station,
    coordinates,
    unw_stack_file,
    geom_dir,
):
    import apertools.latlon
    import apertools.gps

    if coordinates == "geo":
        if ref_station is not None:
            ref_lon, ref_lat = apertools.gps.station_lonlat(
                station_name=ref_station,
            )
            ref_row, ref_col = apertools.latlon.latlon_to_rowcol(
                ref_lat,
                ref_lon,
                filename=unw_stack_file,
            )
        elif ref_lat is not None and ref_lon is not None:
            ref_row, ref_col = apertools.latlon.latlon_to_rowcol(
                ref_lat,
                ref_lon,
                filename=unw_stack_file,
            )
        elif ref_row is not None and ref_col is not None:
            ref_lat, ref_lon = apertools.latlon.rowcol_to_latlon(
                ref_row,
                ref_col,
                filename=unw_stack_file,
            )
    else:
        if ref_station is not None:
            ref_lon, ref_lat = apertools.gps.station_lonlat(
                station_name=ref_station,
            )
            ref_row, ref_col = apertools.latlon.latlon_to_rowcol_rdr(
                ref_lat, ref_lon, geom_dir=geom_dir
            )
        elif ref_lat is not None and ref_lon is not None:
            ref_row, ref_col = apertools.latlon.latlon_to_rowcol_rdr(
                ref_lat, ref_lon, geom_dir=geom_dir
            )
        elif ref_row is not None and ref_col is not None:
            ref_lat, ref_lon = apertools.latlon.rowcol_to_latlon_rdr(
                ref_row, ref_col, geom_dir=geom_dir
            )
    if any((r is None for r in [ref_row, ref_col, ref_lat, ref_lon])):
        raise ValueError("need either ref_row/ref_col or ref_lat/ref_lon")
    return ref_row, ref_col, ref_lat, ref_lon, ref_station


# def remove_elevation(dem_da_sub, ifg_stack_sub, dem, ifg_stack, subfactor=5):
def remove_elevation(ifg_stack, dem, dem_sub, mask=None, subfactor=5):
    if mask is None:
        mask = np.zeros(dem.shape, dtype=bool)

    ifg_stack_sub = np.stack(
        [utils.take_looks(ifg, subfactor, subfactor) for ifg in ifg_stack]
    )
    # cols = dem_sub.shape[1]
    # col_slices = [slice(0, cols // 2), slice(cols // 2, None)]
    # col_slices = [slice(0, cols)]
    # polys = []
    # col_maxes = []
    # cur_col = 0
    # halves = []
    # # for col_slice in col_slices:
    # # col_slice = slice(cur_col, cur_col + cols // 2)
    # # print("Dem subset shape", dem_da_sub[:, col_slice].shape)
    # col_slice = slice(None)
    # # dem_pixels = dem_da_sub[:, col_slice].stack(space=("lat", "lon"))
    # # ifg_pixels = ifg_stack_sub[:, :, col_slice].stack(space=("lat", "lon"))

    # dem_pixels = dem_sub.reshape(-1)
    # mask_pixels = cor_mask.reshape(-1)
    # ifg_pixels = ifg_stack_sub.reshape((len(ifg_stack), -1))

    dem_pixels = dem_sub[mask]
    # mask_pixels = cor_mask.reshape(-1)
    ifg_pixels = ifg_stack_sub[:, mask]
    # print("ifg pixels shape:", ifg_pixels.shape)

    xx = dem_pixels.data  # (K, )
    yy = ifg_pixels.T  # (K, M)
    # mask_na = np.logical_or(np.isnan(xx), np.isnan(yy))
    # xx, yy = xx[~mask_na], yy[~mask_na]

    pf = np.polyfit(xx, np.nan_to_num(yy), 1)
    # polys.append(pf)
    # print(pf)
    # return xr.DataArray(pf)
    # Now get the full sized ifg and dem
    # ifgs_corrected = []
    # for idx, ifg in enumerate(ifg_stack):
    # ifg_half = ifg[:, col_slice]
    # ifgs_corrected.append(ifg_half - np.polyval(pf[:, idx], dem_half))
    # halves.append(np.stack(ifgs_corrected))

    # full_col_slice = slice(subfactor* cur_col, subfactor*(cur_col + cols // 2))
    # dem_half = dem[:, full_col_slice]
    # dem_half = dem
    # return pf
    from numpy.polynomial.polynomial import polyval

    # the new Polynomial API does coeffs low to high (for old, first was x^1, then x^0)
    corrections = polyval(dem, pf[::-1, :], tensor=True)
    return ifg_stack - corrections

    # corrections = corrected_pixels.reshape(ifg_stack[:, :, full_col_slice].shape)
    # corrections = corrected_pixels.reshape(ifg_stack.shape)
    # return corrections
    # halves.append(corrections)

    # cur_col += cols // 2
    # col_maxes.append(cur_col)


def apply_phasemask(unw_low, intf_high):
    """Apply the integer phase ambiguity in the unwrapped phase file unw_low to the
    phase in complex file intf_high, writing the result to outfile.  You should use this
    routine to apply an unwrapping solution obtained on low resolution data
    back to a higher resolution version of the same data.

    """
    from skimage.transform import resize

    # logger.info("Applying phase from %s to %s.", unw_low, intf_high)
    unw_high = resize(unw_low, intf_high.shape, mode="constant", anti_aliasing=False)

    twopi = 2 * np.pi
    # highres = sario.load(intf_high)
    highres = np.angle(intf_high) if np.iscomplexobj(intf_high) else intf_high

    # Do some fancy modular arithmetic to apply the phase ambiguity.
    dx = highres - unw_high
    ambig = twopi * np.around(dx / twopi)
    highres = highres - ambig
    # return np.stack((np.abs(intf_high), highres), axis=0)
    return highres


@log_runtime
def create_ifg_stack(
    ifg_stack_file=IFG_FILENAME,
    dset_name=STACK_DSET,
    directory=".",
    search_term="*.int",
    overwrite=False,
    geom_dir=ISCE_GEOM_DIR,
    coordinates="geo",
    max_temp=10000,
):

    if not sario.check_dset(ifg_stack_file, dset_name, overwrite):
        return
    # First make the empty dataset and save aux info
    file_list = sario.find_ifgs(
        directory=directory, search_term=search_term, parse=False
    )
    ifg_date_list = sario.find_ifgs(directory=directory, search_term=search_term)

    logger.info(f"Found {len(file_list)} total files")
    tuple_list = [
        (f, d)
        for (f, d) in zip(file_list, ifg_date_list)
        if temporal_baseline(f) < max_temp
    ]
    file_list, ifg_date_list = zip(*tuple_list)
    logger.info(f"Creating {ifg_stack_file} of {len(file_list)} files")

    slc_date_list = list(sorted(set(itertools.chain.from_iterable(ifg_date_list))))
    sario.save_ifglist_to_h5(
        ifg_date_list=ifg_date_list, out_file=ifg_stack_file, overwrite=overwrite
    )
    sario.save_slclist_to_h5(
        slc_date_list=slc_date_list, out_file=ifg_stack_file, overwrite=overwrite
    )
    # return

    band = 1
    load_func = _get_load_func(file_list[0], band=band)
    rows, cols = load_func(file_list[0]).shape

    dtype = np.complex64

    shape = (len(file_list), rows, cols)
    create_dset(ifg_stack_file, dset_name, shape, dtype, chunks=True, compress=True)

    with h5py.File(ifg_stack_file, "r+") as f:
        chunk_shape = f[dset_name].chunks
        chunk_depth, chunk_rows, chunk_cols = chunk_shape
        # n = n or chunk_size[0]

    buf = np.empty((chunk_depth, rows, cols), dtype=dtype)
    lastidx = 0
    cur_chunk_size = 0
    for idx, in_fname in enumerate(tqdm(file_list)):
        if idx % chunk_depth == 0 and idx > 0:
            tqdm.write(f"Writing {lastidx}:{lastidx+chunk_depth}")
            assert cur_chunk_size <= chunk_depth
            with h5py.File(ifg_stack_file, "r+") as f:
                f[dset_name][lastidx : lastidx + cur_chunk_size, :, :] = buf

            lastidx = idx
            cur_chunk_size = 0

        # driver = "ROI_PAC" if in_fname.endswith(".int") else None  # let gdal guess
        ifg = load_func(in_fname)
        # store this in the buffer until emptied
        curidx = idx % chunk_depth
        buf[curidx, :, :] = ifg
        cur_chunk_size += 1

    if cur_chunk_size > 0:
        # Write the final part of the buffer:
        with h5py.File(ifg_stack_file, "r+") as f:
            f[dset_name][lastidx : lastidx + cur_chunk_size, :, :] = buf[
                :cur_chunk_size
            ]

    # Save the lat/lon datasets, attach to main data files
    if coordinates == "geo":
        # TODO: best way to check for radar coords? wouldn't want any lat/lon...
        rsc_data = sario.load(os.path.join(directory, "dem.rsc"))
        sario.save_latlon_to_h5(ifg_stack_file, rsc_data=rsc_data)
        sario.attach_latlon(ifg_stack_file, dset_name, depth_dim="ifg_idx")
    else:
        lat, lon = sario.load_rdr_latlon(geom_dir=geom_dir)
        sario.save_latlon_2d_to_h5(
            ifg_stack_file, lat=lat, lon=lon, overwrite=overwrite
        )
        sario.attach_latlon_2d(ifg_stack_file, dset_name, depth_dim="ifg_idx")