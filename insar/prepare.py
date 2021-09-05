"""prepare.py

Preprocessing insar data for timeseries analysis

Forms stacks as .h5 files for easy access to depth-wise slices
"""
import os
import glob
import itertools
import h5py
from pygeos import coordinates
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
    if coordinates is None:
        coordinates = detect_rdr_coordinates(igram_path)
    if coordinates not in COORDINATES_CHOICES:
        raise ValueError("coordinates must be in {}".format(COORDINATES_CHOICES))
    if coordinates == "rdr":
        lon, lat = sario.load_rdr_latlon(geom_dir=geom_dir)

    ifg_date_list = sario.find_ifgs(directory=igram_path, search_term=search_term)
    ifg_file_list = sario.find_ifgs(
        directory=igram_path, search_term=search_term, parse=False
    )
    slc_date_list = list(sorted(set(itertools.chain.from_iterable(ifg_date_list))))

    mask_file = create_mask_stacks(
        igram_path,
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
    )

    cor_stack_file = os.path.join(igram_path, COR_FILENAME)
    create_cor_stack(
        cor_stack_file=cor_stack_file,
        dset_name=STACK_DSET,
        directory=igram_path,
        search_term=cor_search_term,
        overwrite=overwrite,
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
        import apertools.latlon

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

    if ref_row is None or ref_col is None:
        # ref_row, ref_col, ref_station = find_reference_location(
        raise ValueError("Need ref_row, ref_col or ref_station")

    # Now create the unwrapped ifg stack
    unw_stack_file = os.path.join(igram_path, UNW_FILENAME)
    deramp_and_shift_unws(
        ref_row,
        ref_col,
        unw_stack_file=unw_stack_file,
        dset_name=STACK_FLAT_SHIFTED_DSET,
        directory=igram_path,
        deramp_order=deramp_order,
        window=window,
        search_term=search_term,
        overwrite=overwrite,
    )
    # Now record attrs of the dataset
    with h5py.File(unw_stack_file, "r+") as f:
        f[STACK_FLAT_SHIFTED_DSET].attrs["deramp_order"] = deramp_order
        f[STACK_FLAT_SHIFTED_DSET].attrs["reference"] = [ref_row, ref_col]
        if ref_station is not None:
            f[STACK_FLAT_SHIFTED_DSET].attrs["reference_station"] = ref_station

    for f in [mask_file, cor_stack_file, unw_stack_file]:
        sario.save_ifglist_to_h5(
            ifg_date_list=ifg_date_list, out_file=f, overwrite=overwrite
        )

        sario.save_slclist_to_h5(
            slc_date_list=slc_date_list, out_file=f, overwrite=overwrite
        )


def create_dset(h5file, dset_name, shape, dtype, chunks=True, compress=True):
    # comp_dict = hdf5plugin.Blosc() if compress else dict()
    comp_dict = dict(compression="gzip") if compress else dict()
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
    mask_fname=MASK_FILENAME,
    dset_name=STACK_FLAT_SHIFTED_DSET,
    directory=".",
    search_term="*.unw",
    deramp_order=2,
    window=5,
    overwrite=False,
    stack_fname="stackavg.tif",
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
    stackavg = np.zeros((rows, cols), dtype="float32")

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
            with h5py.File(mask_fname, "r") as f:
                mask = f[IFG_MASK_DSET][idx, :, :].astype(bool)
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
        # TODO: best way to check for radar coords? wouldn't want any lat/lon...
        rsc_data = sario.load(os.path.join(directory, "dem.rsc"))
        sario.save_latlon_to_h5(unw_stack_file, rsc_data=rsc_data)
        sario.attach_latlon(unw_stack_file, dset_name, depth_dim="ifg_idx")
    else:
        lat, lon = sario.load_rdr_latlon(geom_dir=geom_dir)
        sario.save_latlon_2d_to_h5(
            unw_stack_file, lat=lat, lon=lon, overwrite=overwrite
        )
        sario.attach_latlon_2d(unw_stack_file, dset_name, depth_dim="ifg_idx")


def _get_load_func(in_fname, band=2):
    try:
        with rio.open(in_fname):
            pass

        def load_func(x):
            with rio.open(in_fname) as inf:
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
    mask_filename=None,
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
    if mask_filename is None:
        mask_file = os.path.join(igram_path, MASK_FILENAME)

    if coordinates == "geo":
        # Save the extra files too
        rsc_data = sario.load(sario.find_rsc_file(os.path.join(igram_path, "dem.rsc")))
        # .save_dem_to_h5( mask_file, rsc_data, dset_name=DEM_RSC_DSET, overwrite=overwrite
        sario.save_latlon_to_h5(mask_file, rsc_data=rsc_data)
        if slc_path is None:
            slc_path = utils.get_parent_dir(igram_path)
    else:
        # TODO:
        lat, lon = sario.load_rdr_latlon(geom_dir=geom_dir)
        sario.save_latlon_2d_to_h5(mask_file, lat=lat, lon=lon, overwrite=overwrite)
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
    else:
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
        igram_path=igram_path,
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
    return mask_file


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
    igram_path=None,
    slc_path=None,
    dem_rsc=None,
    igram_ext=".unw",
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


from helpers import elevation_vs_phase as evp


def redo_deramp(
    unw_stack_file,
    ref_row=None,
    ref_col=None,
    deramp_order=2,
    dset_name=STACK_FLAT_SHIFTED_DSET,
    chunk_layers=None,
    window=5,
    cur_layer=0,
    subfactor=5,
    coordinates="geo",
):
    # cur_layer = 0
    win = window // 2
    dem = sario.load("elevation_looked.dem")
    dem_sub = utils.take_looks(dem, subfactor, subfactor)

    # TODO
    # if ref_station is not None:
    #     rsc_data = sario.load(os.path.join(igram_path, "dem.rsc"))
    #     ref_row, ref_col = apertools.gps.station_rowcol(
    #         station_name=ref_station,
    #         rsc_data=rsc_data,
    #     )
    with h5py.File(unw_stack_file, "a") as hf:
        dset = hf[dset_name]
        total_layers, rows, cols = dset.shape
        if not chunk_layers:
            chunk_shape = hf[dset_name].chunks
            chunk_layers, _, _ = chunk_shape
        buf = np.zeros((chunk_layers, rows, cols), dtype=np.float32)

        while cur_layer < total_layers:
            if cur_layer + chunk_layers > total_layers:
                cur_slice = np.s_[cur_layer:]
                dest_slice = np.s_[: total_layers - cur_layer]

            else:
                cur_slice = np.s_[cur_layer : cur_layer + chunk_layers]
                dest_slice = np.s_[:chunk_layers]

            logger.info(f"Deramping {cur_slice}")
            dset.read_direct(buf, cur_slice, dest_slice)
            # out = deramp.remove_ramp(buf[dest_slice], deramp_order=deramp_order)
            out = remove_elevation(buf[dest_slice], dem, dem_sub, subfactor)

            # # Now center it on the shift window
            # patch = out[
            #     ref_row - win : ref_row + win + 1, ref_col - win : ref_col + win + 1
            # ]
            # if not np.all(np.isnan(patch)):
            #     out -= np.nanmean(patch)
            # else:
            #     # Do I actually just want to ignore this one and give 0s?
            #     logger.debug(f"Patch is all nan for {ref_row},{ref_col}")
            #     out -= np.nanmean(out)

            logger.info(f"Writing {cur_slice} back to dataset")
            dset.write_direct(out, dest_sel=cur_slice)

            cur_layer += chunk_layers

    # Now record attrs of the dataset
    with h5py.File(unw_stack_file, "r+") as f:
        f[dset_name].attrs["deramp_order"] = deramp_order
        f[dset_name].attrs["reference"] = [ref_row, ref_col]
        # TODO
        # if ref_station is not None:
        # f[STACK_FLAT_SHIFTED_DSET].attrs["reference_station"] = ref_station
    dirname = os.path.dirname(unw_stack_file)
    if coordinates == "geo":
        # TODO: best way to check for radar coords? wouldn't want any lat/lon...
        rsc_data = sario.load(os.path.join(dirname, "dem.rsc"))
        sario.save_latlon_to_h5(unw_stack_file, rsc_data=rsc_data)
        sario.attach_latlon(unw_stack_file, dset_name, depth_dim="ifg_idx")
    else:
        lat, lon = sario.load_rdr_latlon(geom_dir=os.path.join(dirname, ISCE_GEOM_DIR))
        sario.save_latlon_2d_to_h5(unw_stack_file, lat=lat, lon=lon)
        sario.attach_latlon_2d(unw_stack_file, dset_name, depth_dim="ifg_idx")


def detect_rdr_coordinates(igram_path):
    if any(ISCE_GEOM_DIR in f for f in glob.glob(os.path.join(igram_path, "*"))):
        return "rdr"
    else:
        return "geo"


def prepare_isce(
    project_dir=".",
    ref_row=None,
    ref_col=None,
    deramp_order=1,
    row_looks=5,
    col_looks=3,
):
    search_term = "Igrams/**/filt*.unw"
    cor_search_term = "Igrams/**/filt*.cor"
    prepare_stacks(
        project_dir,
        ref_row=ref_row,
        ref_col=ref_col,
        deramp_order=deramp_order,
        search_term=search_term,
        cor_search_term=cor_search_term,
        row_looks=row_looks,
        col_looks=col_looks,
        coordinates="rdr",
        mask_from_slcs=False,
        mask_dem=False,
        float_cor=True,
    )


# def remove_elevation(dem_da_sub, ifg_stack_sub, dem, ifg_stack, subfactor=5):
def remove_elevation(ifg_stack, dem, dem_sub, subfactor=5):
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
    dem_pixels = dem_sub.reshape(-1)
    ifg_pixels = ifg_stack_sub.reshape((len(ifg_stack), -1))
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


# return np.stack(halves)


# poly_da = xr.DataArray(
#     np.stack(polys),
#     coords={
#         "max_col": col_maxes,
#         "poly_coeff": [1, 0],
#         "ifg_idx": ifg_stack_sub.ifg_idx,
#     },
# )
