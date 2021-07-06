import h5py
import hdf5plugin
import os

# import subprocess
import numpy as np
from scipy.ndimage.morphology import binary_opening
import rasterio as rio

from apertools import sario, utils, latlon
import apertools.gps
from apertools.log import get_log, log_runtime
import multiprocessing


# TODO: this is for reading windows of a big ratser, not needed for now
def all_bands(file_list, band=2, col_off=0, row_off=0, height=20):
    from rasterio.windows import Window

    with rio.open(file_list[0]) as src:
        rows, cols = src.shape
        # bshape = src.block_shapes[band-1]  # TODO: use?
        dt = src.dtypes[band - 1]

    block = np.empty((len(file_list), height, cols), dtype=dt)
    for idx, f in enumerate(file_list):
        try:
            with rio.open(f, driver="ROI_PAC") as src:
                block[idx] = src.read(
                    band, window=Window(col_off, row_off, cols, height)
                )
        except Exception as e:
            logger.warning(idx, f, e)
    return block


def load_in_chunks(
    unw_stack_file="unw_stack.h5", flist=[], dset="stack_flat_dset", n=None
):
    with h5py.File(unw_stack_file, "r+") as f:
        chunk_size = f[dset].chunks
        dshape = f[dset].shape
        dt = f[dset].dtype

    n = n or chunk_size[0]
    buf = np.empty((n, dshape[1], dshape[2]), dtype=dt)
    lastidx = 0
    for idx, fname in enumerate(flist):
        if idx % n == 0 and idx > 0:
            logger.info(f"Writing {lastidx}:{lastidx+n}")
            with h5py.File("unw_test.h5", "r+") as f:
                f[dset][lastidx : lastidx + n, :, :] = buf
            lastidx = idx

        with rio.open(fname, driver="ROI_PAC") as src:
            curidx = idx % n
            buf[curidx, :, :] = src.read(2)
    return buf


def _run_stack(igram_path, d, overwrite):
    if d["filename"] is None:
        return
    logger.info("Creating hdf5 stack %s" % d["filename"])
    create_hdf5_stack(directory=igram_path, overwrite=overwrite, **d)
    sario.save_slclist_to_h5(igram_path, d["filename"], overwrite=overwrite)
    sario.save_ifglist_to_h5(igram_path, d["filename"], overwrite=overwrite)


@log_runtime
def create_igram_stacks(
    igram_path,
    int_stack_file=INT_FILENAME,
    unw_stack_file=UNW_FILENAME,
    cc_stack_file=CC_FILENAME,
    overwrite=False,
):
    # TODO: make this just make a vrt of unw and .int
    sario.make_unw_vrt(directory=igram_path, output="unw_stack.vrt", ext=".unw")
    stack_dicts = (
        # dict(file_ext=".int", create_mean=False, filename=int_stack_file),
        dict(file_ext=".unw", create_mean=False, filename=unw_stack_file),
        # dict(file_ext=".cc", create_mean=True, filename=cc_stack_file),
    )
    for d in stack_dicts:
        if d["filename"] is None:
            continue
        logger.info("Creating hdf5 stack %s" % d["filename"])
        create_hdf5_stack(directory=igram_path, overwrite=overwrite, **d)
        sario.save_slclist_to_h5(igram_path, d["filename"], overwrite=overwrite)
        sario.save_ifglist_to_h5(igram_path, d["filename"], overwrite=overwrite)

    pool = multiprocessing.Pool()
    results = [
        pool.apply_async(_run_stack, args=(igram_path, d, overwrite))
        for d in stack_dicts
    ]
    return [res.get() for res in results]


@log_runtime
def create_mask_stacks_gdal(
    igram_path, mask_filename=None, geo_path=None, overwrite=False
):
    """Create mask stacks for areas in .geo and .int using `gdal_translate`

    Uses .geo dead areas
    """
    import gdal
    from osgeo import gdalconst  # gdal_array,

    if mask_filename is None:
        mask_file = os.path.join(igram_path, MASK_FILENAME)

    if geo_path is None:
        geo_path = utils.get_parent_dir(igram_path)

    # Used to shrink the .geo masks to save size as .int masks
    row_looks, col_looks = apertools.sario.find_looks_taken(
        igram_path, geo_path=geo_path
    )

    rsc_data = sario.load(sario.find_rsc_file(os.path.join(igram_path, "dem.rsc")))

    save_geo_masks_gdal(
        geo_path,
        mask_file,
        dem_rsc=rsc_data,
        overwrite=overwrite,
    )


def create_hdf5_stack(
    filename=None,
    directory=None,
    dset=STACK_DSET,
    file_ext=None,
    create_mean=True,
    save_rsc=True,
    overwrite=False,
    **kwargs,
):
    """Make stack as hdf5 file from a group of existing files

    Args:
        filename (str): if none provided, creates a file `[file_ext]_stack.h5`

    Returns:
        filename
    """

    def _create_mean(dset):
        """Used to create a mean without loading all into mem with np.mean"""
        mean_buf = np.zeros((dset.shape[1], dset.shape[2]), dset.dtype)
        for idx in range(len(dset)):
            mean_buf += dset[idx]
        return mean_buf / len(dset)

    if not filename:
        fname = "{fext}_stack.h5".format(fext=file_ext.strip("."))
        filename = os.path.abspath(os.path.join(directory, fname))
        logger.info("Creating stack file %s" % filename)

    if utils.get_file_ext(filename) not in (".h5", ".hdf5"):
        raise ValueError("filename must end in .h5 or .hdf5")

    # TODO: do we want to replace the .unw files with .h5 files, then make a Virtual dataset?
    # layout = h5py.VirtualLayout(shape=(len(file_list), nrows, ncols), dtype=dtype)
    if not sario.check_dset(filename, dset, overwrite):
        return

    file_list = sario.find_files(directory=directory, search_term="*" + file_ext)

    testf = sario.load(file_list[0])
    shape = (len(file_list), testf.shape[0], testf.shape[1])
    create_dset(filename, dset, shape, dtype=testf.dtype)
    with h5py.File(filename, "a") as hf:
        # First record the names in a dataset
        filename_dset = dset + "_filenames"
        hf[filename_dset] = np.array(file_list, dtype=np.string_)

        dset = hf[dset]
        for idx, f in enumerate(file_list):
            dset[idx] = sario.load(f)

    if save_rsc:
        dem_rsc = sario.load(os.path.join(directory, "dem.rsc"))
        sario.save_dem_to_h5(
            filename, dem_rsc, dset_name=DEM_RSC_DSET, overwrite=overwrite
        )

    if create_mean:
        if not sario.check_dset(filename, STACK_MEAN_DSET, overwrite):
            return
        with h5py.File(filename, "a") as hf:
            mean_data = _create_mean(hf[STACK_DSET])
            hf.create_dataset(
                STACK_MEAN_DSET,
                data=mean_data,
            )

    return filename


def save_geo_masks_gdal(
    directory=".",
    mask_file="masks.vrt",
    dem_rsc=None,
    dset_name=GEO_MASK_DSET,
    overwrite=False,
):
    """Creates .mask files for geos where zeros occur

    Makes look arguments are to create arrays the same size as the igrams
    Args:
        overwrite (bool): erase the dataset from the file if it exists and recreate
    """
    import gdal
    from osgeo import gdalconst  # gdal_array,

    rsc_data = sario.load(dem_rsc)
    save_path = os.path.split(mask_file)[0]

    geo_file_list = sario.find_files(directory=directory, search_term="*.geo")
    for f in geo_file_list:
        logger.info("Processing mask for %s" % f)
        rsc_geo = sario.find_rsc_file(filename=f)
        apertools.sario.save_as_vrt(filename=f, rsc_file=rsc_geo)
        src = f + ".vrt"
        tmp_tif = os.path.join(save_path, "tmp.tif")
        # gdal_translate S1A_20171019.geo.vrt tmp.tif -outsize 792 624 -r average
        gdal.Translate(
            tmp_tif,
            src,
            # noData=0.0,
            width=rsc_data["width"],
            height=rsc_data["file_length"],
            # resampleAlg=gdalconst.GRA_Average,
            resampleAlg=gdalconst.GRA_NearestNeighbour,
        )
        # Now find zero pixels to get mask:
        ds = gdal.Open(tmp_tif)
        in_arr = ds.GetRasterBand(1).ReadAsArray()
        # Uses for removing single mask pixels from nearest neighbor resample
        mask_arr = binary_opening(np.abs(in_arr) == 0, structure=np.ones((3, 3)))

        outfile = os.path.join(save_path, os.path.split(f + ".mask.tif")[1])
        sario.save_as_geotiff(outfile=outfile, array=mask_arr, rsc_data=rsc_data)
        # cmd = gdal_calc.py -A {inp} --outfile {out} --type Byte --calc="abs(A)==0"
        # logger.info(cmd)
        # subprocess.run(cmd, shell=True, check=True)


def shift_unw_file(
    unw_stack_file,
    ref_row,
    ref_col,
    out_dset=STACK_FLAT_SHIFTED_DSET,
    window=3,
    ref_station=None,
    overwrite=False,
):
    """Runs a reference point shift on flattened stack of unw files stored in .h5"""
    logger.info(
        "Starting shift_stack: using %s, %s as ref_row, ref_col", ref_row, ref_col
    )
    if not sario.check_dset(unw_stack_file, out_dset, overwrite):
        return

    in_files = sario.find_files(".", "*.unwflat")
    rows, cols = sario.load(in_files[0]).shape
    with h5py.File(unw_stack_file, "a") as f:
        f.create_dataset(
            out_dset,
            shape=(len(in_files), rows, cols),
            dtype="float32",
        )
        win = window // 2
        stack_out = f[out_dset]
        for idx, inf in enumerate(in_files):
            layer = sario.load(inf)
            patch = layer[
                ref_row - win : ref_row + win + 1, ref_col - win : ref_col + win + 1
            ]
            stack_out[idx] = layer - np.mean(patch)

    dem_rsc = sario.load("dem.rsc")
    sario.save_dem_to_h5(
        unw_stack_file, dem_rsc, dset_name=DEM_RSC_DSET, overwrite=overwrite
    )
    logger.info("Shifting stack complete")


def shift_stack(stack_in, stack_out, ref_row, ref_col, window=3):
    """Subtracts reference pixel group from each layer

    Args:
        stack_in (ndarray-like): 3D array of images, stacked along axis=0
        stack_out (ndarray-like): empty 3D array, will hold output
            Both can be hdf5 datasets
        ref_row (int): row index of the reference pixel to subtract
        ref_col (int): col index of the reference pixel to subtract
        window (int): size of the group around ref pixel to avg for reference.
            if window=1 or None, only the single pixel used to shift the group.

    Raises:
        ValueError: if window is not a positive int, or if ref pixel out of bounds
    """
    win = window // 2
    for idx, layer in enumerate(stack_in):
        patch = layer[
            ref_row - win : ref_row + win + 1, ref_col - win : ref_col + win + 1
        ]
        stack_out[idx] = layer - np.mean(patch)


def load_reference(unw_stack_file=UNW_FILENAME):
    with h5py.File(unw_stack_file, "r") as f:
        try:
            return f[STACK_FLAT_SHIFTED_DSET].attrs["reference"]
        except KeyError:
            return None, None


@log_runtime
def deramp_stack(
    unw_stack_file=UNW_FILENAME,
    order=1,
    overwrite=False,
):
    """Handles removing linear ramps for all files in a stack

    Saves the files to a new dataset in the same unw stack .h5 file

    Args:
        unw_stack_file (str): Filename for the .h5 stack of .unw
            These layers will be deramped and saved do a new dset
        order (int): order of polynomial surface to use to deramp
            1 is linear (default), 2 is quadratic
    """
    logger.info("Removing any ramp from each stack layer")
    # Get file names to save results/ check if we deramped already

    # Make sure the normal .unw stack file has been created
    with h5py.File(unw_stack_file, "r") as f:
        if STACK_DSET not in f:
            raise ValueError("unw stack dataset doesn't exist at %s" % unw_stack_file)

    if not sario.check_dset(unw_stack_file, STACK_FLAT_DSET, overwrite):
        return

    with h5py.File(MASK_FILENAME) as fmask:
        mask_dset = fmask[IGRAM_MASK_DSET]
        with h5py.File(unw_stack_file, "a") as f:
            logger.info("Creating dataset %s in %s" % (STACK_FLAT_DSET, unw_stack_file))

            f.create_dataset(
                STACK_FLAT_DSET,
                shape=f[STACK_DSET].shape,
                dtype=f[STACK_DSET].dtype,
            )
            # Shape of sario.load_stack with return_amp is (nlayers, 2, nrows, ncols)
            for idx, layer in enumerate(f[STACK_DSET]):
                with mask_dset.astype(bool):
                    mask = mask_dset[idx]
                try:
                    f[STACK_FLAT_DSET][idx] = remove_ramp(layer, order=order, mask=mask)
                except np.linalg.linalg.LinAlgError:
                    logger.info(
                        "Failed to estimate ramp on layer %s: setting to 0" % idx
                    )
                    f[STACK_FLAT_DSET][idx] = np.zeros_like(layer)


# # TODO: change this to the Rowena paper for auto find
# from scipy.ndimage.filters import uniform_filter
# def find_coherent_patch(correlations, window=11):
#     """Looks through 3d stack of correlation layers and finds strongest correlation patch
#
#     Also accepts a 2D array of the pre-compute means of the 3D stack.
#     Uses a window of size (window x window), finds the largest average patch
#
#     Args:
#         correlations (ndarray, possibly masked): 3D array of correlations:
#             correlations = sario.load_stack('path/to/correlations', '.cc')
#
#         window (int): size of the patch to consider
#
#     Returns:
#         tuple[int, int]: the row, column of center of the max patch
#
#     Example:
#         >>> corrs = np.arange(25).reshape((5, 5))
#         >>> print(find_coherent_patch(corrs, window=3))
#         (3, 3)
#         >>> corrs = np.stack((corrs, corrs), axis=0)
#         >>> print(find_coherent_patch(corrs, window=3))
#         (3, 3)
#     """
#     correlations = correlations.view(np.ma.MaskedArray)  # Force to be type np.ma
#     if correlations.ndim == 2:
#         mean_stack = correlations
#     elif correlations.ndim == 3:
#         mean_stack = np.ma.mean(correlations, axis=0)
#     else:
#         raise ValueError("correlations must be a 2D mean array, or 3D correlations")
#
#     # Run a 2d average over the image, then convert to masked array
#     conv = uniform_filter(mean_stack, size=window, mode='constant')
#     conv = np.ma.array(conv, mask=correlations.mask.any(axis=0))
#     # Now find row, column of the max value
#     max_idx = conv.argmax()
#     return np.unravel_index(max_idx, mean_stack.shape)

# TODO: do this with gdal calc
# @log_runtime
# def zero_masked_areas(igram_path=".", mask_filename=None, verbose=True):
#     logger.info("Zeroing out masked area in .cc and .int files")
#
#     if mask_filename is None:
#         mask_filename = os.path.join(igram_path, MASK_FILENAME)
#
#     int_date_list = sario.load_ifglist_from_h5(mask_filename)
#
#     with h5py.File(mask_filename, "r") as f:
#         igram_mask_dset = f[IGRAM_MASK_DSET]
#         for idx, (early, late) in enumerate(int_date_list):
#             cur_mask = igram_mask_dset[idx]
#             base_str = "%s_%s" % (early.strftime(DATE_FMT), late.strftime(DATE_FMT))
#
#             if verbose:
#                 logger.info("Zeroing {0}.cc and {0}.int".format(base_str))
#
#             int_filename = base_str + ".int"
#             zero_file(int_filename, cur_mask, is_stacked=False)
#
#             cc_filename = base_str + ".cc"
#             zero_file(cc_filename, cur_mask, is_stacked=True)

# TODO: do this with gdal_calc.py
# def zero_file(filename, mask, is_stacked=False):
#     if is_stacked:
#         amp, img = sario.load(filename, return_amp=True)
#         img[mask] = 0
#         sario.save(filename, np.stack((amp, img), axis=0))
#     else:
#         img = sario.load(filename)
#         img[mask] = 0
#         sario.save(filename, img)


# TODO: decide if there's a fesible way to add a file to the repacked HDF5...
# @log_runtime
# def merge_files(filename1, filename2, new_filename, overwrite=False):
#     """Merge together 2 (currently mask) hdf5 files into a new file"""
#     def _merge_lists(list1, list2, merged_list, dset_name, dset1, dset2):
#         logger.info("%s: %s from %s and %s from %s into %s in file %s" % (
#             dset_name,
#             len(list1),
#             filename1,
#             len(list2),
#             filename2,
#             len(merged_list),
#             new_filename,
#         ))
#         for idx in range(len(merged_list)):
#             cur_item = merged_list[idx]
#             if cur_item in list1:
#                 jdx = list1.index(cur_item)
#                 fnew[dset_name][idx] = dset1[jdx]
#             else:
#                 jdx = list2.index(cur_item)
#                 fnew[dset_name][idx] = dset2[jdx]
#
#     if overwrite:
#         sario.check_dset(new_filename, IGRAM_MASK_DSET, overwrite)
#         sario.check_dset(new_filename, GEO_MASK_DSET, overwrite)
#
#     f1 = h5py.File(filename1)
#     f2 = h5py.File(filename2)
#     igram_dset1 = f1[IGRAM_MASK_DSET]
#     igram_dset2 = f2[IGRAM_MASK_DSET]
#     geo_dset1 = f1[GEO_MASK_DSET]
#     geo_dset2 = f2[GEO_MASK_DSET]
#
#     ifglist1 = sario.load_ifglist_from_h5(filename1)
#     ifglist2 = sario.load_ifglist_from_h5(filename2)
#     slclist1 = sario.load_slclist_from_h5(filename1)
#     slclist2 = sario.load_slclist_from_h5(filename2)
#     merged_ifglist = sorted(set(ifglist1) | set(ifglist2))
#     merged_slclist = sorted(set(slclist1) | set(slclist2))
#
#     sario.save_ifglist_to_h5(out_file=new_filename, overwrite=True, int_date_list=merged_ifglist)
#     sario.save_slclist_to_h5(out_file=new_filename, overwrite=True, geo_date_list=merged_slclist)
#
#     new_geo_shape = (len(merged_slclist), geo_dset1.shape[1], geo_dset1.shape[2])
#     create_dset(new_filename, GEO_MASK_DSET, new_geo_shape, dtype=igram_dset1.dtype)
#     new_igram_shape = (len(merged_ifglist), igram_dset1.shape[1], igram_dset1.shape[2])
#     create_dset(new_filename, IGRAM_MASK_DSET, new_igram_shape, dtype=igram_dset1.dtype)
#
#     fnew = h5py.File(new_filename, "a")
#     try:
#         _merge_lists(slclist1, slclist2, merged_slclist, GEO_MASK_DSET, geo_dset1, geo_dset2)
#        _merge_lists(ifglist1, ifglist2, merged_ifglist, IGRAM_MASK_DSET, igram_dset1, igram_dset2)
#
#     finally:
#         f1.close()
#         f2.close()
#         fnew.close()
def find_reference_location(
    unw_stack_file=UNW_FILENAME,  # TODO: do i need this
    mask_stack_file=MASK_FILENAME,
    cc_stack_file=CC_FILENAME,
    ref_station=None,
    rsc_data=None,
):
    """Find reference pixel on based on GPS availability and mean correlation"""
    rsc_data = sario.load_dem_from_h5(h5file=unw_stack_file, dset="dem_rsc")

    # CHAGNE
    # Make a latlon image to check for gps data containment
    with h5py.File(unw_stack_file, "r") as f:
        latlon_image = latlon.LatlonImage(data=f[STACK_DSET][0], rsc_data=rsc_data)

    logger.info("Searching for gps station within area")
    # Don't make the invalid GPS here in case the random image chosed above is bad:
    # We'll use the mask ll image to decide which pixels are bad
    stations = apertools.gps.stations_within_image(latlon_image, mask_invalid=False)
    # Make a latlon image From the total masks
    with h5py.File(mask_stack_file, "r") as f:
        mask_ll_image = latlon.LatlonImage(data=f[GEO_MASK_SUM_DSET], rsc_data=rsc_data)

    with h5py.File(cc_stack_file, "r") as f:
        mean_cor = f[STACK_MEAN_DSET][:]
        mean_cor_ll_image = latlon.LatlonImage(data=mean_cor, rsc_data=rsc_data)

    if len(stations) > 0:
        logger.info("Station options:")
        logger.info(stations)
        num_masks = [mask_ll_image[lat, lon] for _, lon, lat in stations]
        pixel_correlations = [mean_cor_ll_image[lat, lon] for _, lon, lat in stations]

        logger.info("Sorting by fewer masked dates and highest correlation")
        # Note: make cor negative to sort large numbers to the front
        sorted_stations = sorted(
            zip(num_masks, pixel_correlations, stations),
            key=lambda tup: (tup[0], -tup[1]),
        )
        logger.info(sorted_stations)

        name, lon, lat = sorted_stations[0][-1]
        logger.info("Using station %s at (lon, lat) (%s, %s)", name, lon, lat)
        ref_row, ref_col = latlon_image.nearest_pixel(lon=lon, lat=lat)
        ref_station = name

    if ref_row is None:
        raise ValueError("GPS station search failed, need reference row/col")
        # logger.warning("GPS station search failed, reverting to coherence")
        # logger.info("Finding most coherent patch in stack.")
        # ref_row, ref_col = find_coherent_patch(mean_cor)
        # ref_station = None

    logger.info("Using %s as .unw reference point", (ref_row, ref_col))
    return ref_row, ref_col, ref_station
