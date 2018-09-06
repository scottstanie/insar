import os
import glob
import subprocess
import numpy as np
import insar.parsers
from insar.log import get_log

logger = get_log()


def unzip_sentinel_files(path="."):
    logger.info("Changing to %s to unzip files", path)
    cur_dir = os.getcwd()  # To return to after
    os.chdir(path)

    logger.info("Unzipping sentinel annotation/xml and VV .tiffs")

    # Unzip all .zip files by piping to xargs, using 10 processes
    # IMPORTANT: Only unzipping the VV .tiff files and annotation/*.xml !

    # Note: -n means "never overwrite existing files", so you can rerun this
    subprocess.check_call(
        "find . -maxdepth 1 -name '*.zip' -print0 | "
        'xargs -0 -I {} --max-procs 10 unzip -n {} "*/annotation/*.xml" "*/measurement/*slc-vv-*.tiff" ',
        shell=True)

    logger.info("Done unzipping, returning to %s", cur_dir)
    os.chdir(cur_dir)


def find_sentinels(data_path, path_num=None):
    sents = [
        insar.parsers.Sentinel(f) for f in glob.glob(os.path.join(data_path, "*"))
        if f.endswith(".zip") or f.endswith(".SAFE")
    ]
    if path_num:
        sents = [s for s in sents if s.path == path_num]
    return list(set(sents))


def total_swath_extent(sentinel_list):
    """Take a list of Sentinels, and find total extent of area covered

    Args:
        list[Sentinel]: Sentinel objects covering one area

    Returns:
        tuple[float]: (min_lon, max_lon, min_lat, max_lat)
    """
    swath_extents = np.array([s.swath_extent for s in sentinel_list])
    min_lon, _, min_lat, _ = np.min(swath_extents, axis=0)
    _, max_lon, _, max_lat = np.max(swath_extents, axis=0)
    return min_lon, max_lon, min_lat, max_lat


def total_swath_width_height(sentinel_list):
    """Width and height in deg across all swath areas in sentinel_list

    Width is longitude extent, height is latitude extent
    """
    left, right, bot, top = total_swath_extent(sentinel_list)
    return right - left, top - bot


def num_tiles(length, tile_size=0.5):
    """Given an edge length (of a swath), find how many tiles in a direction

    Examples:
    >>> num_tiles(1, tile_size=0.5)
    2
    >>> num_tiles(3.2, tile_size=0.5)
    6
    >>> num_tiles(3.3, tile_size=0.5)
    7
    """
    return np.round(length / tile_size)


def tile_dims(swath_length_width, tile_size=0.5):
    """Given a swath (length, width), find sizes of tiles

    Will try to divide into about 0.5 degree tiles, rounding up or down
    """
    swath_lw_arr = np.array(swath_length_width)
    num_tiles_arr = num_tiles(swath_lw_arr, tile_size)
    return swath_lw_arr / num_tiles_arr
