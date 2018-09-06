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

    Returns:
        tuple[float]: (min_lon, max_lon, min_lat, max_lat)
    """
    swath_extents = np.array([s.swath_extent for s in sentinel_list])
    min_lon, _, min_lat, _ = np.min(swath_extents, axis=0)
    max_lon, _, max_lat, _ = np.max(swath_extents, axis=0)
    return min_lon, max_lon, min_lat, max_lat
