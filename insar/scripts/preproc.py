import os
import glob
import subprocess
import insar.parsers
import insar.tile
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


def make_tile_geojsons(data_path, path_num=None, tile_size=0.5, overlap=0.1):
    sentinel_list = find_sentinels(data_path, path_num)
    total_extent = insar.tile.total_swath_extent(sentinel_list)
    tiles, (height, width) = insar.tile.make_tiles(
        total_extent, tile_size=tile_size, overlap=overlap)
    gj_list = [insar.tile.tile_to_geojson(t, height, width) for t in tiles]
    tilename_list = [t.tilename for t in tiles]
    return list(zip(tilename_list, gj_list))
