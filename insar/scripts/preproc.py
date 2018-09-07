import os
import json
import subprocess

import insar.utils
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


def create_tile_directories(data_path, path_num=None, tile_size=0.5, overlap=0.1):
    """Use make_tiles to create a directory structure

    Populates the current directory with dirs and .geojson files (e.g.):
    N28.8W101.6
    N28.8W101.6.geojson
    N28.8W102.0
    N28.8W102.0.geojson
    ...
    """

    def _write_geojson(tilename, geojson):
        with open('{}.geojson'.format(tilename), 'w') as f:
            json.dump(geojson, f)

    sentinel_list = insar.tile.find_sentinels(data_path, path_num)
    tile_list = insar.tile.make_tiles(
        sentinel_list=sentinel_list, tile_size=tile_size, overlap=overlap)

    # new_dirs = []
    for tile in tile_list:
        _write_geojson(tile.tilename, tile.geojson)
        insar.utils.mkdir_p(tile.tilename)
        # new_dirs.append(tile.tilename)
        # Enter the new directory, link to sentinels, then back out
        # os.chdir(tilename)
        link_sentinels(tile, sentinel_list)
        # os.chdir('..')


def link_sentinels(tile, sentinel_list):
    for s in sentinel_list:
        if tile.overlaps_with(s):
            insar.utils.force_symlink(s.filename, tile.filename)
