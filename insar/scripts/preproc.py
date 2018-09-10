import os
import glob
import json
import subprocess

import insar.utils
import insar.tile
import eof
import sardem
from insar.log import get_log

logger = get_log()


def unzip_sentinel_files(path="."):
    """Function to find all .zips and unzip them, skipping overwrites"""
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


def create_tile_directories(data_path, path_num=None, tile_size=0.5, overlap=0.1, verbose=False):
    """Use make_tiles to create a directory structure

    Populates the current directory with dirs and .geojson files (e.g.):
    N28.8W101.6
    N28.8W101.6.geojson
    N28.8W102.0
    N28.8W102.0.geojson
    ...
    """

    data_path = os.path.abspath(data_path)

    def _write_geojson(filename, geojson):
        with open(filename, 'w') as f:
            json.dump(geojson, f, indent=2)

    sentinel_list = insar.tile.find_unzipped_sentinels(data_path, path_num)
    if not sentinel_list:
        logger.error("No sentinel products found in %s for path_num %s", data_path, path_num)
        return [], []

    tile_grid = insar.tile.create_tiles(
        sentinel_list=sentinel_list, tile_size=tile_size, overlap=overlap, verbose=verbose)

    # new_dirs = []
    for tile in tile_grid:
        insar.utils.mkdir_p(tile.name)
        filename = os.path.join(tile.name, '{}.geojson'.format(tile.name))
        _write_geojson(filename, tile.geojson)
        # new_dirs.append(tile.name)
        # Enter the new directory, link to sentinels, then back out
        # os.chdir(name)
        symlink_sentinels(tile, sentinel_list, verbose=verbose)
        # os.chdir('..')

    return sentinel_list, tile_grid


def symlink_sentinels(tile, sentinel_list, verbose=False):
    """Create symlinks from Tile.name to the new directory tile.name"""
    for s in sentinel_list:
        if tile.overlaps_with(s):
            _, fname = os.path.split(s.filename)
            dest = os.path.join(tile.name, fname)
            if verbose:
                logger.info("symlinking %s to %s", s.filename, dest)
            insar.utils.force_symlink(s.filename, dest)
