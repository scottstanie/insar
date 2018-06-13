#!/usr/bin/env python
"""Runs all steps of interferogram processing

    Usage: convert_snaphu.py --geojson "/path/to/dem.geojson" --rate 10 --max-height 10

    geojson used for bounding box of DEM
    rate passed to dem upsampling routine
    max_height passed to snaphu phase unwrapping

    Steps:
    1. Download precise orbits EOF files
    2. Create an upsampled DEM
    3. run sentinel_stack to produce .geo file for all sentinel .zips
    4. Post processing for sentinel stack (igrams folder prep)
    5. create the sbas_list
    6. run ps_sbas_igrams.py
    7. convert .int files to .tif
    8. run snaphu to unwrap all .int files
    9. Convert snaphu outputs to .tif files

"""

import argparse
import math
import sys
import subprocess
import errno
import os
from os.path import abspath, dirname
try:
    import insar
except ImportError:  # add root to pythonpath if import fails
    sys.path.insert(0, dirname(dirname(abspath(__file__))))

import insar.sario
from insar.log import get_log

logger = get_log()


def _mkdir_p(path):
    """Emulates bash `mkdir -p`, in python style
    Used for igrams directory creation
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def calc_sizes(rate, width, length):
    xsize = math.floor(width / rate) * rate
    ysize = math.floor(length / rate) * rate
    return (xsize, ysize)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geojson", "-g", required=True, help="File containing the geojson object for DEM bounds")
    parser.add_argument(
        "--rate",
        "-r",
        required=True,
        help="Rate at which to upsample DEM (default=1, no upsampling)")
    parser.add_argument(
        "--max-height",
        "-m",
        default=10,
        help="Maximum height/max absolute phase in .unw files "
        "(used for contour_interval option to dishgt)")
    args = parser.parse_args()

    # 1. Download precision orbit files
    subprocess.check_call(['download-eofs'])

    # 2. Create an upsampled DEM
    subprocess.check_call(['create-dem', '-g', args.geojson, '-r', args.rate])

    # 3. Produce a .geo file for each .zipped SLC
    logger.info("Starting sentinel_stack.py")
    subprocess.check_call(['~/sentinel/sentinel_stack.py'])

    # 4. Post processing for sentinel stack
    logger.info("Making igrams directory and moving into igrams")
    _mkdir_p('igrams')
    os.chdir('igrams')

    # 5. Sbas_list
    logger.info("Creating sbas_list")
    max_time = 500
    max_spatial = 500
    subprocess.check_call(['~/sentinel/sbas_list.py', max_time, max_spatial])

    logger.info("Gathering file size info from elevation.dem.rsc")
    elevation_dem_rsc_file = '../elevation.dem.rsc'
    rsc_data = insar.sario.load_dem_rsc(elevation_dem_rsc_file)
    xsize, ysize = calc_sizes(args.rate, rsc_data['WIDTH'], rsc_data['LENGTH'])

    # 6. ps_sbas_igrams
    # the "1 1" is xstart ystart
    # We are using the upsampling rate as the number of looks so that
    # the igram is the size of the original DEM (elevation_small.dem)
    logger.info("Running ps_sbas_igrams.py")
    ps_sbas_cmd = "~/sentinel/ps_sbas_igrams.py sbas_list {rsc_file} 1 1 {xsize} {ysize} {looks}".format(
        rsc_file=elevation_dem_rsc_file, xsize=xsize, ysize=ysize, looks=args.rate)
    logger.info(ps_sbas_cmd)
    subprocess.check_call(ps_sbas_cmd.split(' '))

    # Default name by ps_sbas_igrams
    igram_rsc = insar.sario.load_dem_rsc('dem.rsc')
    # 7. convert .int files to .tif
    # TODO: Make this into the script like the convert_snaphu
    convert1 = "for i in *.int ; do dismphfile $i {igram_width} ; mv dismph.tif `echo $i | sed 's/int$/tif/'` ; done".format(
        igram_width=igram_rsc['WIDTH'])
    subprocess.call(convert1, shell=True)

    # 8. run snaphu to unwrap all .int files
    # TODO: probably can't call these like this
    subprocess.call('~/repos/insar/scripts/run_snaphu.sh {}'.format(igram_rsc['WIDTH']), shell=True)

    # 9. Convert snaphu outputs to .tif files
    subprocess.call(
        '~/repos/insar/scripts/convert_snaphu.sh --max-height {}'.format(args.max_height),
        shell=True)


if __name__ == '__main__':
    main()
