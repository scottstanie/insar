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
import os
from os.path import abspath, dirname
try:
    import insar
except ImportError:  # add root to pythonpath if import fails
    sys.path.insert(0, dirname(dirname(abspath(__file__))))

import insar.sario
from insar.log import get_log
from insar.utils import mkdir_p, which

logger = get_log()


def download_eof(*a):
    """1. Download precision orbit files"""
    download_eof_exe = which('download-eofs')
    subprocess.check_call(download_eof_exe, shell=True)


def create_dem(args):
    """2. Download, upsample, and stich a DEM"""
    create_dem_exe = which('create-dem')
    dem_cmd = '{} -g {} -r {}'.format(create_dem_exe, args.geojson, args.rate)
    logger.info("Running: %s", dem_cmd)
    subprocess.check_call(dem_cmd, shell=True)


def run_sentinel_stack(*a):
    """3. Create geocoded slcs as .geo files for each .zip file"""
    logger.info("Starting sentinel_stack.py")
    subprocess.call('~/sentinel/sentinel_stack.py', shell=True)


def prep_igrams_dir(*a):
    """4. prepare directory for igrams"""
    logger.info("Making igrams directory and moving into igrams")
    mkdir_p('igrams')
    os.chdir('igrams')
    logger.info("Now inside %s", os.path.realpath(os.getcwd()))


def create_sbas_list(*a):
    """ 5.run the sbas_list script

    Uses the outputs of the geo coded SLCS to find files with small baselines"""

    logger.info("Creating sbas_list")
    max_time = '500'
    max_spatial = '500'
    sbas_cmd = '~/sentinel/sbas_list.py {} {}'.format(max_time, max_spatial)
    logger.info(sbas_cmd)
    subprocess.call(sbas_cmd, shell=True)


def run_ps_sbas_igrams(args):
    """6. run the ps_sbas_igrams script"""

    def calc_sizes(rate, width, length):
        xsize = int(math.floor(width / rate) * rate)
        ysize = int(math.floor(length / rate) * rate)
        return (xsize, ysize)

    logger.info("Gathering file size info from elevation.dem.rsc")
    elevation_dem_rsc_file = '../elevation.dem.rsc'
    rsc_data = insar.sario.load_dem_rsc(elevation_dem_rsc_file)
    xsize, ysize = calc_sizes(args.rate, rsc_data['WIDTH'], rsc_data['FILE_LENGTH'])

    # the "1 1" is xstart ystart
    # We are using the upsampling rate as the number of looks so that
    # the igram is the size of the original DEM (elevation_small.dem)
    logger.info("Running ps_sbas_igrams.py")
    ps_sbas_cmd = "~/sentinel/ps_sbas_igrams.py sbas_list {rsc_file} 1 1 {xsize} {ysize} {looks}".format(
        rsc_file=elevation_dem_rsc_file, xsize=xsize, ysize=ysize, looks=args.rate)
    logger.info(ps_sbas_cmd)
    subprocess.check_call(ps_sbas_cmd, shell=True)


def convert_int_tif(*a):
    # TODO: Make this into the script like the convert_snaphu

    # Default name by ps_sbas_igrams
    igram_rsc = insar.sario.load_dem_rsc('dem.rsc')
    convert1 = "for i in *.int ; do dismphfile $i {igram_width} ; mv dismph.tif `echo $i | sed 's/int$/tif/'` ; done".format(
        igram_width=igram_rsc['WIDTH'])
    subprocess.call(convert1, shell=True)


def run_snaphu(*a):
    """8. run snaphu to unwrap all .int files"""
    # TODO: probably shouldn't call these like this? idk alternative right now
    igram_rsc = insar.sario.load_dem_rsc('dem.rsc')
    subprocess.call('~/repos/insar/scripts/run_snaphu.sh {}'.format(igram_rsc['WIDTH']), shell=True)


def convert_snaphu_tif(args):
    """9. Convert snaphu outputs to .tif files"""
    subprocess.call(
        '~/repos/insar/scripts/convert_snaphu.py --max-height {}'.format(args.max_height),
        shell=True)


# List of functions that run each step
STEPS = [
    download_eof,
    create_dem,
    run_sentinel_stack,
    prep_igrams_dir,
    create_sbas_list,
    run_ps_sbas_igrams,
    convert_int_tif,
    run_snaphu,
    convert_snaphu_tif,
]
STEP_LIST = ',\n'.join("%d:%s" % (num, func.__name__) for (num, func) in enumerate(STEPS, start=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--working-dir", "-d", default=".", help="Directory where sentinel .zip are located")
    parser.add_argument(
        "--geojson", "-g", required=True, help="File containing the geojson object for DEM bounds")
    parser.add_argument(
        "--rate",
        "-r",
        type=int,
        required=True,
        help="Rate at which to upsample DEM (default=1, no upsampling)")
    parser.add_argument(
        "--max-height",
        "-m",
        default=10,
        help="Maximum height/max absolute phase in .unw files "
        "(used for contour_interval option to dishgt)")
    parser.add_argument(
        "--step",
        "-s",
        type=int,
        help="Choose which step to start on. Steps: {}".format(STEP_LIST),
        choices=range(1,
                      len(STEPS) + 1),
        default=1)
    args = parser.parse_args()

    os.chdir(args.working_dir)
    dir_path = os.path.realpath(args.working_dir)  # save for later reference
    logger.info("Running scripts in {}".format(dir_path))

    # TODO: maybe let user specify individual steps?
    for stepnum in range(args.step - 1, len(STEPS)):
        curfunc = STEPS[stepnum]
        logger.info("Starting step %d: %s", stepnum + 1, curfunc.__name__)
        curfunc(args)


if __name__ == '__main__':
    main()
