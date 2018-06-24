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
    10. Run an SBAS inversion to get the LOS deformation

"""

import argparse
import math
import sys
import subprocess
import os
from os.path import abspath, dirname
import numpy as np
try:
    import insar
except ImportError:  # add root to pythonpath if import fails
    sys.path.insert(0, dirname(dirname(abspath(__file__))))

from insar import sario, timeseries
from insar.log import get_log, log_runtime
from insar.utils import mkdir_p, which

logger = get_log()


def download_eof(*a):
    """1. Download precision orbit files"""
    download_eof_exe = which('download-eofs')
    subprocess.check_call(download_eof_exe, shell=True)


def create_dem(args):
    """2. Download, upsample, and stich a DEM"""
    create_dem_exe = which('create-dem')
    if not args.geojson:
        logger.error("For step 2: create_dem, --geojson is needed.")
        sys.exit(1)
    dem_cmd = '{} -g {} -r {}'.format(create_dem_exe, args.geojson, args.rate)
    logger.info("Running: %s", dem_cmd)
    subprocess.check_call(dem_cmd, shell=True)


def run_sentinel_stack(*a):
    """3. Create geocoded slcs as .geo files for each .zip file"""
    subprocess.call('~/sentinel/sentinel_stack.py', shell=True)


def prep_igrams_dir(*a):
    """4. prepare directory for igrams"""
    mkdir_p('igrams')
    os.chdir('igrams')
    logger.info("Changed directory to %s", os.path.realpath(os.getcwd()))


def create_sbas_list(args):
    """ 5.run the sbas_list script

    Uses the outputs of the geo coded SLCS to find files with small baselines"""

    sbas_cmd = '~/sentinel/sbas_list.py {} {}'.format(args.max_temporal, args.max_spatial)
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
    rsc_data = sario.load_dem_rsc(elevation_dem_rsc_file)
    xsize, ysize = calc_sizes(args.rate, rsc_data['WIDTH'], rsc_data['FILE_LENGTH'])

    # the "1 1" is xstart ystart
    # Default number of looks is the upsampling rate so that
    # the igram is the size of the original DEM (elevation_small.dem)
    looks = args.looks or args.rate
    logger.info("Running ps_sbas_igrams.py")
    ps_sbas_cmd = "~/sentinel/ps_sbas_igrams.py sbas_list {rsc_file} 1 1 {xsize} {ysize} {looks}".format(
        rsc_file=elevation_dem_rsc_file, xsize=xsize, ysize=ysize, looks=looks)
    logger.info(ps_sbas_cmd)
    subprocess.check_call(ps_sbas_cmd, shell=True)


def convert_int_tif(*a):
    # TODO: Make this into the script like the convert_snaphu

    # Default name by ps_sbas_igrams
    igram_rsc = sario.load_dem_rsc('dem.rsc')
    convert1 = "for i in *.int ; do dismphfile $i {igram_width} ; mv dismph.tif `echo $i | sed 's/int$/tif/'` ; done".format(
        igram_width=igram_rsc['WIDTH'])
    subprocess.call(convert1, shell=True)


def run_snaphu(args):
    """8. run snaphu to unwrap all .int files

    Assumes we are in the directory with all .unw files
    """
    # TODO: probably shouldn't call these like this? idk alternative right now
    igram_rsc = sario.load_dem_rsc('dem.rsc')
    subprocess.call(
        '~/repos/insar/scripts/run_snaphu.sh {width} {lowpass}'.format(
            width=igram_rsc['WIDTH'], lowpass=args.lowpass),
        shell=True)


def convert_snaphu_tif(args):
    """9. Convert snaphu outputs to .tif files

    Assumes we are in the directory with all .unw files
    """
    subprocess.call(
        '~/repos/insar/scripts/convert_snaphu.py --max-height {}'.format(args.max_height),
        shell=True)


def run_sbas_inversion(args):
    """10. Perofrm SBAS inversion, save the deformation as .npy

    Assumes we are in the directory with all .unw files"""
    if not args.ref_row or not args.ref_col:
        logger.warning("--ref-row and --ref-col required for run_sbas_inversion: skipping.")
        return

    igram_path = os.path.realpath(os.getcwd())
    geolist, phi_arr, deformation, varr, unw_stack = timeseries.run_inversion(
        igram_path, reference=(args.ref_row, args.ref_col))
    logger.info("Saving deformation, velocity_array, and geolist")
    np.save('deformation.npy', deformation)
    np.save('velocity_array.npy', varr)
    np.save('geolist.npy', geolist)


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
    run_sbas_inversion,
]
# Form string for help function "1:download_eof,2:..."
STEP_LIST = ',\n'.join("%d:%s" % (num, func.__name__) for (num, func) in enumerate(STEPS, start=1))


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--working-dir", "-d", default=".", help="Directory where sentinel .zip are located")
    parser.add_argument("--geojson", "-g", help="File containing the geojson object for DEM bounds")
    parser.add_argument(
        "--rate",
        "-r",
        type=int,
        default=1,
        help="Rate at which to upsample DEM (default=1, no upsampling)")
    parser.add_argument(
        "--max-height",
        "-m",
        default=10,
        help="Maximum height/max absolute phase for converting .unw files to .tif"
        "(used for contour_interval option to dishgt)")
    parser.add_argument(
        "--step",
        "-s",
        type=int,
        help="Choose which step to start on. Steps: {}".format(STEP_LIST),
        choices=range(1,
                      len(STEPS) + 1),
        default=1)
    parser.add_argument(
        "--max-temporal",
        type=int,
        default=500,
        help="Maximum temporal baseline for igrams (fed to sbas_list)")
    parser.add_argument(
        "--max-spatial",
        type=int,
        default=500,
        help="Maximum spatial baseline for igrams (fed to sbas_list)")
    parser.add_argument(
        "--looks",
        type=int,
        help="Number of looks to perform on .geo files to shrink down .int, "
        "Default is the upsampling rate, makes the igram size=original DEM size")
    parser.add_argument(
        "--lowpass",
        type=int,
        default=1,
        help="Size of lowpass filter to use on igrams before unwrapping")
    parser.add_argument(
        "--ref-row",
        type=int,
        help="Row number of pixel to use as unwrapping reference for SBAS inversion")
    parser.add_argument(
        "--ref-col",
        type=int,
        help="Column number of pixel to use as unwrapping reference for SBAS inversion")
    return parser.parse_args()


@log_runtime
def main():
    args = get_cli_args()

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
