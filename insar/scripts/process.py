#!/usr/bin/env python
"""Runs all steps of interferogram processing

    TODO: allow ranges of steps like https://stackoverflow.com/a/4726287

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
import math
import sys
import subprocess
import os
import numpy as np

import insar
from insar.log import get_log, log_runtime
from insar.utils import mkdir_p

logger = get_log()


def download_eof(mission=None, date=None, **kwargs):
    """1. Download precision orbit files"""
    insar.eof.main(mission=mission, date=date)


def create_dem(geojson=None, rate=1, data_source='NASA', output='elevation.dem', **kwargs):
    """2. Download, upsample, and stich a DEM"""
    if not geojson:
        logger.error("For step 2: create_dem, --geojson is needed.")
        sys.exit(1)
    logger.info("Running: insar.dem:main")
    insar.dem.main(geojson, data_source, rate, output)


def run_sentinel_stack(**kwargs):
    """3. Create geocoded slcs as .geo files for each .zip file"""
    subprocess.check_call('/usr/bin/python ~/sentinel/sentinel_stack.py', shell=True)


def prep_igrams_dir(**kwargs):
    """4. prepare directory for igrams"""
    mkdir_p('igrams')
    os.chdir('igrams')
    logger.info("Changed directory to %s", os.path.realpath(os.getcwd()))


def create_sbas_list(max_temporal=500, max_spatial=500, **kwargs):
    """ 5.run the sbas_list script

    Uses the outputs of the geo coded SLCS to find files with small baselines"""

    sbas_cmd = '/usr/bin/python ~/sentinel/sbas_list.py {} {}'.format(max_temporal, max_spatial)
    logger.info(sbas_cmd)
    subprocess.check_call(sbas_cmd, shell=True)


def run_ps_sbas_igrams(rate=1, looks=None, **kwargs):
    """6. run the ps_sbas_igrams script"""

    def calc_sizes(rate, width, length):
        xsize = int(math.floor(width / rate) * rate)
        ysize = int(math.floor(length / rate) * rate)
        return (xsize, ysize)

    logger.info("Gathering file size info from elevation.dem.rsc")
    elevation_dem_rsc_file = '../elevation.dem.rsc'
    rsc_data = insar.sario.load_dem_rsc(elevation_dem_rsc_file)
    xsize, ysize = calc_sizes(rate, rsc_data['WIDTH'], rsc_data['FILE_LENGTH'])

    # the "1 1" is xstart ystart
    # Default number of looks is the upsampling rate so that
    # the igram is the size of the original DEM (elevation_small.dem)
    looks = looks or rate
    logger.info("Running ps_sbas_igrams.py")
    ps_sbas_cmd = "/usr/bin/python ~/sentinel/ps_sbas_igrams.py sbas_list {rsc_file} 1 1 {xsize} {ysize} {looks}".format(
        rsc_file=elevation_dem_rsc_file, xsize=xsize, ysize=ysize, looks=looks)
    logger.info(ps_sbas_cmd)
    subprocess.check_call(ps_sbas_cmd, shell=True)


def convert_int_tif(**kwargs):
    # TODO: Make this into the script like the convert_snaphu

    # Default name by ps_sbas_igrams
    igram_rsc = insar.sario.load_dem_rsc('dem.rsc')
    convert1 = """for i in ./*.int ; do dismphfile "$i" {igram_width} ; mv dismph.tif `echo "$i" | sed 's/int$/tif/'` ; done""".format(
        igram_width=igram_rsc['WIDTH'])
    subprocess.check_call(convert1, shell=True)


def run_snaphu(lowpass=None):
    """8. run snaphu to unwrap all .int files

    Assumes we are in the directory with all .unw files
    """
    # TODO: probably shouldn't call these like this? idk alternative right now
    igram_rsc = insar.sario.load_dem_rsc('dem.rsc')
    subprocess.call(
        '~/repos/insar/scripts/run_snaphu.sh {width} {lowpass}'.format(
            width=igram_rsc['WIDTH'], lowpass=lowpass),
        shell=True)


def convert_snaphu_tif(max_height=None, **kwargs):
    """9. Convert snaphu outputs to .tif files

    Assumes we are in the directory with all .unw files
    """
    subprocess.call(
        '~/repos/insar/scripts/convert_snaphu.py --max-height {}'.format(max_height), shell=True)


def run_sbas_inversion(ref_row=None, ref_col=None, **kwargs):
    """10. Perofrm SBAS inversion, save the deformation as .npy

    Assumes we are in the directory with all .unw files"""
    if not ref_row or not ref_col:
        logger.warning("--ref-row and --ref-col required for run_sbas_inversion: skipping.")
        return

    igram_path = os.path.realpath(os.getcwd())
    geolist, phi_arr, deformation, varr, unw_stack = insar.timeseries.run_inversion(
        igram_path, reference=(ref_row, ref_col))
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


@log_runtime
def main(working_dir, kwargs):
    # TODO: maybe let user specify individual steps?
    if working_dir != ".":
        logger.info("Changing directory to {}".format(working_dir))
        os.chdir(working_dir)

    for stepnum in range(kwargs['step'] - 1, len(STEPS)):
        curfunc = STEPS[stepnum]
        logger.info("Starting step %d: %s", stepnum + 1, curfunc.__name__)
        curfunc(**kwargs)
