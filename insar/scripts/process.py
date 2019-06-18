#!/usr/bin/env python
"""Runs all steps of interferogram processing

    Steps:
    1. Download precise orbits EOF files
    2. Create an upsampled DEM
    3. run sentinel_stack to produce .geo file for all sentinel .zips
    4. Record LOS ENU vector from midpoint of DEM
    5. Post processing for sentinel stack (igrams folder prep)
    6. create the sbas_list
    7. run ps_sbas_igrams.py
    8. run snaphu to unwrap all .int files
    9. Convert .int files and .unw files to .tif files
    10. Run an SBAS inversion to get the LOS deformation

"""
import math
import subprocess
import os
import glob
import numpy as np
from click import BadOptionUsage

import sardem
import eof
import apertools.los
import apertools.utils
from apertools.log import get_log, log_runtime
from apertools.utils import mkdir_p, force_symlink
from apertools.parsers import Sentinel
import insar.timeseries

logger = get_log()
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def download_eof(mission=None, date=None, **kwargs):
    """1. Download precision orbit files"""
    eof.download.main(mission=mission, date=date)


def create_dem(left_lon=None,
               top_lat=None,
               dlat=None,
               dlon=None,
               geojson=None,
               rate=1,
               data_source='NASA',
               **kwargs):
    """2. Download, upsample, and stich a DEM"""
    if not (left_lon and top_lat and dlon and dlat) and not geojson:
        raise BadOptionUsage("Need either lat/lon arguments or --geojson option"
                             " for create_dem step.")
    output_name = 'elevation.dem'
    logger.info("Running: sardem.dem:main")
    sardem.dem.main(
        left_lon=left_lon,
        top_lat=top_lat,
        dlon=dlon,
        dlat=dlat,
        geojson=geojson,
        data_source=data_source,
        rate=rate,
        output_name=output_name,
    )


def run_sentinel_stack(sentinel_path="~/sentinel/", unzip=True, **kwargs):
    """3. Create geocoded slcs as .geo files for each .zip file"""
    script_path = os.path.join(sentinel_path, "sentinel_stack.py")
    unzip_arg = "--no-unzip" if unzip is False else ''
    subprocess.check_call('/usr/bin/env python {} {}'.format(script_path, unzip_arg), shell=True)


def record_los_vectors(path=".", **kwargs):
    """4. With .geos processed, record the ENU LOS vector from DEM center to sat"""
    enu_coeffs = apertools.los.find_east_up_coeffs(path)
    np.save("los_enu_midpoint_vector.npy", enu_coeffs)


def _reorganize_files(new_dir="extra_files"):
    """Records current file names for Sentinel dir, renames to short names"""

    def _move_files(new_dir):
        # Save all sentinel_stack output to new_dir
        mkdir_p(new_dir)
        subprocess.call("mv ./* {}/".format(new_dir), shell=True)

    def _make_symlinks(geofiles):
        for geofile in geofiles:
            s = Sentinel(geofile)
            # Use just mission and date: S1A_20170101.geo
            new_name = "{}_{}".format(s.mission, s.date.strftime("%Y%m%d"))
            logger.info("Renaming {} to {}".format(geofile, new_name))
            force_symlink(geofile, new_name + ".geo")
            # also move corresponding orb timing file
            orbtiming_file = geofile.replace('geo', 'orbtiming')
            force_symlink(orbtiming_file, new_name + ".orbtiming")

    _move_files(new_dir)
    # Then bring back the useful ones to the cur dir as symlinks renamed
    geofiles = glob.glob(os.path.join(new_dir, "*.geo"))
    _make_symlinks(geofiles)

    # Move extra useful files back in main directory
    for fname in ('params', 'elevation.dem', 'elevation.dem.rsc'):
        os.symlink(os.path.join(new_dir, fname), os.path.join('.', fname))


def prep_igrams_dir(cleanup=False, **kwargs):
    """5. Reorganize and rename .geo files, stitches .geos, prepare for igrams"""

    def _find_num_geos():
        return len(glob.glob('./*.geo'))

    if cleanup:
        logger.info("Renaming .geo files, creating symlinks")
        new_dir = 'extra_files'
        _reorganize_files(new_dir)
        # For now, leave out the "bad_geo" making
        # apertools.utils.clean_files(".geo", path=".", zero_threshold=0.50, test=False)

        # Now stitch together duplicate dates of .geos
        apertools.utils.stitch_same_dates(geo_path="extra_files/", output_path=".")

    num_geos = _find_num_geos()
    if num_geos < 2:  # Can't make igrams
        logger.error("%s .geo file in current folder, can't form igram: exiting", num_geos)
        return 1

    mkdir_p('igrams')
    os.chdir('igrams')
    logger.info("Changed directory to %s", os.path.realpath(os.getcwd()))


def create_sbas_list(max_temporal=500, max_spatial=500, **kwargs):
    """6. run the sbas_list script

    Uses the outputs of the geo coded SLCS to find files with small baselines
    Searches one directory up from where script is run for .geo files
    """
    sbas_cmd = '/usr/bin/env python ~/sentinel/sbas_list.py {} {}'.format(max_temporal, max_spatial)
    logger.info(sbas_cmd)
    subprocess.check_call(sbas_cmd, shell=True)


def run_ps_sbas_igrams(rate=1, looks=None, **kwargs):
    """7. run the ps_sbas_igrams script"""

    def calc_sizes(rate, width, length):
        xsize = int(math.floor(width / rate) * rate)
        ysize = int(math.floor(length / rate) * rate)
        return (xsize, ysize)

    logger.info("Gathering file size info from elevation.dem.rsc")
    elevation_dem_rsc_file = '../elevation.dem.rsc'
    rsc_data = sardem.loading.load_dem_rsc(elevation_dem_rsc_file)
    xsize, ysize = calc_sizes(rate, rsc_data['width'], rsc_data['file_length'])

    # the "1 1" is xstart ystart
    # Default number of looks is the upsampling rate so that
    # the igram is the size of the original DEM (elevation_small.dem)
    looks = looks or rate
    logger.info("Running ps_sbas_igrams.py")
    ps_sbas_cmd = "/usr/bin/env python ~/sentinel/ps_sbas_igrams.py \
sbas_list {rsc_file} 1 1 {xsize} {ysize} {looks}".format(
        rsc_file=elevation_dem_rsc_file, xsize=xsize, ysize=ysize, looks=looks)
    logger.info(ps_sbas_cmd)
    subprocess.check_call(ps_sbas_cmd, shell=True)

    # Also create masks of invalid areas of igrams/.geos
    insar.timeseries.create_igram_masks('.', row_looks=looks, col_looks=looks)


def run_snaphu(lowpass=None, **kwargs):
    """8. run snaphu to unwrap all .int files

    Assumes we are in the directory with all .unw files
    """
    # TODO: probably shouldn't call these like this? idk alternative right now
    igram_rsc = sardem.loading.load_dem_rsc('dem.rsc')
    snaphu_script = os.path.join(SCRIPTS_DIR, 'run_snaphu.sh')
    snaphu_cmd = '{filepath} {width} {lowpass}'.format(
        filepath=snaphu_script, width=igram_rsc['width'], lowpass=lowpass)
    logger.info(snaphu_cmd)
    subprocess.check_call(snaphu_cmd, shell=True)


def convert_to_tif(max_height=None, **kwargs):
    """9. Convert igrams (.int) and snaphu outputs (.unw) to .tif files

    Assumes we are in the directory with all .int and .unw files
    """
    # Default name by ps_sbas_igrams
    igram_rsc = sardem.loading.load_dem_rsc('dem.rsc')
    # "shopt -s nullglob" skips the for-loop when nothing matches
    convert_cmd = """shopt -s nullglob; for i in ./*.int ; do dismphfile "$i" {igram_width} ; \
 mv dismph.tif `echo "$i" | sed 's/int$/tif/'` ; done""".format(igram_width=igram_rsc['width'])
    logger.info(convert_cmd)
    subprocess.check_call(convert_cmd, shell=True)

    snaphu_script = os.path.join(SCRIPTS_DIR, 'convert_snaphu.py')
    snaphu_cmd = 'python {filepath} --max-height {hgt}'.format(
        filepath=snaphu_script, hgt=max_height)
    logger.info(snaphu_cmd)
    subprocess.check_call(snaphu_cmd, shell=True)


def run_sbas_inversion(ref_row=None,
                       ref_col=None,
                       window=None,
                       alpha=0,
                       constant_vel=False,
                       difference=False,
                       deramp_order=1,
                       **kwargs):
    """10. Perofrm SBAS inversion, save the deformation as .npy

    Assumes we are in the directory with all .unw files"""
    igram_path = os.path.realpath(os.getcwd())
    geolist, phi_arr, deformation = insar.timeseries.run_inversion(
        igram_path,
        reference=(ref_row, ref_col),
        window=window,
        alpha=alpha,
        constant_vel=constant_vel,
        difference=difference,
        deramp_order=deramp_order,
        verbose=kwargs['verbose'])
    logger.info("Saving deformation.npy and geolist.npy")
    np.save('deformation.npy', deformation)
    np.save('geolist.npy', geolist)


# List of functions that run each step
STEPS = [
    download_eof,
    create_dem,
    run_sentinel_stack,
    record_los_vectors,
    prep_igrams_dir,
    create_sbas_list,
    run_ps_sbas_igrams,
    run_snaphu,
    convert_to_tif,
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

    # Use the --step option first, or else use the --start
    # Subtract 1 so that they are list indices, starting at 0
    step_list = [s - 1 for s in kwargs['step']] or range(kwargs['start'] - 1, len(STEPS))
    logger.info("Running steps %s", ','.join(str(s + 1) for s in step_list))
    for stepnum in step_list:
        curfunc = STEPS[stepnum]
        logger.info("Starting step %d: %s", stepnum + 1, curfunc.__name__)
        ret = curfunc(**kwargs)
        if ret:  # Option to have step give non-zero return to halt things
            return ret
