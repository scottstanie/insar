#!/usr/bin/env python
"""Runs all steps of interferogram processing

    Steps:
    1. create a (upsampled) DEM over area of interest
    2. download sentinel files and precise orbits EOF
    3. run sentinel_stack to produce .geo file for all sentinel .zips
    4. Post processing for sentinel stack (igrams folder prep)
    5. create the sbas_list
    6. run ps_sbas_igrams.py
    7. create map of LOS ENU vectors
    8. run snaphu to unwrap all .int files
    9. convert .int files and .unw files to .tif files
    10. run an SBAS inversion to get the LOS deformation

"""
import math
import subprocess
import os
import glob
from multiprocessing import cpu_count

# import numpy as np
# from click import BadOptionUsage

# import apertools.los

from apertools.log import get_log, log_runtime
from apertools.utils import mkdir_p, force_symlink
from apertools.parsers import Sentinel

logger = get_log()
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SENTINEL_SLC_PATH = os.path.expanduser("~/sentinel/")
SENTINEL_RAW_PATH = os.path.expanduser("~/sentinel_l0/")
LOS_PATH = os.path.expanduser("~/repos/insar-los/")
# TODO: Make DEM is 1, download data is 2, then the sentinel raw script gest EOF already


def _log_and_run(cmd, check=True, shell=True):
    logger.info("Running command:")
    logger.info(cmd)
    if check:
        return subprocess.check_call(cmd, shell=shell)
    else:
        return subprocess.run(cmd, shell=shell)


def create_dem(
    left_lon=None,
    top_lat=None,
    dlat=None,
    dlon=None,
    geojson=None,
    xrate=1,
    yrate=2,
    data_source="NASA",
    **kwargs,
):
    """1. Download, upsample, and stich a DEM"""
    # TODO:
    _log_and_run("createdem")
    # output_name = 'elevation.dem'
    # logger.info("Running: sardem.dem:main")
    # sardem.dem.main(
    #     left_lon=left_lon,
    #     top_lat=top_lat,
    #     dlon=dlon,
    #     dlat=dlat,
    #     geojson=geojson,
    #     data_source=data_source,
    #     xrate=xrate,
    #     yrate=yrate,
    #     output_name=output_name,
    # )


def download_data(mission=None, date=None, **kwargs):
    """2. Download data from ASF precision orbit files"""
    # TODO
    _log_and_run("asfdownload")
    pass


def _get_product_type(path):
    # check if we have RAW or SLC type files in `path`
    p = None
    for f in glob.glob(os.path.join(path, "*")):
        try:
            p = Sentinel(f)
            break
        except ValueError:
            continue
    logger.info("Found %s product level sentinel files." % p.product_type)
    if p.product_type not in ("RAW", "SLC"):
        raise ValueError(
            "Unknown sentinel product type %s of %s" % (p.product_type, p.filename)
        )
    return p.product_type


def run_sentinel_stack(unzip=True, product_type="", gpu=False, **kwargs):
    """3. Create geocoded slcs as .geo files for each .zip file"""
    # Download all EOF files
    _log_and_run("eof")

    if not product_type:
        product_type = _get_product_type("./")

    if product_type == "RAW":
        script_path = os.path.join(SENTINEL_RAW_PATH, "sentinel_stack.py")
    elif product_type == "SLC":
        script_path = os.path.join(SENTINEL_SLC_PATH, "sentinel_stack.py")

    unzip_arg = "" if unzip else "--no-unzip"
    gpu_arg = "--gpu" if (gpu and product_type == "RAW") else ""
    cmd = "python {} {} {}".format(script_path, unzip_arg, gpu_arg)
    _log_and_run(cmd)
    for f in glob.glob("S*.geo"):
        # make symlinks of rsc file for loading

        force_symlink("elevation.dem.rsc", f + ".rsc")


def _make_symlinks(geofiles):
    for geofile in geofiles:
        s = Sentinel(geofile)
        # Use just mission and date: S1A_20170101.geo
        new_name = "{}_{}".format(s.mission, s.date.strftime("%Y%m%d"))
        logger.info("Renaming {} to {}".format(geofile, new_name))
        try:
            force_symlink(geofile, new_name + ".geo")
        except:
            logger.info("{} already exists: skipping".format(new_name))
        # move corresponding orb timing file
        orbtiming_file = geofile.replace("geo", "orbtiming")
        try:
            force_symlink(orbtiming_file, new_name + ".orbtiming")
        except ValueError as e:
            logger.info(f"{new_name} already exists: skipping ({e})")


def _cleanup_bad_dates(new_dir="extra_files", bad_dir_name="bad_files"):
    """Moves dates with missing data to separate folder"""
    from apertools import stitching, utils

    mkdir_p(bad_dir_name)
    with utils.chdir_then_revert(new_dir):
        bad_dates = stitching.find_safes_with_missing_data(".", "../elevation.dem")
        for d in bad_dates:
            dstr = d.strftime("%Y%m%d")
            for f in glob.glob(f"*{dstr}*"):
                logger.info("Bad date: Moving {} to {}".format(f, bad_dir_name))
                os.rename(f, os.path.join(bad_dir_name, dstr))


def prep_igrams_dir(cleanup=False, **kwargs):
    """4. Reorganize and rename .geo files, stitches .geos, prepare for igrams"""
    from apertools import stitching

    new_dir = "extra_files"
    if cleanup:
        logger.info("Renaming .geo files, creating symlinks")
        if os.path.exists(new_dir):
            logger.info("%s exists already, skipping reorganize files", new_dir)
        else:
            # Save all sentinel_stack output to new_dir
            mkdir_p(new_dir)
            _cleanup_bad_dates(new_dir)
            subprocess.call("mv ./* {}/".format(new_dir), shell=True)

    # Then bring back the useful ones to the cur dir as symlinks renamed
    geofiles = glob.glob(os.path.join(new_dir, "*.geo"))
    _make_symlinks(geofiles)

    # Move extra useful files back in main directory
    for fname in ("params", "elevation.dem", "elevation.dem.rsc"):
        try:
            src, dest = os.path.join(new_dir, fname), os.path.join(".", fname)
            # force_symlink(src, dest)
            subprocess.run(f"cp {src} {dest}", shell=True)
        except ValueError as e:
            logger.info(f"{dest} already exists: skipping ({e})")

    for geofile in geofiles:
        try:
            # force_symlink("elevation.dem.rsc", geofile + ".rsc")
            subprocess.run(f"cp elevation.dem.rsc {geofile}.rsc", shell=True)
        except ValueError as e:
            logger.info(f"{geofile + '.rsc'} already exists: skipping ({e})")

    # Now stitch together duplicate dates of .geos
    stitching.stitch_same_dates(
        geo_path="extra_files/", output_path=".", overwrite=False
    )

    num_geos = len(glob.glob("./*.geo"))
    if num_geos < 2:  # Can't make igrams
        logger.error(
            "%s .geo file in current folder, can't form igram: exiting", num_geos
        )
        return 1

    # Make vrts of files
    cmd = "aper save-vrt --rsc-file elevation.dem.rsc *geo"
    _log_and_run(cmd)

    mkdir_p("igrams")
    os.chdir("igrams")
    logger.info("Changed directory to %s", os.path.realpath(os.getcwd()))


def create_sbas_list(max_temporal=500, max_spatial=500, **kwargs):
    """5. run the sbas_list script

    Uses the outputs of the geo coded SLCS to find files with small baselines
    Searches one directory up from where script is run for .geo files
    """
    sbas_cmd = "/usr/bin/env python ~/sentinel/sbas_list.py {} {}".format(
        max_temporal, max_spatial
    )
    _log_and_run(sbas_cmd)


def run_ps_sbas_igrams(xrate=1, yrate=1, xlooks=None, ylooks=None, **kwargs):
    """6. run the ps_sbas_igrams script"""
    import apertools.sario

    def calc_sizes(xrate, yrate, width, length):
        xsize = int(math.floor(width / xrate) * xrate)
        ysize = int(math.floor(length / yrate) * yrate)
        return (xsize, ysize)

    logger.info("Gathering file size info from elevation.dem.rsc")
    elevation_dem_rsc_file = "../elevation.dem.rsc"
    rsc_data = apertools.sario.load(elevation_dem_rsc_file)
    xsize, ysize = calc_sizes(xrate, yrate, rsc_data["width"], rsc_data["file_length"])

    # the "1 1" is xstart ystart
    # Default number of looks is the upsampling rate so that
    # the igram is the size of the original DEM (elevation_small.dem)
    xlooks = xlooks or xrate
    ylooks = ylooks or yrate
    logger.info("Running ps_sbas_igrams.py")
    ps_sbas_cmd = "/usr/bin/env python ~/sentinel/ps_sbas_igrams.py \
sbas_list {rsc_file} 1 1 {xsize} {ysize} {xlooks} {ylooks}".format(
        rsc_file=elevation_dem_rsc_file,
        xsize=xsize,
        ysize=ysize,
        xlooks=xlooks,
        ylooks=ylooks,
    )
    _log_and_run(ps_sbas_cmd)

    # Also create masks of invalid areas of igrams/.geos
    # logger.info("Making stacks for new igrams, overwriting old mask file")
    # insar.prepare.create_mask_stacks(igram_path='.', overwrite=True)

    # Uses the computed mask areas to set the .int and .cc bad values to 0
    # (since they are non-zero from FFT smearing rows)

    # insar.prepare.zero_masked_areas(igram_path='.', mask_filename=mask_filename, verbose=True)
    # cmd = "julia --start=no /home/scott/repos/InsarTimeseries.jl/src/runprepare.jl --zero "
    # logger.info(cmd)
    # subprocess.check_call(cmd, shell=True)


def run_form_igrams(xlooks=1, ylooks=1, **kwargs):
    from insar import form_igrams

    form_igrams.create_igrams(ylooks, xlooks)


def record_los_vectors(path=".", **kwargs):
    """7. With .geos processed, record the ENU LOS vector from DEM center to sat
    for the igrams DEM grid"""
    # Copy the DEM, downlooked
    _log_and_run("aper looked-dem", check=False)
    cmd = os.path.join(LOS_PATH, "create_los_map.py")
    sent_file = glob.glob("../extra_files/*.SAFE")[0]
    cmd += " --dem elevation_looked.dem --sentinel-file {}".format(sent_file)
    print("Creating LOS maps:")
    _log_and_run(cmd, check=False)
    subprocess.run(cmd, check=True, shell=True)

    # enu_coeffs = apertools.los.find_east_up_coeffs(path)
    # np.save("los_enu_midpoint_vector.npy", enu_coeffs)


def run_snaphu(max_jobs=None, **kwargs):
    """8. run snaphu to unwrap all .int files

    Assumes we are in the directory with all .unw files
    """
    import apertools.sario

    igram_rsc = apertools.sario.load("dem.rsc")
    width = igram_rsc["width"]

    snaphu_script = os.path.join(SCRIPTS_DIR, "run_snaphu.py")
    snaphu_cmd = f"{snaphu_script} --path . --ext-cor '.cc' --cols {width} "
    if max_jobs is not None:
        snaphu_cmd += f" --max-jobs {max_jobs}"
    _log_and_run(snaphu_cmd)


def convert_to_tif(max_height=None, max_jobs=None, **kwargs):
    """9. Convert igrams (.int) and snaphu outputs (.unw) to .tif files

    Assumes we are in the directory with all .int and .unw files
    Also adds .rsc files for all .int and .unw
    """
    if max_jobs is None:
        # TODO:to i really care about this
        max_jobs = cpu_count()

    add_int_rsc = """find . -name "*.int" -print0 | \
xargs -0 -n1 -I{} --max-procs=50 cp dem.rsc {}.rsc """
    _log_and_run(add_int_rsc, check=False)

    add_cc_rsc = add_int_rsc.replace(".int", ".cc")
    _log_and_run(add_cc_rsc, check=False)

    add_unw_rsc = add_int_rsc.replace(".int", ".unw")
    _log_and_run(add_unw_rsc, check=False)

    logger.info("SKIPPING CONVERT TO TIF")
    return
    # Default name by ps_sbas_igrams
    igram_rsc = apertools.sario.load("dem.rsc")
    # "shopt -s nullglob" skips the for-loop when nothing matches
    convert_ints = """find . -name "*.int" -print0 | \
xargs -0 -n1 -I{} --max-procs=50 dismphfile {} %s """ % (
        igram_rsc["width"]
    )
    _log_and_run(convert_ints)

    convert_unws = """find . -name "*.unw" -print0 | \
xargs -0 -n1 -I{} --max-procs=50 dishgtfile {} %s 1 100000 %s """ % (
        igram_rsc["width"],
        max_height,
    )
    # snaphu_script = os.path.join(SCRIPTS_DIR, 'convert_snaphu.py')
    # convert_unws = 'python {filepath} --max-height {hgt}'.format(filepath=snaphu_script,
    #                                                           hgt=max_height)
    _log_and_run(convert_unws)

    # Now also add geo projection and SRS info to the .tif files
    projscript = os.path.join(SCRIPTS_DIR, "gdalcopyproj.py")
    # Make fake .int and .int.rsc to use the ROI_PAC driver
    open("fake.int", "w").close()
    force_symlink("dem.rsc", "fake.int.rsc")

    copyproj_cmd = (
        """find . -name "*.tif" -print0 | \
xargs -0 -n1 -I{} --max-procs=50 %s fake.int {} """
        % projscript
    )
    _log_and_run(copyproj_cmd)

    os.remove("fake.int")
    os.remove("fake.int.rsc")


# TODO: fix this function for new stuff
def run_sbas_inversion(
    ref_row=None,
    ref_col=None,
    ref_station=None,
    window=None,
    alpha=0,
    constant_velocity=False,
    difference=False,
    deramp_order=2,
    ignore_geos=False,
    stackavg=False,
    **kwargs,
):
    """10. Perofrm SBAS inversion, save the deformation as .npy

    Assumes we are in the directory with all .unw files"""
    import insar.prepare

    igram_path = os.path.realpath(os.getcwd())

    # Note: with overwrite=False, this will only take a long time once
    insar.prepare.prepare_stacks(
        igram_path,
        ref_row=ref_row,
        ref_col=ref_col,
        ref_station=ref_station,
        overwrite=False,
        deramp_order=deramp_order,
    )

    # Now this happens in prepare
    # insar.stackavg.run_stack(
    #     # unw_stack_file="unw_stack.vrt",
    #     outfile=None,
    #     ignore_geo_file=None,
    #     geo_dir="../",
    #     igram_dir=igram_path,
    #     max_temporal_baseline=900,
    #     min_date=None,
    #     max_date=None,
    #     ramp_order=1,
    # )
    return

    cmd = (
        "julia --start=no /home/scott/repos/InsarTimeseries.jl/src/runcli.jl "
        " -o {output_name} --alpha {alpha} "
    )
    # cmd = "/home/scott/repos/InsarTimeseries.jl/builddir/insarts " \

    if ignore_geos:
        cmd += " --ignore slclist_ignore.txt "

    if constant_velocity:
        output_name = "deformation_linear.h5"
        cmd += " --constant-velocity "
    elif stackavg:
        output_name = "deformation_stackavg.h5"
        cmd += " --stack-average "
    else:
        output_name = "deformation.h5"

    cmd = cmd.format(output_name=output_name, alpha=alpha)
    logger.info("Saving to %s" % output_name)
    _log_and_run(cmd)


# List of functions that run each step
STEPS = [
    create_dem,
    download_data,
    run_sentinel_stack,
    prep_igrams_dir,
    create_sbas_list,
    run_form_igrams,  # run_ps_sbas_igrams,
    record_los_vectors,
    run_snaphu,
    convert_to_tif,
    run_sbas_inversion,
]
# Form string for help function "1:download_eof,2:..."
STEP_LIST = ",\n".join(
    "%d:%s" % (num, func.__name__) for (num, func) in enumerate(STEPS, start=1)
)


@log_runtime
def main(working_dir, kwargs):
    # TODO: maybe let user specify individual steps?
    if working_dir != ".":
        logger.info("Changing directory to {}".format(working_dir))
        os.chdir(working_dir)

    # Use the --step option first, or else use the --start
    # Subtract 1 so that they are list indices, starting at 0
    step_list = [s - 1 for s in kwargs["step"]] or range(
        kwargs["start"] - 1, len(STEPS)
    )
    logger.info("Running steps %s", ",".join(str(s + 1) for s in step_list))
    for stepnum in step_list:
        curfunc = STEPS[stepnum]
        logger.info("Starting step %d: %s", stepnum + 1, curfunc.__name__)
        ret = curfunc(**kwargs)
        if ret:  # Option to have step give non-zero return to halt things
            return ret
