#!/usr/bin/env python
"""Stiches two .hgt files to make one DEM and .dem.rsc file"""
import argparse
import json
import sys
import subprocess
from os.path import abspath, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from insar.sario import load_file
import insar.dem
import insar.geojson
from insar.log import get_log


def positive_integer(argstring):
    """Test for rate input argument"""
    num = int(argstring)
    if num <= 0:
        raise argparse.ArgumentTypeError("Not a positive integer.")
    return num


def main():
    logger = get_log()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geojson", "-g", required=True, help="File containing the geojson object for DEM bounds")
    parser.add_argument(
        "--rate",
        "-r",
        default=1,
        type=positive_integer,
        help="Rate at which to upsample DEM (default=1, no upsampling)")
    parser.add_argument("--output", "-o", default="elevation.dem", help="Name of output dem file")
    args = parser.parse_args()

    if args.geojson == 'stdin':
        geojson = json.load(sys.stdin)
    else:
        with open(args.geojson, 'r') as f:
            geojson = json.load(f)

    bounds = insar.geojson.geojson_to_bounds(geojson)
    logger.info("Bounds: %s", " ".join(str(b) for b in bounds))

    d = insar.dem.Downloader(*bounds)
    d.download_all()

    s = insar.dem.Stitcher(d.srtm1_tile_names())
    stitched_dem = s.load_and_stitch()

    # Now create corresponding rsc file
    rsc_dict = s.create_dem_rsc()

    # Cropping: get very close to the bounds asked for:
    logger.info("Cropping stitched DEM to boundaries")
    stitched_dem, new_starts, new_sizes = insar.dem.crop_stitched_dem(bounds, stitched_dem,
                                                                      rsc_dict)
    new_x_first, new_y_first = new_starts
    new_rows, new_cols = new_sizes
    # Now adjust the .dem.rsc data to reflect new top-left corner and new shape
    rsc_dict['X_FIRST'] = new_x_first
    rsc_dict['Y_FIRST'] = new_y_first
    rsc_dict['FILE_LENGTH'] = new_rows
    rsc_dict['WIDTH'] = new_cols

    # Upsampling:
    rate = args.rate
    dem_filename = args.output
    rsc_filename = dem_filename + '.rsc'
    if rate == 1 and False:
        logger.info("Rate = 1: No upsampling to do")
        logger.info("Writing DEM to %s", dem_filename)
        stitched_dem.tofile(dem_filename)
        logger.info("Writing .dem.rsc file to %s", rsc_filename)
        with open(rsc_filename, "w") as f:
            f.write(s.format_dem_rsc(rsc_dict))
        sys.exit(0)

    logger.info("Upsampling by {}".format(rate))
    dem_filename_small = dem_filename.replace(".dem", "_small.dem")
    rsc_filename_small = rsc_filename.replace(".dem.rsc", "_small.dem.rsc")

    logger.info("Writing non-upsampled dem to %s", dem_filename_small)
    stitched_dem.tofile(dem_filename_small)
    logger.info("Writing non-upsampled dem.rsc to %s", rsc_filename_small)
    with open(rsc_filename_small, "w") as f:
        f.write(s.format_dem_rsc(rsc_dict))

    # Now upsample this block
    nrows, ncols = stitched_dem.shape
    upsample_cmd = [
        'bin/upsample', dem_filename_small,
        str(rate),
        str(ncols),
        str(nrows), dem_filename
    ]
    logger.info("Upsampling using %s:", upsample_cmd[0])
    logger.info(' '.join(upsample_cmd))
    subprocess.check_call(upsample_cmd)

    # Redo a new .rsc file for it
    logger.info("Writing new upsampled dem to %s", rsc_filename)
    with open(rsc_filename, "w") as f:
        upsampled_rsc = insar.dem.upsample_dem_rsc(rate=rate, rsc_dict=rsc_dict)
        f.write(upsampled_rsc)


if __name__ == '__main__':
    main()
