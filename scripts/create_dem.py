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
    parser.add_argument("--geojson", "-g", required=True, help="File containing the geojson object for DEM bounds")
    parser.add_argument("--rate", "-r", default=1, type=positive_integer, help="Rate at which to upsample DEM (default=1, no upsampling)")
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

    # Upsampling:
    rate = args.rate
    filename = args.output
    rsc_filename = filename + '.rsc'
    if rate == 1:
        logger.info("Rate = 1: No upsampling to do")
        logger.info("Writing DEM to %s", filename)
        stitched_dem.tofile(filename)
        logger.info("Writing .dem.rsc file to %s", rsc_filename)
        with open(rsc_filename, "w") as f:
            f.write(s.format_dem_rsc(rsc_dict))
        sys.exit(0)

    logger.info("Upsampling by {}".format(rate))
    filename_small = filename.replace(".dem", "_small.dem")
    rsc_filename_small = rsc_filename.replace(".dem.rsc", "_small.dem.rsc")

    logger.info("Writing non-upsampled dem to %s", filename_small)
    stitched_dem.tofile(filename_small)
    logger.info("Writing non-upsampled dem.rsc to %s", rsc_filename_small)
    with open(rsc_filename_small, "w") as f:
        f.write(s.format_dem_rsc(rsc_dict))

    # Now upsample this block
    subprocess.check_call(['bin/upsample', filename, str(rate)])

    # Redo a new .rsc file for it
    logger.info("Writing new upsampled dem to %s", rsc_filename)
    with open(rsc_filename, "w") as f:
        upsampled_rsc = insar.dem.upsample_dem_rsc(rate=rate, rsc_dict=rsc_dict)
        f.write(upsampled_rsc)


if __name__ == '__main__':
    main()
