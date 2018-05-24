#!/usr/bin/env python
"""Stiches two .hgt files to make one DEM and .dem.rsc file"""
import argparse
import sys
import os.path
from insar.sario import load_file
import insar.dem
from insar.log import get_log


def positive_integer(argstring):
    num = int(argstring)
    if num <= 0:
        raise argparse.ArgumentTypeError("Not a positive integer.")
    return num


def main():
    logger = get_log()
    parser = argparse.ArgumentParser()
    parser.add_argument("left", help="Left .hgt block for the dem")
    parser.add_argument("right", help="Right .hgt block in mosaic")
    parser.add_argument("--rate", "-r", type=positive_integer, help="Rate at which to upsample DEM")
    parser.add_argument("--output", "-o", default="elevation.dem", help="Name of output dem file")
    args = parser.parse_args()

    if not all(insar.sario.get_file_ext(f) == '.hgt' for f in (args.left, args.right)):
        logger.error('Both files must be .hgt files.')
        sys.exit(1)

    left_block = load_file(args.left)
    right_block = load_file(args.right)
    if not left_block.shape == right_block.shape:
        logger.error(
            'Both files must be same data type/shape, either 1 degree (30 m) or 3 degree (90 m)')
        sys.exit(1)

    if left_block.shape == (3601, 3601):
        logger.info('SRTM type for {}: 1 degree data, 30 m'.format(args.left))

    big_output_path = os.path.join(os.path.dirname(args.left), args.output)
    big_rsc_output_path = big_output_path + '.rsc'
    small_output_path = big_output_path.replace('.dem', '_small.dem')
    small_rsc_output_path = small_output_path + '.rsc'

    full_block = insar.dem.mosaic_dem(left_block, right_block)
    logger.info('Writing stitched DEM to {}'.format(small_output_path))
    full_block.tofile(small_output_path)

    small_rsc_dict = insar.dem.create_dem_rsc([args.left, args.right])
    with open(small_rsc_output_path, 'w') as f:
        logger.info('Writing rsc to {}'.format(small_rsc_output_path))
        f.write(insar.dem.format_dem_rsc(small_rsc_dict))

    rate = args.rate
    if rate == 1:
        logger.info("Rate = 1: No upsampling to do")
        logger.info("Renaming {} to {}".format(small_output_path, big_output_path))
        logger.info("Renaming {} to {}".format(small_rsc_output_path, big_rsc_output_path))
        os.rename(small_output_path, big_output_path)
        os.rename(small_rsc_output_path, big_rsc_output_path)
        sys.exit(0)
    else:
        logger.info("Upsampling by {}".format(rate))

    # Now upsample this block
    big_dem = insar.dem.upsample_dem(full_block, rate)

    # Stick output in same path as input .hgt files
    big_dem.tofile(big_output_path)

    # Redo a new .rsc file for it
    insar.dem.upsample_dem_rsc(small_rsc_output_path, rate)


if __name__ == '__main__':
    main()
