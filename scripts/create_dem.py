"""Stiches two .hgt files to make one DEM and .dem.rsc file"""
import argparse
import sys
import os.path
from insar.sario import load_file
import insar.dem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left", help="Left .hgt block for the dem")
    parser.add_argument("right", help="Right .hgt block in mosaic")
    parser.add_argument("--output", "-o", default="elevation.dem", help="Name of output dem file")
    args = parser.parse_args()

    if not all(insar.sario.get_file_ext(f) == '.hgt' for f in args):
        print('Both files must be .hgt files.')
        sys.exit(1)

    left = load_file(args.left)
    right = load_file(args.right)
    if not left.shape == right.shape:
        print('Both files must be same data type/shape, either 1 degree (30 m) or 3 degree (90 m)')
        sys.exit(1)

    if left.shape == (3601, 3601):
        print('SRTM type for {}: 1 degree data, 30 m'.format(args.left))

    full_block = insar.dem.mosaic_dem(left, right)
    big_dem = insar.dem.upsample_dem(full_block)

    # Stick output in same path as input .hgt files
    output_path = os.path.join(os.path.dirname(args.left), args.output)
    big_dem.tofile(output_path)

    # Redo a new .rsc file for it
    rsc_output_path = os.path.join(output_path, '.rsc')


if __name__ == '__main__':
    main()
