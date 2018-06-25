#!/usr/bin/env python
"""Creates a .kml file to load tif-converted image

Usage:
    ./make_kml.py --rscfile RSCFILE --tif-img TIF_IMG  [--title TITLE] [--desc DESC]

    python make_kml.py -r dem.rsc -i 20180420_20180502.tif -t "My igram" -d "From April in Hawaii" > out.kml
"""
import argparse
import sys
from os.path import abspath, dirname, join, exists
from insar.dem import upsample_dem_rsc, create_kml
from insar.sario import load_dem_rsc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rscfile",
        "-r",
        default="dem.rsc",
        required=True,
        help="The associated .rsc file containing lat/lon start and steps")
    parser.add_argument(
        "--tif-img", "-i", required=True, help="The .tif file to load into Google Earth")
    parser.add_argument("--title", "-t", help="Title of the KML object once loaded.")
    parser.add_argument("--desc", "-d", help="Description for google Earth.")
    args = parser.parse_args()

    rsc_data = load_dem_rsc(args.rscfile)
    print(create_kml(rsc_data, args.tif_img, title=args.title, desc=args.desc))
