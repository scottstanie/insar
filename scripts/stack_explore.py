#!/usr/bin/env python
import argparse
import sys
from os.path import dirname, abspath
import numpy as np
try:
    import insar
except ImportError:  # add root to pythonpath if import fails
    sys.path.insert(0, dirname(dirname(abspath(__file__))))
from insar import timeseries, plotting


def explore_stack(igram_path=".", ref_row=None, ref_col=None):
    try:
        deformation = np.load('deformation.npy')
        geolist = np.load('geolist.npy')
    except (IOError, OSError):
        if not ref_col and not ref_col:
            print("Need ref_row, ref_col to invert, or need deformation.npy and geolist.npy")
            return

        geolist, phi_arr, deformation, varr, unw_stack = timeseries.run_inversion(
            igram_path, reference=(ref_row, ref_col))
        np.save('deformation.npy', deformation)
        np.save('geolist.npy', geolist)

    plotting.explore_stack(deformation, geolist, image_num=-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--igram-path",
        "-p",
        default=".",
        help="Directory where deformation.npy and geolist.npy or .unw files are located")
    parser.add_argument(
        "--ref-row",
        type=int,
        help="Row number of pixel to use as unwrapping reference for SBAS inversion")
    parser.add_argument(
        "--ref-col",
        type=int,
        help="Column number of pixel to use as unwrapping reference for SBAS inversion")
    args = parser.parse_args()
    explore_stack(args.igram_path, args.ref_row, args.ref_col)
