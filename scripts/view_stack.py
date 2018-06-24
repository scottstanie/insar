#!/usr/bin/env python
import argparse
import sys
from os.path import join, dirname, abspath
import numpy as np
try:
    import insar
except ImportError:  # add root to pythonpath if import fails
    sys.path.insert(0, dirname(dirname(abspath(__file__))))
from insar import timeseries, plotting


def load_deformation(igram_path, ref_row=None, ref_col=None):
    try:
        deformation = np.load(join(igram_path, 'deformation.npy'))
        # geolist is a list of datetimes: encoding must be bytes
        geolist = np.load(join(igram_path, 'geolist.npy'), encoding='bytes')
    except (IOError, OSError):
        if not ref_col and not ref_col:
            print("Need ref_row, ref_col to invert, or need deformation.npy and geolist.npy")
            return

        geolist, phi_arr, deformation, varr, unw_stack = timeseries.run_inversion(
            igram_path, reference=(ref_row, ref_col))
        np.save(join(igram_path, 'deformation.npy'), deformation)
        np.save(join(igram_path, 'geolist.npy'), geolist)

    return geolist, deformation


def view_stack(args):
    """Wrapper to run `view_stack <plotting.view_stack>` with command line args"""
    geolist, deformation = load_deformation(args.igram_path, args.ref_row, args.ref_col)
    plotting.view_stack(deformation, geolist, image_num=-1)


def animate(args):
    """Wrapper to run `animate_stack <plotting.animate_stack>` with command line args"""
    geolist, deformation = load_deformation(args.igram_path, args.ref_row, args.ref_col)
    titles = [d.strftime("%Y-%m-%d") for d in geolist]
    plotting.animate_stack(
        deformation,
        pause_time=args.pause_time,
        display=args.display,
        titles=titles,
        save_title=args.save_title)


def main():
    parser = argparse.ArgumentParser()
    # TODO: probably want to use http://click.pocoo.org/5/commands/ to nest command options
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Select to view all layers of deformation stack with plotting.animate_stack."
        "Leave out to click to explore pixel timeseries.")
    parser.add_argument(
        "--igram-path",
        "-p",
        default=".",
        help="Directory where deformation.npy and geolist.npy or .unw files are located")
    parser.add_argument(
        "--ref-row",
        type=int,
        help="Row number of pixel to use as unwrapping reference (for SBAS inversion)")
    parser.add_argument(
        "--ref-col",
        type=int,
        help="Column number of pixel to use as unwrapping reference (for SBAS inversion)")
    parser.add_argument(
        "--display",
        action="store_true",
        help="For --animate option, select --display to pop up the GUI (instead of just saving)")
    parser.add_argument(
        "--pause-time",
        type=int,
        default=200,
        help=
        "For --animate option, time in milliseconds to pause between stack layers (default 200).")
    parser.add_argument(
        "--save-title",
        help="For --animate option, If you want to save the animation as a movie,"
        " title to save file as.")
    args = parser.parse_args()

    if args.animate:
        animate(args)
    else:
        view_stack(args)


if __name__ == '__main__':
    main()
