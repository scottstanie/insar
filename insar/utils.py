#! /usr/bin/env python
"""Author: Scott Staniewicz
Helper functions to prepare and process UAVSAR data
Most functions will be stuck in here until there's a more sensible module.
Email: scott.stanie@utexas.edu
"""

import argparse
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import insar.sario


def downsample_im(image, rate=10):
    """Takes a numpy matrix of an image and returns a smaller version

    Args:
        image (np.array) 2D array of an image
        rate (int) the reduction rate to downsample
    """
    return image[::rate, ::rate]


def upsample_dem(dem_img, rate=3):
    """Interpolates a DEM to higher resolution for better InSAR quality


    Args:
        dem_img: numpy.ndarray (int16)
        rate: int, default = 3

    Returns:
        numpy.ndarray (int16): original dem_img upsampled by `rate`. Needs
            to return same type since downstream scripts expect int16 DEMs

    """

    s1, s2 = dem_img.shape
    # TODO: figure out whether makeDEM.m really needs to form using
    # points 1:size and then interpolate with points 0:size
    orig_points = (np.arange(1, s1 + 1), np.arange(1, s2 + 1))
    rgi = RegularGridInterpolator(points=orig_points, values=dem_img)

    # Make a grid from 0 to (size-1) inclusive, in both directions
    # 1j used to say "make s1*rate number of points exactly"
    numx = 1 + (s1 - 1) * rate
    numy = 1 + (s2 - 1) * rate
    X, Y = np.mgrid[1:s1:numx * 1j, 1:s2:numy * 1j]

    # new_points will be a 2xN matrix, N=(numx*numy)
    new_points = np.vstack([X.ravel(), Y.ravel()])

    # rgi expects Nx2 as input, and will output as a 1D vector
    return rgi(new_points.T).reshape(numx, numy).round().astype(dem_img.dtype)


def mosaic_dem(d1, d2):
    D = np.concatenate((d1, d2), axis=1)
    nrows, ncols = d1.shape
    D = np.delete(D, nrows, axis=1)
    return D


def clip(image):
    """Convert float image to only range 0 to 1 (clips)"""
    return np.clip(np.abs(image), 0, 1)


def log(image):
    """Converts magnitude amplitude image to log scale"""
    return 20 * np.log10(image)


def split_array_into_blocks(data):
    """Takes a long rectangular array (like UAVSAR) and creates blocks

    Useful to look at small data pieces at a time in dismph

    Returns:
        blocks (list[np.ndarray])
    """
    rows, cols = data.shape
    blocks = np.array_split(data, rows // cols + 1)
    return blocks


def split_and_save(filename):
    """Creates several files from one long data file

    Saves them with same filename with .1,.2,.3... at end before ext
    e.g. brazos_14937_17087-002_17088-003_0001d_s01_L090HH_01.int produces
        brazos_14937_17087-002_17088-003_0001d_s01_L090HH_01.1.int
        brazos_14937_17087-002_17088-003_0001d_s01_L090HH_01.2.int...

    Output:
        newpaths (list[str]): full paths to new files created
    """

    data = insar.sario.load_file(filename)
    blocks = split_array_into_blocks(data)

    ext = insar.sario.get_file_ext(filename)
    newpaths = []

    for idx, block in enumerate(blocks, start=1):
        fname = filename.replace(ext, ".{}{}".format(str(idx), ext))
        print("Saving {}".format(fname))
        insar.sario.save_array(fname, block)
        newpaths.append(fname)

    return newpaths


def combine_cor_amp(corfilename, save=True):
    """Takes a .cor file from UAVSAR (which doesn't contain amplitude),
    and creates a new file with amplitude data interleaved for dishgt

    dishgt brazos_14937_17087-002_17088-003_0001d_s01_L090HH_01_withamp.cor 3300 1 5000 1
      where 3300 is number of columns/samples, and we want the first 5000 rows. the final
      1 is needed for the contour interval to set a max of 1 for .cor data

    Inputs:
        corfilename (str): string filename of the .cor from UAVSAR
        save (bool): True if you want to save the combined array

    Returns:
        cor_with_amp (np.ndarray) combined correlation + amplitude (as complex64)
        outfilename (str): same name as corfilename, but _withamp.cor
            Saves a new file under outfilename
    Note: .ann and .int files must be in same directory as .cor
    """
    ext = insar.sario.get_file_ext(corfilename)
    assert ext == '.cor', 'corfilename must be a .cor file'

    intfilename = corfilename.replace('.cor', '.int')

    intdata = insar.sario.load_file(intfilename)
    amp = np.abs(intdata)

    cordata = insar.sario.load_file(corfilename)
    # For dishgt, it expects the two matrices stacked [[amp]; [cor]]
    cor_with_amp = np.vstack((amp, cordata))

    outfilename = corfilename.replace('.cor', '_withamp.cor')
    insar.sario.save_array(outfilename, cor_with_amp)
    return cor_with_amp, outfilename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, help="Specify command to run on file.")
    parser.add_argument("filename", type=str, help="Specify the input UAVSAR filename")
    args = parser.parse_args()

    if args.command == 'info':
        ann_data = insar.sario.parse_ann_file(args.filename)
        print(ann_data)
    elif args.command == 'split':
        split_and_save(args.filename)


if __name__ == "__main__":
    main()
