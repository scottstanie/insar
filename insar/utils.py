#! /usr/bin/env python
"""Author: Scott Staniewicz
Helper functions to prepare and process UAVSAR data
Most functions will be stuck in here until there's a more sensible module.
Email: scott.stanie@utexas.edu
"""

import argparse
import numpy as np

import insar.sario


def downsample_im(image, rate=10):
    """Takes a numpy matrix of an image and returns a smaller version

    Args:
        image (np.array) 2D array of an image
        rate (int) the reduction rate to downsample
    """
    return image[::rate, ::rate]


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
