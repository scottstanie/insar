#!/usr/bin/env python
"""
Author: Scott Staniewicz

Script to create binary image masks of flooding from .int or .cor files.
Breaks the very long UAVSAR image into approximately square blocks first,
saves these as .1.int, .2.int, etc. in same file location,
then creates a mask on each of these new files.

Example Usage:
    python create_masks.py ~/uav-sar-data/one-strip-test/brazos-20170901-int/brazos_14937_17087-002_17088-003_0001d_s01_L090HH_01.int --threshold 0.002 --downsample 5
    python create_masks.py ~/uav-sar-data/one-strip-test/brazos-20170901-int/brazos_14937_17087-002_17088-003_0001d_s01_L090HH_01.cor -t 0.2 -d 5

Note: Adding --despeckle will run a lee filter on the image first before creating mask
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import os
import sys
import numpy as np
# For despeckling with lee filter
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
# for morphological operations
# from skimage.morphology import erosion
# from skimage.morphology import disk

import insar.sario
import insar.utils


# TODO: figure out better module to put this function
def lee_filter(img, size=5):
    """Run a lee filter on an image to remove speckle noise"""
    # https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python
    img = img.astype(np.float32)
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def mask(filepath, threshold=0.002):
    cur_file = insar.sario.load_file(filepath)

    # Note: abs for complex files, but also fine for .cor magnitude files
    ampfile = np.abs(cur_file)

    binary_im = ampfile < threshold

    # To run a morphological closure on the mask:
    # if args.despeckle:
    #     ampfile = remove_speckles(ampfile)
    #     strel = disk(1)
    #     binary_im = erosion(binary_im, strel)
    return binary_im


def save(filepath, mask, downsample=0):
    print('ok')
    # maskfile = filepath.replace(ext, '.jpg')
    # Keep old .ext so we know what type was masked
    ext = insar.sario.get_file_ext(filepath)
    maskfile = filepath.replace(ext, ext + '.png')

    if args.downsample:
        insar.sario.save(maskfile, insar.utils.downsample_im(mask, args.downsample))
    else:
        insar.sario.save(maskfile, mask)
    return True


def mask_and_save(arglist):
    filepath, args = arglist
    print('Processing', filepath)
    binary_im = mask(filepath, args)
    save(filepath, binary_im, args)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Specify the input UAVSAR filename")
    parser.add_argument(
        "--threshold",
        '-t',
        type=float,
        default=0.002,
        help="Threshold of magnitude for water segmenting")
    parser.add_argument(
        "--downsample", '-d', type=int, default=0, help="Factor to shrink final outputs by")
    parser.add_argument(
        "--despeckle",
        action="store_true",
        default=False,
        help="Remove speckles from mask with morphological operations.")
    args = parser.parse_args()

    filepath = os.path.expanduser(args.filename)
    ext = insar.sario.get_file_ext(filepath)
    allowed_exts = ('.int', '.cor', '.mlc')
    if ext not in allowed_exts:
        print('Error: Only taking {} files for now.'.format(', '.join(allowed_exts)))
        print('Cannot process {}'.format(ext))
        sys.exit(-1)

    block1_path = filepath.replace(ext, '.1' + ext)
    if not os.path.exists(block1_path):
        block_paths = insar.utils.split_and_save(filepath)
    else:
        block_paths = glob.glob(filepath.replace(ext, '.[0-9]{}'.format(ext)))

    with ProcessPoolExecutor() as executor:
        print('oo')
        executor.map(lambda x: mask_and_save(x, args), [block_paths, args])
