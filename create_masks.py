import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob

import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Specify the input UAVSAR filename")
    parser.add_argument("--threshold", '-t', type=float, required=False, default=0.002,
                        help="Threshold of magnitude for water segmenting")
    parser.add_argument("--downsample", '-d', type=int, required=False, default=0,
                        help="Factor to shrink final outputs by")
    args = parser.parse_args()

    filepath = os.path.expanduser(args.filename)
    ext = utils.get_file_ext(filepath)
    if ext != '.int':
        print('Error: Only taking .int files for now.')
        print('Cannot process {}'.format(ext))
        sys.exit(-1)

    block1_path = filepath.replace(ext, '.1' + ext)
    if not os.path.exists(block1_path):
        block_paths = utils.split_and_save(filepath)
    else:
        block_paths = glob.glob(filepath.replace(ext, '.[0-9]{}'.format(ext)))


    for cur_filepath in block_paths:
        print('Processing', cur_filepath)
        cur_file = utils.load_file(cur_filepath)

        ampfile = np.abs(cur_file)

        mask = ampfile < args.threshold

        # maskfile = cur_filepath.replace(ext, '.jpg')
        maskfile = cur_filepath.replace(ext, '.png')

        # import pdb; pdb.set_trace()
        if args.downsample:
            utils.save_array(maskfile, utils.downsample_im(mask, args.downsample))
        else:
            utils.save_array(maskfile, mask)
