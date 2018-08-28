import argparse
import os
import insar
import sardem
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('igram_path')
    parser.add_argument('--load', action='store_true', help='Load last calculated blobs')
    parser.add_argument('--row-start', default=0)
    parser.add_argument('--row-end', default=-1)
    parser.add_argument('--col-start', default=0)
    parser.add_argument('--col-end', default=-1)
    parser.add_argument('--title-extra', default='')
    return parser.parse_args()


def main():
    args = get_args()
    print("Searching %s for igram_path" % args.igram_path)
    geolist, deformation = insar.timeseries.load_deformation(args.igram_path)
    rsc_data = sardem.loading.load_dem_rsc(os.path.join(args.igram_path, 'dem.rsc'))
    img = deformation[-1]
    img = img[args.row_start:args.row_end, args.col_start:args.col_end]

    title = "Deformation from %s to %s. %s" % (geolist[0], geolist[-1], args.title_extra)
    imagefig, axes_image = insar.plotting.plot_image_shifted(
        img, img_data=rsc_data, title=title, xlabel='Longitude', ylabel='Latitude')

    if args.load:
        blobs = np.load('blobs.npy')
    else:
        print("Finding neg blobs")
        blobs_neg = insar.blobs.find_blobs(
            -img, blob_func='blob_log', threshold=1, min_sigma=3, max_sigma=40)
        print("Finding pos blobs")
        blobs_pos = insar.blobs.find_blobs(
            img, blob_func='blob_log', threshold=.7, min_sigma=3, max_sigma=40)
        print("Blobs found:")
        print(blobs_neg.astype(int))
        print(blobs_pos.astype(int))
        blobs = np.vstack((blobs_neg, blobs_pos))
        np.save('blobs.npy', blobs)

    blobs_ll = insar.blobs.blobs_latlon(blobs, rsc_data)
    for lat, lon, r in blobs_ll:
        print('({0:.4f}, {1:.4f}): radius: {2}'.format(lat, lon, r))

    insar.blobs.plot_blobs(img, blobs=blobs_ll, cur_axes=imagefig.gca())


if __name__ == '__main__':
    main()
