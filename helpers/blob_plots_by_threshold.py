#!/usr/bin/env python
import argparse
import os
import glob
from collections import defaultdict, Counter
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
from insar import blob, timeseries, latlon
import sardem


def run_blob(thresh, val_thresh, fname):
    extra_args = {
        'threshold': thresh,
        'value_threshold': val_thresh,
        'max_sigma': 100,
        'min_sigma': 10,
    }
    blobs = blob._make_blobs(img, extra_args)

    print("saving", fname, "size:", len(blobs))
    np.save(fname, blobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    igram_path = '.'
    print("Loading deformation image")
    geolist, deformation = timeseries.load_deformation(igram_path)
    rsc_data = sardem.loading.load_dem_rsc(os.path.join(igram_path, 'dem.rsc'))

    img = latlon.LatlonImage(data=np.mean(deformation[-3:], axis=0), dem_rsc=rsc_data)

    blob_name_list = glob.glob("./blobs_*.npy")
    if len(blob_name_list) == 0 or args.overwrite:

        threshold_list = [0.3, 0.5, 0.8, 1]  # For filter response
        value_threshold_list = np.linspace(0.2, 2, 20)  # Blob magnitude

        procs = []
        blob_name_list = []
        for thresh in threshold_list:
            for val_thresh in value_threshold_list:
                blobs_name = 'blobs_{0}_{1:.1f}.npy'.format(thresh, val_thresh)
                blob_name_list.append(blobs_name)
                p = mp.Process(target=run_blob, args=(thresh, val_thresh, blobs_name))
                p.start()
                procs.append(p)

        [p.join() for p in procs]

    # Now load and manipulate for stats

    print("Loading blobs")
    blobs_list = [np.load(features_vs_size) for features_vs_size in blob_name_list]

    features_thresh_raw = defaultdict(dict)
    features_vs_size = {}
    print("Parsing blobs")
    for blobs, b_name in zip(blobs_list, blob_name_list):
        _, thresh, val_thresh = b_name.strip('.npy').split('_')
        thresh = float(thresh)
        val_thresh = float(val_thresh)
        features_thresh_raw[thresh][val_thresh] = len(blobs)

        if thresh == 0.3:
            c = Counter(blobs.astype(int)[:, 2])
            features_vs_size[val_thresh] = np.array(list(c.items()))

    features_vs_thresh = {}
    for thresh, dic in features_thresh_raw.items():
        features_vs_thresh[thresh] = np.array(list(dic.items()))

    fig, axes = plt.subplots(1, 2)
    legends = []
    # import ipdb
    # ipdb.set_trace()
    for thresh, data in sorted(features_vs_thresh.items()):
        legends.append('thresh: %s' % thresh)
        axes[0].scatter(data[:, 0], data[:, 1])

    axes[0].set_title('Features vs thresh')
    axes[0].set_xlabel('value threshold')
    axes[0].set_ylabel('number of blobs found')
    axes[0].set_yscale('log')
    axes[0].legend(legends)

    legends = []
    for thresh, data in sorted(features_vs_size.items()):
        legends.append('thresh: %s' % thresh)
        axes[1].scatter(img.blob_size(data[:, 0]), data[:, 1])
    axes[1].set_title('Features vs size (thresh=0.5)')
    # axes[1].set_xlabel('Size, $\sigma$')
    axes[1].set_xlabel('Size, km')
    axes[1].set_ylabel('number of blobs found')
    axes[1].legend(legends)

    print("Showing")
    plt.show(block=True)
