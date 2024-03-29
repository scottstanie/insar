#!/usr/bin/env python
import argparse
import os
import glob
from collections import defaultdict, Counter
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
from insar import blob
from apertools import sario, latlon
import sardem


def run_blob(thresh, mag_thresh, fname, min_sigma=10, max_sigma=100):
    extra_args = {
        "threshold": thresh,
        "mag_threshold": mag_thresh,
        "max_sigma": max_sigma,
        "min_sigma": min_sigma,
        # 'overlap': 2,  # 2 overlap will never apply, keep all blobs
    }
    blobs = blob._make_blobs(img, extra_args)

    print("saving", fname, "size:", len(blobs))
    np.save(fname, blobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--overwrite", action="store_true", default=False)
    parser.add_argument("--min-km", type=float, default=0.5)
    parser.add_argument("--max-km", type=float, default=10)
    args = parser.parse_args()

    igram_path = "."
    print("Loading deformation image")
    slclist, deformation = sario.load_deformation(igram_path)
    rsc_data = sardem.loading.load_dem_rsc(os.path.join(igram_path, "dem.rsc"))

    img = latlon.LatlonImage(data=np.mean(deformation[-3:], axis=0), dem_rsc=rsc_data)

    blob_name_list = glob.glob("./blobs_*.npy")
    if len(blob_name_list) == 0 or args.overwrite:
        min_sigma = img.km_to_pixels(args.min_km)
        max_sigma = img.km_to_pixels(args.max_km)
        print("Minimum blob size: %.4f pixels, %.4f km" % (min_sigma, args.min_km))
        print("Maximum blob size: %.4f pixels, %.4f km" % (max_sigma, args.max_km))

        # threshold_list = [0.3, 0.5, 0.8, 1]  # For filter response
        threshold_list = [0.5]  # For filter response
        mag_threshold_list = np.linspace(0.2, 2, 20)  # Blob magnitude

        procs = []
        blob_name_list = []
        for thresh in threshold_list:
            for mag_thresh in mag_threshold_list:
                blobs_name = "blobs_{0:.1f}_{1:.1f}.npy".format(thresh, mag_thresh)
                blob_name_list.append(blobs_name)
                p = mp.Process(
                    target=run_blob,
                    args=(thresh, mag_thresh, blobs_name),
                    kwargs={
                        "min_sigma": min_sigma,
                        "max_sigma": max_sigma,
                    },
                )
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
        _, thresh, mag_thresh = b_name.strip(".npy").split("_")
        thresh = float(thresh)
        mag_thresh = float(mag_thresh)
        features_thresh_raw[thresh][mag_thresh] = len(blobs)

        if thresh == 0.3:
            c = Counter(blobs.astype(int)[:, 2])
            features_vs_size[mag_thresh] = np.array(list(c.items()))

    features_vs_thresh = {}
    for thresh, dic in features_thresh_raw.items():
        features_vs_thresh[thresh] = np.array(list(dic.items()))

    fig, axes = plt.subplots(1, 2)
    legends = []
    # import ipdb
    # ipdb.set_trace()
    for thresh, data in sorted(features_vs_thresh.items()):
        legends.append("thresh: %s" % thresh)
        axes[0].scatter(data[:, 0], data[:, 1])

    axes[0].set_title("Features vs thresh")
    axes[0].set_xlabel("mag threshold")
    axes[0].set_ylabel("number of blobs found")
    axes[0].set_yscale("log")
    axes[0].legend(legends)

    legends = []
    for thresh, data in sorted(features_vs_size.items()):
        legends.append("thresh: %s" % thresh)
        axes[1].scatter(img.pixel_to_km(data[:, 0]), data[:, 1])
    axes[1].set_title("Features vs size (thresh=0.5)")
    # axes[1].set_xlabel('Size, $\sigma$')
    axes[1].set_xlabel("Size, km")
    axes[1].set_ylabel("number of blobs found")
    axes[1].legend(legends)

    print("Showing")
    plt.show(block=True)
