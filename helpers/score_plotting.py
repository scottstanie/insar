import numpy as np
import matplotlib.pyplot as plt
from insar.blob import synthetic
from insar import blob

if __name__ == "__main__":

    fp_blobs, fp_patches = synthetic.load_run_blob_type(
        "/home/scott/repos/insar/patches5", "fp"
    )
    fp_patch_scores = blob.scores.analyze_patches(fp_patches)
    fp_dummy = fp_patch_scores + 0.01
    td_blobs, td_patches = synthetic.load_run_blob_type(
        "/home/scott/repos/insar/patches5", "td"
    )
    td_patch_scores = blob.scores.analyze_patches(td_patches)
    all_patch_scores = np.concatenate((fp_patch_scores, fp_dummy, td_patch_scores))

    red = np.concatenate(
        (np.ones(len(fp_patch_scores) + len(fp_dummy)), np.zeros(len(td_patch_scores)))
    ).astype(bool)
    green = np.logical_not(red)

    all_patch_scores = np.abs(all_patch_scores)

    blob.plot.plot_scores(
        all_patch_scores,
        nrows=3,
        y_idxs=(red, green),
        titles=blob.scores.FUNC_LIST_NAMES,
    )
    plt.show()
