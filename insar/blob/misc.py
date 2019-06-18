import numpy as np
import multiprocessing as mp
import functools
from . import find_blobs
from apertools import sario
from insar import timeseries

MAX_PROCS = mp.cpu_count()
BLOB_KWARG_DEFAULTS = {'threshold': 1, 'min_sigma': 3, 'max_sigma': 40}


def find_blobs_parallel(image_list, processes=MAX_PROCS, **kwargs):
    pool = mp.pool.Pool(processes=processes)
    find_partial = functools.partial(find_blobs, **kwargs)
    results = pool.map(find_partial, image_list)
    pool.close()
    pool.join()
    return results


def stack_blob_bins(unw_file_list,
                    num_row_bins=10,
                    num_col_bins=10,
                    save_file='all_blobs.npy',
                    weight_by_mag=True,
                    plot=True,
                    **kwargs):
    files_gen = (sario.load(f) * timeseries.PHASE_TO_CM for f in unw_file_list)
    b = BLOB_KWARG_DEFAULTS.copy()
    b.update(**kwargs)
    results = find_blobs_parallel(files_gen, **b)
    if save_file:
        np.save(save_file, results)
    nrows, ncols = sario.load(unw_file_list[0]).shape
    hist, row_edges, col_edges = bin_blobs(
        results, nrows, ncols, num_row_bins, num_col_bins, weight_by_mag=weight_by_mag)
    # if plot is True:
    # plot_hist(hist, row_edges, col_edges)
    return hist, row_edges, col_edges


# In [1]: amp_data = sario.load('20171218_20171230.amp')
#
# In [2]: img = np.load('blob_img.npy')
#
# In [3]: sario.save_hgt('height_test.unw', np.abs(amp_data), img)
def bin_blobs(list_of_blobs, nrows, ncols, num_row_bins=10, num_col_bins=10, weight_by_mag=True):
    """Make a 2D histogram of occurrences of row, col locations for blobs
    """
    row_edges = np.linspace(0, nrows, num_row_bins + 1)  # one more edges than bins
    col_edges = np.linspace(0, ncols, num_col_bins + 1)
    cumulative_hist = np.zeros((num_row_bins, num_col_bins))

    for blobs in list_of_blobs:
        if len(blobs) == 0:
            continue
        row_idxs = blobs[:, 0]
        col_idxs = blobs[:, 1]
        if weight_by_mag:
            weights = blobs[:, 3]
        else:
            weights = np.ones(blobs.shape[0])
        H, _, _ = np.histogram2d(row_idxs, col_idxs, bins=(row_edges, col_edges), weights=weights)
        cumulative_hist += H
    return cumulative_hist, row_edges, col_edges
