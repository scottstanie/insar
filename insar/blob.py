"""blob.py: Functions for finding blobs in deformation maps
"""
from __future__ import print_function
import functools
import os
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
# Note: This is just a temp stopgap to not make skimage a hard requirement
# In the future, will port just the blob function, ski rest of skimage
try:
    import skimage.feature
except ImportError:
    print("Warning: scikit-image not installed. Blob function not available.")
    print("pip install scikit-image")
    pass
import sardem
from insar.log import get_log
from insar import latlon, plotting, timeseries, sario, utils

logger = get_log()
MAX_PROCS = mp.cpu_count()
BLOB_KWARG_DEFAULTS = {'threshold': 1, 'min_sigma': 3, 'max_sigma': 40}


def find_blobs(image,
               blob_func='blob_log',
               include_values=True,
               negative=False,
               value_threshold=1.0,
               min_sigma=3,
               max_sigma=60,
               threshold=0.5,
               **kwargs):
    """Use skimage to find blobs in image

    Note: when looking for negative blobs, you should pass in -image,
    as the sorter performs a `max` to find the largest value within the
    radius of the blob

    Args:
        image (ndarray): image containing blobs
        blob_func (str): which of the functions to use to find blobs
            Options: 'blob_log', 'blob_dog', 'blob_doh'
        negative (bool): default False: if True, multiplies image by -1 and
            searches for negative blobs in the image
        value_threshold (float): absolute value in the image blob must exceed
            Should be positive number even if negative=True (since image is inverted)
        threshold (float): response threshold passed to the blob finding function
        min_sigma (int): minimum pixel size to check for blobs
        max_sigma (int): max pixel size to check for blobs

    Returns:
        ndarray: rows are blobs with values: [(r, c, s, value)], where
        r = row num of center, c is column, s is sigma (size of Gaussian
        that detected blob), value is the extreme value within the blob radius.

    Notes:
        kwargs are passed to the blob_func (such as overlap).
        See reference for full list

    Reference:
    [1] http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html
    """

    image = -1 * image if negative else image
    image = image.astype('float64')  # skimage fails for float32 when unnormalized

    blob_func = getattr(skimage.feature, blob_func)
    blobs = blob_func(
        image, threshold=threshold, min_sigma=min_sigma, max_sigma=max_sigma, **kwargs)

    if not blobs.size:  # Empty return: no blobs matched criteria
        return None

    # Multiply each sigma by sqrt(2) to convert to a radius
    blobs = blobs * np.array([1, 1, np.sqrt(2)])

    # Append values as a column and sort by it
    blobs_with_values = sort_blobs_by_val(blobs, image)

    if value_threshold:
        blobs_with_values = blobs_with_values[blobs_with_values[:, -1] >= value_threshold]

    # If negative, flip back last col to get correct img values
    if negative:
        blobs_with_values = blobs_with_values * np.array([1, 1, 1, -1])

    return blobs_with_values


def find_blobs_parallel(image_list, processes=MAX_PROCS, **kwargs):
    pool = mp.pool.Pool(processes=processes)
    find_partial = functools.partial(find_blobs, **kwargs)
    results = pool.map(find_partial, image_list)
    pool.close()
    pool.join()
    return results


def plot_blobs(image, blobs=None, cur_fig=None, cur_axes=None, color='blue', **kwargs):
    """Takes the blob results from find_blobs and overlays on image

    Can either make new figure of plot on top of existing axes.
    """
    if cur_fig:
        cur_axes = cur_fig.gca()
    elif not cur_axes:
        cur_fig = plt.figure()
        cur_axes = cur_fig.gca()
        cur_axes.imshow(image)

    if blobs is None:
        logger.info("Searching for blobs in image.")
        blobs = find_blobs(image, **kwargs)

    for blob in blobs:
        c = plt.Circle(
            (blob[1], blob[0]), blob[2], color=color, fill=False, linewidth=2, clip_on=False)
        cur_axes.add_patch(c)

    plt.draw()
    plt.show()
    return blobs, cur_axes


def indexes_within_circle(cx, cy, radius, height, width):
    """Get a mask of indexes within a circle"""
    X, Y = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    return dist_from_center <= radius


def get_blob_values(blobs, image, center_only=False):
    """Finds most extreme image value within each blob

    Checks all pixels within the radius of the blob

    Args:
        blobs (ndarray): 2D, entries [row, col, radius], from find_blobs
        image (ndarray): 2D image where blobs were found
        center_only (bool): (default False) Only look at the center pixel of the blob.
            False means searching all pixels within the blob radius on the image

    Returns:
        ndarray: length = number of blobs, each value is the max of the image
        within the blob radius
    """
    if center_only:
        coords = blobs[:, :2].astype(int)
        return image[coords[:, 0], coords[:, 1]]

    height, width = image.shape
    # blob: [row, col, radius, [possibly value]]
    masks = map(lambda blob: indexes_within_circle(blob[0], blob[1], blob[2], height, width), blobs)
    return np.stack([np.max(image[mask]) for mask in masks])


def sort_blobs_by_val(blobs, image):
    """Sort the blobs by their absolute value in the image

    Note: blobs must be in (row, col, sigma) form, not (lat, lon, sigma_ll)

    Returns:
        tuple[tuple[ndarrays], tuple[floats]]: The pair of (blobs, values)
    """
    blob_vals = get_blob_values(blobs, image)
    blobs_with_values = np.hstack((blobs, utils.force_column(blob_vals)))
    # Sort rows based on the 4th column, blob_value, and in reverse order
    return blobs_with_values[blobs_with_values[:, 3].argsort()[::-1]]


def blobs_latlon(blobs, blob_info):
    """Converts (y, x, sigma, val) format to (lat, lon, sigma_latlon, val)

    Uses the dem x_step/y_step data to rescale blobs so that appear on an
    image using lat/lon as the `extent` argument of imshow.
    """
    blob_info = {k.lower(): v for k, v in blob_info.items()}
    blobs_latlon = []
    for blob in blobs:
        row, col, r, val = blob
        lat, lon = latlon.rowcol_to_latlon(row, col, blob_info)
        new_radius = r * blob_info['x_step']
        blobs_latlon.append((lat, lon, new_radius, val))

    return np.array(blobs_latlon)


def _handle_args(extra_args):
    keys = [arg.lstrip('--').replace('-', '_') for arg in list(extra_args)[::2]]
    vals = []
    for val in list(extra_args)[1::2]:
        try:
            vals.append(float(val))
        except ValueError:
            vals.append(val)
    return dict(zip(keys, vals))


def _make_blobs(img, extra_args):
    blob_kwargs = BLOB_KWARG_DEFAULTS.copy()
    blob_kwargs.update(extra_args)
    logger.info("Using the following blob function settings:")
    logger.info(blob_kwargs)

    logger.info("Finding neg blobs")
    blobs_neg = find_blobs(img, negative=True, **blob_kwargs)

    logger.info("Finding pos blobs")
    blobs_pos = find_blobs(img, **blob_kwargs)

    logger.info("Blobs found:")
    logger.info(blobs_neg)
    logger.info(blobs_pos)
    # Skip empties
    return np.vstack((b for b in (blobs_neg, blobs_pos) if b is not None))


def make_blob_image(igram_path=".",
                    load=True,
                    title_prefix='',
                    blob_filename='blobs.npy',
                    row_start=0,
                    row_end=-1,
                    col_start=0,
                    col_end=-1,
                    verbose=False,
                    blobfunc_args=None):
    """Find and view blobs in deformation"""

    logger.info("Searching %s for igram_path" % igram_path)
    geolist, deformation = timeseries.load_deformation(igram_path)
    rsc_data = sardem.loading.load_dem_rsc(os.path.join(igram_path, 'dem.rsc'))

    # TODO: Is mean/max better than just looking at last image? probably
    # MAKE OPTION FOR THE COMMENTED PARTS
    # img = deformation[-1, row_start:row_end, col_start:col_end]
    # img = np.mean(deformation[-3:, row_start:row_end, col_start:col_end], axis=0)
    img = latlon.LatlonImage(data=np.mean(deformation[-3:], axis=0), dem_rsc=rsc_data)
    img = img[row_start:row_end, col_start:col_end]
    # Note: now we use img.dem_rsc after cropping to keep track of new latlon bounds

    title = "%s Deformation from %s to %s" % (title_prefix, geolist[0], geolist[-1])
    imagefig, axes_image = plotting.plot_image_shifted(
        img, img_data=img.dem_rsc, title=title, xlabel='Longitude', ylabel='Latitude')
    # imagefig, axes_image = plotting.plot_image_shifted(img, title=title)

    blob_filename = 'blobs.npy'

    if load and os.path.exists(blob_filename):
        blobs = np.load(blob_filename)
    else:
        extra_args = _handle_args(blobfunc_args)
        blobs = _make_blobs(img, extra_args)
        np.save(blob_filename, blobs)

    blobs_ll = blobs_latlon(blobs, img.dem_rsc)
    if verbose:
        for lat, lon, r, val in blobs_ll:
            logger.info('({0:.4f}, {1:.4f}): radius: {2}, val: {3}'.format(lat, lon, r, val))

    plot_blobs(img, blobs=blobs_ll, cur_axes=imagefig.gca())
    # plot_blobs(img, blobs=blobs, cur_axes=imagefig.gca())


def stack_blob_bins(unw_file_list,
                    num_row_bins=10,
                    num_col_bins=10,
                    save_file='all_blobs.npy',
                    weight_by_value=True,
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
        results, nrows, ncols, num_row_bins, num_col_bins, weight_by_value=weight_by_value)
    if plot is True:
        plot_hist(hist, row_edges, col_edges)
    return hist, row_edges, col_edges


# In [1]: amp_data = sario.load('20171218_20171230.amp')
#
# In [2]: img = np.load('blob_img.npy')
#
# In [3]: sario.save_hgt('height_test.unw', np.abs(amp_data), img)


def bin_blobs(list_of_blobs, nrows, ncols, num_row_bins=10, num_col_bins=10, weight_by_value=True):
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
        if weight_by_value:
            weights = blobs[:, 3]
        else:
            weights = np.ones(blobs.shape[0])
        H, _, _ = np.histogram2d(row_idxs, col_idxs, bins=(row_edges, col_edges), weights=weights)
        cumulative_hist += H
    return cumulative_hist, row_edges, col_edges


def plot_hist(H, row_edges, col_edges, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    axes_image = ax.imshow(H, extent=[col_edges[0], col_edges[-1], row_edges[-1], row_edges[0]])
    fig.colorbar(axes_image)
    return fig, ax
