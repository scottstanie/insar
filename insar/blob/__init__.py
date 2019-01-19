"""Module for finding blobs in deformation maps
"""
from __future__ import print_function
import os
import multiprocessing
import numpy as np
from insar.log import get_log
from . import skblob, utils, plot
from skimage import feature
from insar import latlon, plotting

logger = get_log()
BLOB_KWARG_DEFAULTS = {'threshold': 1, 'min_sigma': 3, 'max_sigma': 40}

# __all__ = ["BLOB_KWARG_DEFAULTS", "find_blobs"]


# TODO: clean up redundancies in this function
def find_blobs(image,
               positive=True,
               negative=True,
               threshold=0.5,
               mag_threshold=None,
               min_sigma=3,
               max_sigma=60,
               num_sigma=20,
               sigma_bins=1,
               prune_edges=True,
               prune_border=2,
               log_scale=False,
               **kwargs):
    """Find blob features within an image

    Args:
        image (ndarray): image containing blobs
        positive (bool): default True: if True, searches for positive (light, uplift)
            blobs within image
        negative (bool): default True: if True, finds dark, subsidence blobs
        mag_threshold (float): absolute value in the image blob must exceed
            Should be positive number even if negative=True (since image is inverted)
        threshold (float): response threshold passed to the blob finding function
        min_sigma (int): minimum pixel size to check for blobs
        max_sigma (int): max pixel size to check for blobs
        num_sigma : int, optional: number of intermediate values of filter size to use
        sigma_bins : int or array-like of edges: Will only prune overlapping
            blobs that are within the same bin (to keep big very large and nested
            small blobs). int passed will divide range evenly
        prune_edges (bool):  will look for "ghost blobs" near strong extrema to remove,
        prune_border (int): Blobs with centers within `prune_border` pixels of
            image borders will be discarded
        log_scale : bool, optional
            If set intermediate values of standard deviations are interpolated
            using a logarithmic scale. If not, linear

    Returns:
        blobs: ndarray: rows are blobs with values: [(r, c, s, mag)], where
            r = row num of center, c is column, s is sigma (size of Gaussian
            that detected blob), mag is the extreme value within the blob radius.
        sigma_list: ndarray
            array of sigma values used to filter scale space

    Notes:
        kwargs are passed to the blob_log function (such as overlap).
        See reference for full list

    Reference:
    [1] http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html
    """
    # some skimage funcs fail for float32 when unnormalized [0,1]
    image = image.astype('float64')
    sigma_list = skblob.create_sigma_list(
        min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, log_scale=log_scale)
    image_cube = skblob.create_gl_cube(image, sigma_list=sigma_list)

    blobs = np.empty((0, 4))
    if positive:

        print('bpos')
        blobs_pos = skblob.blob_log(
            threshold=threshold,
            image_cube=image_cube,
            sigma_list=sigma_list,
            sigma_bins=sigma_bins,
            **kwargs)
        # Append mags as a column and sort by it
        # TODO: FIX vvv
        if blobs_pos.size:
            blobs_with_mags = utils.sort_blobs_by_val(blobs_pos, image, positive=True)
        else:
            blobs_with_mags = np.empty((0, 4))
        # print(blobs_with_mags)
        if mag_threshold is not None:
            blobs_with_mags = blobs_with_mags[blobs_with_mags[:, -1] >= mag_threshold]
        if prune_edges:
            blobs_with_mags = skblob.prune_edge_extrema(image, blobs_with_mags, positive=True)
        if prune_border > 0:
            blobs_with_mags = prune_border_blobs(image.shape, blobs_with_mags, prune_border)
        blobs = np.vstack((blobs, blobs_with_mags))
    if negative:
        print('bneg')
        blobs_neg = skblob.blob_log(
            threshold=threshold,
            image_cube=-1 * image_cube,
            sigma_list=sigma_list,
            sigma_bins=sigma_bins,
            **kwargs)
        if blobs_neg.size:
            blobs_with_mags = utils.sort_blobs_by_val(blobs_neg, image, positive=False)
        else:
            blobs_with_mags = np.empty((0, 4))
        # print(blobs_with_mags)
        if mag_threshold is not None:
            blobs_with_mags = blobs_with_mags[-1 * blobs_with_mags[:, -1] >= mag_threshold]
        if prune_edges:
            blobs_with_mags = skblob.prune_edge_extrema(image, blobs_with_mags, positive=False)
        if prune_border > 0:
            blobs_with_mags = prune_border_blobs(image.shape, blobs_with_mags, prune_border)
        blobs = np.vstack((blobs, blobs_with_mags))

    # Multiply each sigma by sqrt(2) to convert sigma to a circle radius
    blobs = blobs * np.array([1, 1, np.sqrt(2), 1])

    return blobs, sigma_list


def _make_blobs(image, extra_args, verbose=False):
    blob_kwargs = BLOB_KWARG_DEFAULTS.copy()
    blob_kwargs.update(extra_args)
    logger.info("Using the following blob function settings:")
    logger.info(blob_kwargs)

    logger.info("Finding neg blobs")
    blobs, _ = find_blobs(image, positive=True, negative=True, **blob_kwargs)

    logger.info("Blobs found:")
    if verbose:
        logger.info(blobs)
    return blobs


def make_blob_image(igram_path=".",
                    load=True,
                    title_prefix='',
                    blob_filename='blobs.npy',
                    row_start=0,
                    row_end=-1,
                    col_start=0,
                    col_end=-1,
                    verbose=False,
                    blobfunc_args=()):
    """Find and view blobs in deformation"""

    logger.info("Searching %s for igram_path" % igram_path)
    image = latlon.load_deformation_img(igram_path, n=3)
    image = image[row_start:row_end, col_start:col_end]
    # Note: now we use image.dem_rsc after cropping to keep track of new latlon bounds

    try:
        geolist = np.load(os.path.join(igram_path, 'geolist.npy'), encoding='bytes')
        title = "%s Deformation from %s to %s" % (title_prefix, geolist[0], geolist[-1])
    except FileNotFoundError:
        logger.warning("No geolist found in %s" % igram_path)
        title = "%s Deformation" % title_prefix

    imagefig, axes_image = plotting.plot_image_shifted(
        image, img_data=image.dem_rsc, title=title, xlabel='Longitude', ylabel='Latitude')
    # Or without lat/lon data:
    # imagefig, axes_image = plotting.plot_image_shifted(image, title=title)

    if load and os.path.exists(blob_filename):
        print("Loading %s" % blob_filename)
        blobs = np.load(blob_filename)
    else:
        blobs = _make_blobs(image, blobfunc_args)
        print("Saving %s" % blob_filename)
        np.save(blob_filename, blobs)

    blobs_ll = utils.blobs_to_latlon(blobs, image.dem_rsc)
    if verbose:
        for lat, lon, r, val in blobs_ll:
            logger.info('({0:.4f}, {1:.4f}): radius: {2}, val: {3}'.format(lat, lon, r, val))

    plot.plot_blobs(blobs=blobs_ll, cur_axes=imagefig.gca())
    # plot_blobs(blobs=blobs, cur_axes=imagefig.gca())


def compute_harris_peaks(image, sigma_list, gamma=1.4, threshold_rel=0.1):
    """Computes harris corner response on image at each sigma, finds peaks

    Args:
        image (ndarray): input image to compute corner harris reponse
        sigma_list (array-like): output of create_sigma_list
        gamma (float): adjustment from sigma in LoG to harris
            The Gaussian kernel for the LoG scale space (t) is smaller
            than the window used to pre-smooth and find the Harris response (s),
            where s = t * gamma**2
        threshold_rel (float): passed to find peaks. Using relative since
            smaller sigmas for corner_harris have higher

    Returns:
        peaks: output of peak_local_max on stack of corner responses
        corner_img_list: the corner response at each level

    TODO: see if doing corner_harris * s**2 to normalize, and using threshold_abs
        works better than threshold_rel

    Sources:
    https://en.wikipedia.org/wiki/Corner_detection#The_multi-scale_Harris_operator
    """
    # TODO: parallelize this
    # corner_img_list = [feature.corner_harris(image, sigma=s * gamma) for s in sigma_list]
    # peaks = [
    #     skblob.peak_local_max(corner_img, threshold_rel=threshold_rel)
    #     for corner_img in corner_img_list
    # ]
    pool = multiprocessing.Pool()
    jobs = []
    for s in sigma_list:
        jobs.append(pool.apply_async(feature.corner_harris, (image, ), {'sigma': s * gamma}))
    corner_img_list = [s**2 * result.get() for result, s in zip(jobs, sigma_list)]

    jobs = []
    for corner_img in corner_img_list:
        jobs.append(
            pool.apply_async(skblob.peak_local_max, (corner_img, ),
                             {'threshold_rel': threshold_rel}))
    peaks = [result.get() for result in jobs]

    return peaks, corner_img_list


def find_blobs_with_harris_peaks(image,
                                 blobs=None,
                                 sigma_list=None,
                                 gamma=1.4,
                                 threshold_rel=0.1,
                                 **kwargs):
    """Takes the list of blobs found from find_blobs, check for high cornerness

    Computes a harris corner response at each level gamma*sigma_list, finds
    peaks, then checks if blobs at that sigma level have some corner inside.
    Blobs with no corner peak found are discarded (they are edges or ghost
    blobs found at the ring of sharp real blobs)

    Args:
        image (ndarray): input image to compute corners on
        blobs (ndarray): rows are blobs with values: [(r, c, s, ...)]
        sigma_list (array-like): output of create_sigma_list
        gamma (float): adjustment from sigma in LoG to harris
        threshold_rel (float): passed to find peaks to threshold real peaks.
        kwargs: passed to find_blobs if `blobs` not passed as argument

    Returns:
        ndarray: like blobs, with some rows deleted that contained no corners
    """
    if blobs is None:
        blobs, sigma_list = find_blobs(image, **kwargs)

    # Find peaks for every sigma in sigma_list
    corner_peaks, _ = compute_harris_peaks(
        image, sigma_list, gamma=gamma, threshold_rel=threshold_rel)

    sigma_idxs = utils.find_sigma_idxs(blobs, sigma_list)
    # import ipdb
    # ipdb.set_trace()
    out_blobs = []
    for blob, sigma_idx in zip(blobs, sigma_idxs):
        # Get the peaks that correspond to the currend sigma level
        cur_peaks = corner_peaks[sigma_idx]
        blob_mask = utils.indexes_within_circle(blob=blob, mask_shape=image.shape)
        corners_contained_in_mask = blob_mask[cur_peaks[:, 0], cur_peaks[:, 1]]
        # corners_contained_in_mask = blob_mask[cur_peaks[:, 1], cur_peaks[:, 0]]
        if any(corners_contained_in_mask):
            out_blobs.append(blob)

    return np.array(out_blobs)
