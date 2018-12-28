"""Module for finding blobs in deformation maps
"""
from __future__ import print_function
import os
import numpy as np
from insar.log import get_log
from . import skblob, utils, plot
from insar import latlon, plotting

logger = get_log()
BLOB_KWARG_DEFAULTS = {'threshold': 1, 'min_sigma': 3, 'max_sigma': 40}

# __all__ = ["BLOB_KWARG_DEFAULTS", "find_blobs"]


def find_blobs(image,
               include_values=True,
               negative=False,
               mag_threshold=1.0,
               min_sigma=3,
               max_sigma=60,
               threshold=0.5,
               **kwargs):
    """Find blob features within an image

    Args:
        image (ndarray): image containing blobs
        negative (bool): default False: if True, searchers for negative
            (dark, subsidence) blobs within image
        mag_threshold (float): absolute value in the image blob must exceed
            Should be positive number even if negative=True (since image is inverted)
        threshold (float): response threshold passed to the blob finding function
        min_sigma (int): minimum pixel size to check for blobs
        max_sigma (int): max pixel size to check for blobs

    Returns:
        ndarray: rows are blobs with values: [(r, c, s, mag)], where
        r = row num of center, c is column, s is sigma (size of Gaussian
        that detected blob), mag is the extreme value within the blob radius.

    Notes:
        kwargs are passed to the blob_log function (such as overlap).
        See reference for full list

    Reference:
    [1] http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html
    """

    image = -1 * image if negative else image
    # some skimage funcs fail for float32 when unnormalized [0,1]
    image = image.astype('float64')

    blobs = skblob.blob_log(
        image, threshold=threshold, min_sigma=min_sigma, max_sigma=max_sigma, **kwargs)

    if not blobs.size:  # Empty return: no blobs matched criteria
        return None

    # Multiply each sigma by sqrt(2) to convert to a radius
    blobs = blobs * np.array([1, 1, np.sqrt(2)])

    # Append mags as a column and sort by it
    blobs_with_mags = utils.sort_blobs_by_val(blobs, image)

    if mag_threshold:
        blobs_with_mags = blobs_with_mags[blobs_with_mags[:, -1] >= mag_threshold]

    # If negative, flip back last col to get correct img values
    if negative:
        blobs_with_mags = blobs_with_mags * np.array([1, 1, 1, -1])

    return blobs_with_mags


def _make_blobs(img, extra_args, verbose=False):
    blob_kwargs = BLOB_KWARG_DEFAULTS.copy()
    blob_kwargs.update(extra_args)
    logger.info("Using the following blob function settings:")
    logger.info(blob_kwargs)

    logger.info("Finding neg blobs")
    blobs_neg = find_blobs(img, negative=True, **blob_kwargs)

    logger.info("Finding pos blobs")
    blobs_pos = find_blobs(img, **blob_kwargs)

    logger.info("Blobs found:")
    if verbose:
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
    img = latlon.load_deformation_img(igram_path, n=3)
    img = img[row_start:row_end, col_start:col_end]
    # Note: now we use img.dem_rsc after cropping to keep track of new latlon bounds

    try:
        geolist = np.load(os.path.join(igram_path, 'geolist.npy'), encoding='bytes')
        title = "%s Deformation from %s to %s" % (title_prefix, geolist[0], geolist[-1])
    except FileNotFoundError:
        logger.warning("No geolist found in %s" % igram_path)
        title = "%s Deformation" % title_prefix

    imagefig, axes_image = plotting.plot_image_shifted(
        img, img_data=img.dem_rsc, title=title, xlabel='Longitude', ylabel='Latitude')
    # Or without lat/lon data:
    # imagefig, axes_image = plotting.plot_image_shifted(img, title=title)

    if load and os.path.exists(blob_filename):
        print("Loading %s" % blob_filename)
        blobs = np.load(blob_filename)
    else:
        extra_args = _handle_args(blobfunc_args)
        blobs = _make_blobs(img, extra_args)
        print("Saving %s" % blob_filename)
        np.save(blob_filename, blobs)

    blobs_ll = utils.blobs_to_latlon(blobs, img.dem_rsc)
    if verbose:
        for lat, lon, r, val in blobs_ll:
            logger.info('({0:.4f}, {1:.4f}): radius: {2}, val: {3}'.format(lat, lon, r, val))

    plot.plot_blobs(blobs=blobs_ll, cur_axes=imagefig.gca())
    # plot_blobs(blobs=blobs, cur_axes=imagefig.gca())


def _handle_args(extra_args):
    """Convert command line args into function-usable strings

    '--num-sigma' gets processed into num_sigma
    """
    keys = [arg.lstrip('--').replace('-', '_') for arg in list(extra_args)[::2]]
    vals = []
    for val in list(extra_args)[1::2]:
        try:
            vals.append(float(val))
        except ValueError:
            vals.append(val)
    return dict(zip(keys, vals))
