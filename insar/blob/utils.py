"""utils.py: Functions for finding blobs in deformation maps
"""
from __future__ import print_function
import insar.utils
import insar.latlon
import numpy as np


def indexes_within_circle(mask_shape=None, center=None, radius=None, blob=None):
    """Get a mask of indexes within a circle

    Args:
        center (tuple[float, float]): row, column of center of circle
        radius (float): radius of circle
        mask_shape (tuple[int, int]) rows, cols to make enture mask
        blob (tuple[float, float, float]): row, col, radius of blob
            This option is instead of using `center` and `radius`
    """
    if mask_shape is None:
        raise ValueError("Need mask_shape to determine output array size")
    height, width = mask_shape
    if blob is not None:
        cy, cx, radius = blob[:3]
    elif center is not None:
        cy, cx = center
    if radius is None:
        raise ValueError("Need radius if not using `blob` input")
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    return dist_from_center <= radius


def get_blob_stats(blobs, image, center_only=False, accum_func=np.max):
    """Find statistics about image values within each blob

    Checks all pixels within the radius of the blob, and runs some
    numpy function `accum_func` on these values

    Args:
        blobs (ndarray): 2D, entries [row, col, radius, ...], from find_blobs
        image (ndarray): 2D image where blobs were found
        center_only (bool): (default False) Only get the value of the center pixel of the blob
        accum_func (bool): (default np.max) Function to run on all pixels within blob to accumulate to one value

    Returns:
        ndarray: length = N, number of blobs, each value is the max of the image
        within the blob radius.
        If all_pixels = True, each entry is a list of pixel values
    """
    if center_only:
        coords = blobs[:, :2].astype(int)
        return image[coords[:, 0], coords[:, 1]]

    height, width = image.shape
    # blob: [row, col, radius, [possibly mag]]
    masks = map(lambda blob: indexes_within_circle(blob=blob, mask_shape=image.shape), blobs)
    return np.stack([accum_func(image[mask]) for mask in masks])


def append_stats(blobs, image, stat_funcs=[np.var, np.ptp]):
    """Append columns based on the statistic functions in stats

    Default: adds the variance and peak-to-peak within blob"""
    new_blobs = blobs.copy()
    for func in stat_funcs:
        blob_stat = get_blob_stats(new_blobs, image, accum_func=func)
        new_blobs = np.hstack((new_blobs, blob_stat.reshape((-1, 1))))
    return new_blobs


def _sort_by_col(arr, col, reverse=False):
    sorted_arr = arr[arr[:, col].argsort()]
    return sorted_arr[::-1] if reverse else sorted_arr


def sort_blobs_by_val(blobs, image):
    """Sort the blobs by their absolute value in the image

    Note: blobs must be in (row, col, sigma) form, not (lat, lon, sigma_ll)

    Returns:
        tuple[tuple[ndarrays], tuple[floats]]: The pair of (blobs, mags)
    """
    blob_vals = get_blob_stats(blobs, image, accum_func=np.max)
    blobs_with_mags = np.hstack((blobs, insar.utils.force_column(blob_vals)))
    # Sort rows based on the 4th column, blob_mag, and in reverse order
    return _sort_by_col(blobs_with_mags, 3, reverse=True)


def blobs_to_latlon(blobs, blob_info):
    """Converts (y, x, sigma, ...) format to (lat, lon, sigma_latlon, ...)

    Uses the dem x_step/y_step data to rescale blobs so that appear on an
    image using lat/lon as the `extent` argument of imshow.
    """
    blobs_latlon = []
    for blob in blobs:
        row, col, r = blob[:3]
        lat, lon = insar.latlon.rowcol_to_latlon(row, col, blob_info)
        new_radius = r * blob_info['x_step']
        blobs_latlon.append((lat, lon, new_radius) + tuple(blob[3:]))

    return np.array(blobs_latlon)


def blobs_to_rowcol(blobs, blob_info):
    """Converts (lat, lon, sigma, val) format to (row, col, sigma_latlon, val)

    Inverse of blobs_to_latlon function
    """
    blobs_rowcol = []
    for blob in blobs:
        lat, lon, r, val = blob
        lat, lon = insar.latlon.latlon_to_rowcol(lat, lon, blob_info)
        old_radius = r / blob_info['x_step']
        blobs_rowcol.append((lat, lon, old_radius, val))

    return np.array(blobs_rowcol)
