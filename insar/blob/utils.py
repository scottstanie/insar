"""utils.py: Functions for finding blobs in deformation maps
"""
from __future__ import print_function
import insar.utils
import insar.latlon
import numpy as np
import cv2 as cv
from scipy.spatial.qhull import ConvexHull


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


def append_stats(blobs, image, stat_funcs=(np.var, np.ptp)):
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


def sort_blobs_by_val(blobs, image, positive=True):
    """Sort the blobs by their absolute value in the image

    Note: blobs must be in (row, col, sigma) form, not (lat, lon, sigma_ll)

    Returns:
        tuple[tuple[ndarrays], tuple[floats]]: The pair of (blobs, mags)
    """
    if positive:
        reverse = True
        func = np.max
    else:
        reverse = False
        func = np.min
    blob_vals = get_blob_stats(blobs, image, accum_func=func)
    blobs_with_mags = np.hstack((blobs, insar.utils.force_column(blob_vals)))
    # Sort rows based on the 4th column, blob_mag, and in reverse order
    return _sort_by_col(blobs_with_mags, 3, reverse=reverse)


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


def cv_bbox_to_extent(bbox):
    """convert opencv (top x, top y, w, h) to (left, right, bot, top)

    For use in latlon.intersection_over_union"""
    x, y, w, h = bbox
    # Note: these are row, col pixels, but we still do y + h so that
    # top is a larger number
    return (x, x + w, y, y + h)


def _box_is_bad(bbox, min_pix=3, max_ratio=5):
    """Returns true if (x, y, w, h) box is too small or w/h is too oblong"""
    x, y, w, h = bbox
    return w < min_pix or h < min_pix or w / h > max_ratio or h / w > max_ratio


def find_mser_regions(img, min_area=50):
    mser = cv.MSER_create()
    # TODO: get minarea, maxarea from some km set, convert to pixel
    mser.setMinArea(220)
    # mser.setMaxArea(220)
    regions, bboxes = mser.detectRegions(img)
    regions, bboxes = prune_regions(regions, bboxes, overlap=0.5)
    return regions, bboxes


def combine_hulls(points1, points2):
    # h = ConvexHull(np.vstack((points1, points2)))
    # Or maybe
    # h.add_points(points2)
    # h.close()
    h = ConvexHull(points1)
    h.add_points(points2)
    return h


def prune_regions(regions, bboxes, overlap_thresh=0.5):
    """Takes in mser regions and bboxs, prunes smaller overlapped regions"""
    # tup = (box, region)
    # bb = [cv_bbox_to_extent(b) for b in bboxes]
    sorted_bbox_regions = sorted(
        zip(bboxes, regions),
        key=lambda tup: insar.latlon.box_area(cv_bbox_to_extent(tup[0])),
        reverse=True)
    # Break apart again
    sorted_bboxes, sorted_regions = zip(*sorted_bbox_regions)

    # TODO: i'm just eliminating now (like nonmax suppression)... do i
    # want to merge by combining hulls?
    eliminated_idxs = set()
    # Start with current largest box, check all smaller for overlaps to eliminate
    for idx, big_box in enumerate(sorted_bboxes):
        if idx in eliminated_idxs:
            continue
        bbig = cv_bbox_to_extent(big_box)
        for jdx, box in enumerate(sorted_bboxes[idx + 1:], start=idx + 1):
            if jdx in eliminated_idxs:
                continue
            bsmall = cv_bbox_to_extent(box)
            if (insar.latlon.intersect_area(bbig, bsmall) /
                    insar.latlon.box_area(bsmall)) > overlap_thresh:
                eliminated_idxs.add(jdx)

    # Now get the non-eliminated indices
    all_idx = np.arange(len(sorted_bboxes))
    remaining = list(set(all_idx) - set(eliminated_idxs))
    # Converts to np.array to use fancy indexing
    return list(np.array(sorted_regions)[remaining]), np.array(sorted_bboxes)[remaining]
