"""utils.py: Functions for finding blobs in deformation maps
"""
from __future__ import print_function
import collections
import itertools
import insar.utils
import insar.latlon
import numpy as np
import skimage
import cv2 as cv
# from scipy.spatial.qhull import ConvexHull
from shapely.geometry import Point, MultiPoint, Polygon, box


def indexes_within_circle(mask_shape=None, center=None, radius=None, blob=None):
    """Get a mask of indexes within a circle

    Args:
        center (tuple[float, float]): row, column of center of circle
        radius (float): radius of circle
        mask_shape (tuple[int, int]) rows, cols to make mask for entire image
        blob (tuple[float, float, float]): row, col, radius of blob
            This option is instead of using `center` and `radius`
    Returns:
       np.array[bool]: boolean mask of size `mask_shape`
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

    # blob: [row, col, radius, [possibly mag]]
    masks = map(lambda blob: indexes_within_circle(blob=blob, mask_shape=image.shape), blobs)
    return np.stack([accum_func(image[mask]) for mask in masks])


def mask_border(mask):
    rows = np.where(np.nanargmax(mask, axis=1))[0]
    cols = np.where(np.nanargmax(mask, axis=0))[0]
    return np.min(rows), np.max(rows), np.min(cols), np.max(cols)


def crop_image_to_mask(image, mask):
    masked_out = image.copy()
    masked_out[~mask] = np.nan
    min_row, max_row, min_col, max_col = mask_border(mask)
    return masked_out[min_row:max_row + 1, min_col:max_col + 1]


def crop_blob(image, blob):
    """Crops an image to the box around a blob with nans outside blob area
    Args:
        image:
        blob: (row, col, radius, ...)

    Returns:
        ndarray: size = (2r, 2r), r = radius of blob
    """
    mask = indexes_within_circle(blob=blob, mask_shape=image.shape)
    return crop_image_to_mask(image, mask)


def get_dist_to_extreme(image, blob, positive=True):
    """Finds how far from center a blobs extreme point is that caused the result

    Assumes blob[2] is radius, not sigma, to pass to indexes_within_circle

    Returns as a ratio distance/blob_radius (0 to 1)
    """
    patch = crop_blob(image, blob)
    # Remove any bias to just look at peak difference
    if positive:
        patch += np.nanmin(patch)
        row, col = np.unravel_index(np.nanargmax(patch), patch.shape)
    else:
        patch -= np.nanmax(patch)
        row, col = np.unravel_index(np.nanargmin(patch), patch.shape)

    nrows, _ = patch.shape  # circle, will have same rows and cols
    midpoint = nrows // 2
    dist = np.sqrt((row - midpoint)**2 + (col - midpoint)**2)
    return dist / blob[2]


def append_stats(blobs, image, stat_funcs=(np.var, np.ptp), center_only=False):
    """Append columns based on the statistic functions in stats

    Default: adds the variance and peak-to-peak within blob

    Args:
        center_only (bool, or array-list[bool]): pass to get_blob_stats
            Can either do 1 true false for all stat_funcs, or an iterable
            of matching size
    """
    new_blobs = blobs.copy()
    if isinstance(center_only, collections.Iterable):
        center_iter = center_only
    else:
        center_iter = itertools.repeat(center_only)

    for func, center in zip(stat_funcs, center_iter):
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


def find_sigma_idxs(blobs, sigma_list):
    """Finds which sigma each blob uses by its index in sigma_list

    Assumes blobs already like (r, c, radius,...), where radius=sqrt(2) * sigma"""
    idxs = np.searchsorted(sigma_list, blobs[:, 2] / np.sqrt(2), 'right')
    # Clip in case we are passed something larger than any sigma_list
    return np.clip(idxs, 0, len(sigma_list) - 1)


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


def img_as_uint8(img, vmin=None, vmax=None):
    # Handle invalids with masked array, set it to 0
    out = np.ma.masked_invalid(img).filled(0)
    if vmin is not None or vmax is not None:
        out = np.clip(out, vmin, vmax)
    return skimage.img_as_ubyte(skimage.exposure.rescale_intensity(out))


def cv_bbox_to_extent(bbox):
    """convert opencv (top x, top y, w, h) to (left, right, bot, top)

    For use in latlon.intersection_over_union"""
    x, y, w, h = bbox
    # Note: these are row, col pixels, but we still do y + h so that
    # top is a larger number
    return (x, x + w, y, y + h)


def bbox_to_coords(bbox, cv_format=False):
    if cv_format:
        bbox = cv_bbox_to_extent(bbox)
    left, right, bot, top = bbox
    return ((left, bot), (right, bot), (right, top), (left, top), (left, bot))


def regions_to_shapes(regions):
    return [Polygon(r) for r in regions]


def _box_is_bad(bbox, min_pix=3, max_ratio=5):
    """Returns true if (x, y, w, h) box is too small or w/h is too oblong"""
    x, y, w, h = bbox
    return w < min_pix or h < min_pix or w / h > max_ratio or h / w > max_ratio


# TODO: GET THE KM TO PIXEL CONVERSSERION
# TODO: maybe make a gauss pyramid, then only do small MSERs
# TODO: does this work on negative regions at same time as pos? try big path85
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
    """Takes in mser regions and bboxs, merges nested regions into the largest"""
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
        for jdx, sbox in enumerate(sorted_bboxes[idx + 1:], start=idx + 1):
            if jdx in eliminated_idxs:
                continue
            bsmall = cv_bbox_to_extent(sbox)
            if (insar.latlon.intersect_area(bbig, bsmall) /
                    insar.latlon.box_area(bsmall)) > overlap_thresh:
                eliminated_idxs.add(jdx)

    # Now get the non-eliminated indices
    all_idx = np.arange(len(sorted_bboxes))
    remaining = list(set(all_idx) - set(eliminated_idxs))
    # Converts to np.array to use fancy indexing
    return list(np.array(sorted_regions)[remaining]), np.array(sorted_bboxes)[remaining]
