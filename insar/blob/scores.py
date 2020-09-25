"""Module of scoring functions for blob patches"""
import numpy as np
from math import sqrt
from scipy import ndimage as ndi
from insar.blob import utils as blob_utils
from insar.blob.skblob import shape_index


def shape_index_stat(
    patch, accum_func, sigma=None, sigma_scale=None, patch_size="auto"
):
    """Finds some state about the shape_index of a patch

    Args:
        patch: ndarray, pixel box around blob center
        sigma (float): specific size gaussian to smooth by
        sigma_scale (float): number > 1 to divide the sigma
            of the patch by
        patch_size (int or str): choices are 'auto', 'full', or int
            'auto' takes the size of the blob and divides
            by 5 for a proportional patch
            'full' takes patch size of sigma
            int will use that patch size

    Returns:
        float: score of shape_index within patch size
    """
    # If they don't specify one sigma, use patch size
    if sigma is None:
        sigma = blob_utils.sigma_from_blob(patch=patch)
        # print('sfp', sigma)
    # if they want to scale down the default (which smooths proportional to the blob sigma)
    if sigma_scale:
        sigma /= sigma_scale

    if patch_size == "auto":
        # 2 * because patch size excepts height, not radius
        psize = int(2 * max(3, sigma / 5))
    elif patch_size == "full":
        psize = patch.shape[0]
    else:
        psize = patch_size
    if not isinstance(psize, int):
        raise ValueError("patch_size must be int")
    # print('psize', psize)

    # For all these functions, grab the image of shape indexes to use
    shape_index_arr = shape_index(patch, sigma=sigma)
    return blob_utils.get_center_value(
        shape_index_arr, patch_size=psize, accum_func=accum_func
    )


def shape_index_center(patch):
    """Finds the mean shape_index of a 3x3 patch around center pixel
    sigma=proportional to the patch size computed with sigma_from_patch"""
    return shape_index_stat(patch, np.mean, sigma=None, patch_size=3)


def shape_index_center_sigma1(patch):
    """Finds the mean shape_index of a 3x3 patch around center pixel, sigma=1"""
    return shape_index_stat(patch, np.mean, sigma=1, patch_size=3)


def shape_index_center_sigma3(patch):
    """Finds the mean shape_index of a 3x3 patch around center pixel, sigma=3"""
    return shape_index_stat(patch, np.mean, sigma=3, patch_size=3)


def shape_index_center_min_sigma3(patch):
    """Finds the min shape_index of a 3x3 patch around center pixel, sigma=3"""
    return shape_index_stat(patch, lambda x: np.min(np.mean(x)), sigma=3, patch_size=3)


def shape_index_center_min_sigma1(patch):
    """Finds the min shape_index of a 3x3 patch around center pixel, sigma=3"""
    return shape_index_stat(patch, lambda x: np.min(np.mean(x)), sigma=1, patch_size=1)


def shape_index_variance_patch3_sigma3(patch):
    """Smooth by a small sigma=3, look in a patch=3 at variance"""
    return shape_index_stat(patch, np.var, sigma=3, patch_size=3)


def shape_index_variance_patch_full_sigma3(patch):
    """Smooth by a small sigma=3, look at entire patch for variance"""
    return shape_index_stat(patch, np.var, sigma=3, patch_size="full")


def shape_index_variance_patch_full(patch):
    """Smooth over a large sigma equal to blob sigma, take variance over all patch"""
    return shape_index_stat(patch, np.var, sigma=None, patch_size="full")


def shape_index_ptp_patch3(patch):
    """Check peak-to-peak in patch=3, smoothing with sigma proportional"""
    return shape_index_stat(patch, np.ptp, sigma=None, patch_size=3)


def shape_index_ptp_patch_full(patch):
    """Look in a total patch for large changes in the shape index peak-to-peak"""
    return shape_index_stat(patch, np.ptp, sigma=None, patch_size="full")


def max_gradient(patch, sigma=0.5):
    p = blob_utils.gaussian_filter_nan(patch, sigma=sigma)
    imy = np.abs(ndi.sobel(patch, axis=0, mode="nearest"))
    imx = np.abs(ndi.sobel(patch, axis=1, mode="nearest"))
    return max(np.max(imx), np.max(imy))


def max_gradient_sigma3(patch):
    return max_gradient(patch, sigma=3)


FUNC_LIST = [
    shape_index_center,
    shape_index_center_sigma1,
    shape_index_center_sigma3,
    shape_index_center_min_sigma3,
    shape_index_center_min_sigma1,
    shape_index_variance_patch3_sigma3,
    shape_index_variance_patch_full_sigma3,
    shape_index_variance_patch_full,
    shape_index_ptp_patch3,
    shape_index_ptp_patch_full,
    max_gradient,
    max_gradient_sigma3,
]

FUNC_LIST_NAMES = [f.__name__ for f in FUNC_LIST]


def analyze_patches(patch_list, funcs=FUNC_LIST, *args, **kwargs):
    """Get scores from functions on a series of patches

    Runs each function in `funcs` over each `patch` to get stats on it
    Each function must have a signature func(patch, *args, **kwargs),
        and return a single float number
    Args:
        patch_list:
        funcs:
        *args:
        **kwargs:

    Returns:
        ndarray: size (p, N) where p = num patches, N = len(funcs)
            rows are scores on one patch
    """
    results = []
    for patch in patch_list:
        results.append([func(patch, *args, **kwargs) for func in funcs])
    return np.array(results)
