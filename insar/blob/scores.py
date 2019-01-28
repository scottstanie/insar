"""Module of scoring functions for blob patches"""
import numpy as np
from math import sqrt
from scipy import ndimage as ndi
from insar.blob import utils as blob_utils
from insar.blob.skblob import shape_index


def shape_index_stat(patch, accum_func, sigma=None, sigma_scale=None, patch_size='auto'):
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
        print('sfp', sigma)
    # if they want to scale down the default (which smooths proportional to the blob sigma)
    if sigma_scale:
        sigma /= sigma_scale

    if patch_size == 'auto':
        # 2 * because patch size excepts height, not radius
        psize = int(2 * max(3, sigma / 5))
    elif patch_size == 'full':
        psize = patch.shape[0]
    else:
        psize = patch_size
    if not isinstance(psize, int):
        raise ValueError('patch_size must be int')
    print('psize', psize)

    # For all these functions, grab the image of shape indexes to use
    shape_index_arr = shape_index(patch, sigma=sigma)
    return blob_utils.get_center_value(shape_index_arr, patch_size=psize, accum_func=accum_func)


def shape_index_center(patch, patch_size=3, sigma=None):
    """Finds the mean shape_index of a 3x3 patch around center pixel"""
    return shape_index_stat(patch, np.mean, sigma=sigma, patch_size=patch_size)


def shape_index_variance_small(patch, sigma=3, patch_size=3):
    """Smooth by a small sized filter, look in a small window at variance"""
    return shape_index_stat(patch, np.var, sigma=sigma, patch_size=patch_size)


def shape_index_variance_full(patch, sigma_scale=None, patch_size='full'):
    """Smooth over a large sigma equal to blob sigma, take variance over all"""
    return shape_index_stat(patch, np.var, patch_size=patch_size)


def shape_index_ptp_small(patch, sigma=None, sigma_scale=None):
    """Look in a small window for large changes in the shape index peak-to-peak"""
    return shape_index_stat(patch, np.ptp, sigma=sigma, sigma_scale=sigma_scale, patch_size=3)


def shape_index_ptp_full(patch, sigma=None, sigma_scale=None):
    """Look in a total patch for large changes in the shape index peak-to-peak"""
    return shape_index_stat(patch, np.ptp, sigma=sigma, sigma_scale=sigma_scale, patch_size='full')


def max_gradient(patch):
    imy = np.abs(ndi.sobel(patch, axis=0, mode='nearest'))
    imx = np.abs(ndi.sobel(patch, axis=1, mode='nearest'))
    return max(np.max(imx), np.max(imy))


FUNC_LIST = [
    shape_index_center,
    shape_index_variance_small,
    shape_index_variance_full,
    shape_index_ptp_small,
    shape_index_ptp_full,
]

