"""Module of scoring functions for blob patches"""
import numpy as np
from math import sqrt
from insar.blob import utils as blob_utils
from insar.blob import get_center_value

def _sigma_from_patch(patch):
    """Back out what the sigma is based on size of patch

    Uses the fact that r = sqrt(2)*sigma
    """
    rows, _ = patch.shape
    radius = rows // 2
    return radius / sqrt(2)


def center_shape_index(patch, sigma_scale=None):
    """

    Args:
        patch:
        sigma_scale (float): number > 1 to divide the sigma
            of the patch by

    Returns:

    """
    sigma = _sigma_from_patch(patch)
    if sigma_scale:
        sigma /= sigma_scale
    get_center_value(shape_index(blob_utils.crop_blob(gauss_patch, blobs_gauss[0], crop_val=None), sigma=9), patch_size = 1)
    return