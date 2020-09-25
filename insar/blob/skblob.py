"""Functions transferred/modified from skimage.feature"""
from __future__ import division

import multiprocessing

import numpy as np
from scipy.ndimage import gaussian_laplace, maximum_filter, gaussian_filter
import math
from math import sqrt, log
from scipy import spatial
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from . import utils as blob_utils

# This basic blob detection algorithm is based on:
# http://www.cs.utah.edu/~jfishbau/advimproc/project1/ (04.04.2013)
# Theory behind: http://en.wikipedia.org/wiki/Blob_detection (04.04.2013)


def blob_log(
    image=None,
    min_sigma=3,
    max_sigma=60,
    num_sigma=20,
    image_cube=None,
    sigma_list=None,
    threshold=0.5,
    overlap=0.5,
    sigma_bins=1,
    prune_edges=True,
    border_size=2,
    positive=True,
    log_scale=False,
    verbose=0,
):
    """Finds blobs in the given grayscale image.

    Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.

    Args:
    image : 2D or 3D ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    min_sigma : float, optional
        The minimum standard deviation for Gaussian Kernel. Keep this low to
        detect smaller blobs.
    max_sigma : float, optional
        The maximum standard deviation for Gaussian Kernel. Keep this high to
        detect larger blobs.
    num_sigma : int, optional
        The number of intermediate values of standard deviations to consider
        between `min_sigma` and `max_sigma`.
    image_cube (ndarray): optional: 3D volume, output of create_gl_cube
        if provided, bypasses the creation of the response cube using
        sigma_list or min_sigma, max_sigma, num_sigma args
    sigma_list (ndarray): optional: bypasses the creation of sigma_list from
        min_sigma, max_sigma, num_sigma args
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities.
        For a gaussian with height 1, the filter response is 0.5. Thus,
        0.5 would filter blobs of lower height or worse shape
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    sigma_bins : int or array-like of edges: Will only prune overlapping
        blobs that are within the same bin (to keep big very large and nested
        small blobs). int passed will divide range evenly
    prune_edges (bool):  will look for "ghost blobs" near strong extrema to remove,
    border_size (int): Blobs with centers within `border_size` pixels of
        image borders will be discarded
    positive (bool): default True: if True, searches for positive (light, uplift)
    log_scale : bool, optional
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale . If not, linear
        interpolation is used.
    verbose (bool or int): level of verbose printing while blob finding
        0 or False means no printing, 1 prints out more local max steps, 2 more

    Returns:
    A (n, image.ndim + 1) ndarray:
        A 2d array with each row representing 3 values for a 2D image,
        and 4 values for a 3D image: ``(r, c, sigma)`` or ``(p, r, c, sigma)``
        where ``(r, c)`` or ``(p, r, c)`` are coordinates of the blob and
        ``sigma`` is the standard deviation of the Gaussian kernel which
        detected the blob.

    References
    .. [1] http://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian

    Examples:
    >>> from skimage import data, feature, exposure
    >>> import numpy as np; np.set_printoptions(legacy="1.13")
    >>> img = data.coins()
    >>> img = exposure.equalize_hist(img)  # improves detection
    >>> blob_log(img, threshold = .3)
    array([[ 266.        ,  115.        ,   11.88888889],
           [ 263.        ,  302.        ,   17.33333333],
           [ 263.        ,  244.        ,   17.33333333],
           [ 260.        ,  174.        ,   17.33333333],
           [ 198.        ,  155.        ,   11.88888889],
           [ 198.        ,  103.        ,   11.88888889],
           [ 197.        ,   44.        ,   11.88888889],
           [ 194.        ,  276.        ,   17.33333333],
           [ 194.        ,  213.        ,   17.33333333],
           [ 185.        ,  344.        ,   17.33333333],
           [ 128.        ,  154.        ,   11.88888889],
           [ 127.        ,  102.        ,   11.88888889],
           [ 126.        ,  208.        ,   11.88888889],
           [ 126.        ,   46.        ,   11.88888889],
           [ 124.        ,  336.        ,   11.88888889],
           [ 121.        ,  272.        ,   17.33333333],
           [ 113.        ,  323.        ,    1.        ]])

    Notes:
    The radius of each blob is approximately :math:`\sqrt{2}\sigma` for
    a 2-D image and :math:`\sqrt{3}\sigma` for a 3-D image.
    """
    if sigma_list is None:
        sigma_list = create_sigma_list(min_sigma, max_sigma, num_sigma, log_scale)
    if image_cube is None:
        image = image.astype(np.floating)
        image_cube = create_gl_cube(image, sigma_list)

    # Note: we have to use exclude_border=False so that it will output blobs at
    # first and last sigma values (they are part of the "border" of the cube
    local_maxima = peak_local_max(
        image_cube, threshold_abs=threshold, min_distance=1, exclude_border=False
    )

    # TODO: something here is hanging if a nan happens... causing all nans

    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3))
    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)
    # Convert the last index to its corresponding scale value
    lm[:, -1] = sigma_list[local_maxima[:, -1]]

    # Multiply each sigma by sqrt(2) to convert sigma to a circle radius
    lm = lm * np.array([1, 1, np.sqrt(2)])
    if verbose > 2:
        print("initial local max lm:")
        print(lm)
        print(lm.shape)
    # Now remove first the spatial border blobs
    if border_size > 0:
        lm = prune_border_blobs(image.shape, lm, border_size)
    if verbose > 1:
        print("lm post prune_border_blobs:")
        print(lm)
        print(lm.shape)

    # Next remove blobs that look like edges
    smoothed_image = gaussian_filter(image, sigma=3, mode="constant")
    if prune_edges:
        lm = prune_edge_extrema(smoothed_image, lm, positive=positive, smooth=True)
    if verbose > 0:
        print("lm post prune_edge_extrema:")
        print(lm)
        print(lm.shape)
    return prune_overlap_blobs(lm, overlap, sigma_bins=sigma_bins)


def create_sigma_list(
    min_sigma=1, max_sigma=50, num_sigma=20, log_scale=False, **kwargs
):
    """Make array of sigmas for scale-space.

    Example with log_scale:
        min_sigma=4, max_sigma=145
    array([  4.        ,   5.93776639,   8.81426744,  13.08426524,
        19.42282761,  28.83205326,  42.79949922,  63.53335704,
        94.31155807, 140.        ])
    """

    if log_scale:
        base = 2
        start, stop = log(min_sigma, base), log(max_sigma, base)
        sigma_list = np.logspace(start, stop, num_sigma, base=base)
    else:
        sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)
    return sigma_list


def create_gl_cube(
    image, sigma_list=None, min_sigma=1, max_sigma=50, num_sigma=20, log_scale=False
):
    """Compute gaussian laplace for a range of sigma on image

    Multiplying by s**2 provides scale invariance to Gaussian sizes
    Negative in front of filter flips the filter to be "upright cowboy hat"

    Can either pass a premade sigma_list from create_sigma_list, or pass the
    parameters to make a list
    """
    if sigma_list is None:
        sigma_list = create_sigma_list(min_sigma, max_sigma, num_sigma, log_scale)
        print(sigma_list)
    filtered = []

    # Run each convolution in a separate process
    pool = multiprocessing.Pool()
    for s in sigma_list:
        filtered.append(
            pool.apply_async(
                gaussian_laplace, args=(image, s), kwds={"mode": "reflect"}
            )
        )
    # Include -s**2 for scale invariance, searches for positive signal
    return np.stack(
        [-(s ** 2) * res.get() for res, s in zip(filtered, sigma_list)], axis=-1
    )
    # Old way:
    # gl_images = [ -gaussian_laplace(im, s)*s**2 for s in sigma_list]
    # return np.stack(gl_images, axis=-1)


def _compute_disk_overlap(d, r1, r2):
    """
    Compute surface overlap between two disks of radii ``r1`` and ``r2``,
    with centers separated by a distance ``d``.

    Args:
        d (float): Distance between centers.
        r1 (float): Radius of the first disk.
        r2 (float): Radius of the second disk.

    Returns:
        area (float): area of the overlap between the two disks.
    """

    ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    ratio1 = np.clip(ratio1, -1, 1)
    acos1 = math.acos(ratio1)

    ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
    ratio2 = np.clip(ratio2, -1, 1)
    acos2 = math.acos(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = r1 ** 2 * acos1 + r2 ** 2 * acos2 - 0.5 * sqrt(abs(a * b * c * d))
    return area


def _disk_area(r):
    return math.pi * r ** 2


def _sphere_vol(r):
    return 4.0 / 3 * math.pi * r ** 3


def _compute_sphere_overlap(d, r1, r2):
    """
    Compute volume overlap between two spheres of radii ``r1`` and ``r2``,
    with centers separated by a distance ``d``.

    Args:
        d : float Distance between centers.
        r1 : float Radius of the first sphere.
        r2 : float Radius of the second sphere.

    Returns:
        vol: float Volume of the overlap between the two spheres.

    Notes:
    See for example http://mathworld.wolfram.com/Sphere-SphereIntersection.html
    for more details.
    """
    vol = (
        math.pi
        / (12 * d)
        * (r1 + r2 - d) ** 2
        * (d ** 2 + 2 * d * (r1 + r2) - 3 * (r1 ** 2 + r2 ** 2) + 6 * r1 * r2)
    )  # yapf:disable
    return vol


def _blob_dist(blob1, blob2):
    return sqrt(np.sum((blob1[:2] - blob2[:2]) ** 2))


def blob_overlap(blob1, blob2):
    """Finds the overlapping area fraction between two blobs.

    Returns a float representing fraction of overlapped area.

    Args:
    blob1 (sequence of arrays):
        A sequence of ``(row, col, radius)``, where ``row, col`` are coordinates
        of blob and ``radius`` is sqrt(2)* standard deviation of the Gaussian kernel
        which detected the blob.
    blob2 (sequence of arrays): same type as blob1

    Returns:
        f (float): Fraction of overlapped area
    """
    r1 = blob1[2]
    r2 = blob2[2]

    d = _blob_dist(blob1, blob2)
    if d > r1 + r2:
        return 0

    # one blob is inside the other, the smaller blob must die
    if d <= abs(r1 - r2):
        return 1

    return _compute_disk_overlap(d, r1, r2) / _disk_area(min(r1, r2))
    # return _compute_sphere_overlap(d, r1, r2) / _sphere_vol(min(r1, r2))


def intersection_over_union(blob1, blob2, using_sigma=False):
    """Alternative measure of closeness of 2 blobs

    Used to see if guesses are close to real blob (x, y, radius)

    Args:
    blob1 (sequence of arrays):
        A sequence of ``(row, col, radius)``, where ``row, col`` are coordinates
        of blob and ``radius`` is sqrt(2)* standard deviation of the Gaussian kernel
        which detected the blob.
    blob2 (sequence of arrays): same type as blob1
    using_sigma (bool): default False. If False, blobs are ``(row, col, sigma)``
    instead of (row, col, radius)

    Returns:
        f (float): Fraction of overlapped area (or volume in 3D).

    Examples:
        >>> blob1 = np.array([0, 0, 6])
        >>> blob2 = np.array([0, 0, 3])
        >>> print(intersection_over_union(blob1, blob2))
        0.25
        >>> blob1 = np.array([0, 0, 1])
        >>> blob2 = np.array([0.8079455, 0, 1])
        >>> is_close = (intersection_over_union(blob1, blob2) - 0.472) < 1e3
        >>> print(is_close)
        True
    """
    # extent of the blob is given by sqrt(2)*scale if using sigma
    r1 = blob1[2] * sqrt(2) if using_sigma else blob1[2]
    r2 = blob2[2] * sqrt(2) if using_sigma else blob2[2]

    a1 = _disk_area(r1)
    a2 = _disk_area(r2)
    d = _blob_dist(blob1, blob2)
    if d > r1 + r2:
        return 0
    elif d <= abs(r1 - r2):  # One inside the other
        return _disk_area(min(r1, r2)) / _disk_area(max(r1, r2))

    intersection = _compute_disk_overlap(d, r1, r2)
    union = a1 + a2 - intersection
    return intersection / union


def prune_overlap_blobs(blobs_array, overlap, sigma_bins=1):
    """Eliminated blobs with area overlap.

    Args:
        blobs_array (ndarray): A 2d array with each row representing ``(row, col, radius)``
            where ``(row, col)`` are coordinates of the blob and ``radius`` is
            the sqrt(2) * standard deviation of the Gaussian kernel which
            detected the blob.
            This array must not have a dimension of size 0.
        overlap (float): A value between 0 and 1. If the fraction of area overlapping
            for 2 blobs is greater than `overlap` the smaller blob is eliminated.
        num_radius_bands (int): if provided, divides the sigma/radius range of blobs
            and only removes overlap within bands.
            For example, if you want to prune small to mid blobs, and keep bigger blobs
            that entirely overlap, set num_radius_bands = 2 or 3

    Returns
    -------
    A (ndarray): `array` with overlapping blobs removed.

    Examples:
        >>> import numpy as np; np.set_printoptions(legacy="1.13")
        >>> blobs = np.array([[ 0, 0,  4], [1, 1, 10], [2, 2, 20]])
        >>> print(prune_overlap_blobs(blobs, 0.5))
        [[ 2  2 20]]
        >>> print(prune_overlap_blobs(blobs, 0.5, sigma_bins=2))
        [[ 1  1 10]
         [ 2  2 20]]
        >>> print(prune_overlap_blobs(blobs, 0.5, sigma_bins=5))
        [[ 0  0  4]
         [ 1  1 10]
         [ 2  2 20]]
    """
    sigma_bins = int(sigma_bins)
    # If specified, divide into sigma bins, only prune within each range
    if sigma_bins > 1:
        out_blobs = []
        for b_arr in bin_blobs(blobs_array, sigma_bins):
            # Now recurse at bin level, then stack all together
            out_blobs.append(prune_overlap_blobs(b_arr, overlap, 1))
        if not np.array(out_blobs).size:
            return np.empty((0, blobs_array.shape[1]))
        return np.vstack([b for b in out_blobs if b.size])

    if not blobs_array.size:
        return blobs_array

    # Note: changed from blobs_array[:, -1] to blobs_array[:, 2]- assuming we have
    # only 2d blobs, but it may be passed a blobs_array with amplitude (N, 4)
    sigma = blobs_array[:, 2].max()
    max_distance = 2 * sigma * sqrt(blobs_array.shape[1] - 1)
    tree = spatial.cKDTree(blobs_array[:, :-1])
    pairs = np.array(list(tree.query_pairs(max_distance)))

    if len(pairs) == 0:
        return blobs_array
    else:
        # Use in
        keep_idxs = np.ones((blobs_array.shape[0],)).astype(bool)
        for (i, j) in pairs:
            blob1, blob2 = blobs_array[i], blobs_array[j]
            if blob_overlap(blob1, blob2) > overlap:
                # kill the smaller blob if enough overlap
                if blob1[2] > blob2[2]:
                    keep_idxs[j] = False
                else:
                    keep_idxs[i] = False

    return blobs_array[keep_idxs]
    # Note: skimage way overwrote the original blobs array's sigma value: blob1[-1] = 0
    # return np.array([b for b in blobs_array if b[-1] > 0])


def prune_edge_extrema(image, blobs, max_dist_ratio=0.7, positive=True, smooth=True):
    """Finds filters out blobs whose extreme point is far from center

    Searches for local maxima in case there is a nearby larger blob
    at the edge overpowering a real smaller blob which was detected
    This may produce "ghost" blobs, which don't look like a blob should
    when the extreme point is far from the detected center

    Args:
        image (ndarray): image to detect blobs within
        blobs (ndarray): (N, 4) array of blobs from find_blobs
        max_dist_ratio (float): from 0 to 1, the (sigma normalized) distance
            from the extrema to the center pixel.
            Ghost blobs should have extrema near the edges
        positive (bool): search for maxima (uplift). If False, search minima
        smooth (bool): if True, smooth the blob patch with a gaussian filter
            to remove noise when finding extreme value.
            Filter size depends on size of blob radius

    Returns:
        out_blobs: rows from `blobs` which have a local max within
        `max_dist_ratio` of the center

    """
    # TODO: check how it can give > 1 values
    # Removing [  0., 131., 29.47353778, 1.70626337] for dist_to_extreme=1.199

    out_blobs = []
    for b in blobs:
        # if b[0] == 81:
        # ipdb.set_trace()
        if smooth:
            sigma = 3
            # Use 1/4 of the radius (ad hoc value) or 3 (to not wash out tiny blobs)
            # radius = b[2]
            # sigma = np.clip(radius / 4, 3, None)
            # sigma = radius / sqrt(2)
        else:
            sigma = 0
        dist_to_extreme = get_dist_to_extreme(image, b, positive=positive, sigma=sigma)
        if dist_to_extreme < max_dist_ratio:
            out_blobs.append(b)
        # else:
        # print("Removing %s for dist_to_extreme=%s" % (str(b), dist_to_extreme))

    if out_blobs:
        return np.vstack(out_blobs)
    else:
        return np.empty((0, blobs.shape[1]))


def get_dist_to_extreme(image=None, blob=None, positive=True, sigma=0, patch=None):
    """Finds how far from center a blob's extreme point is that caused the result

    Assumes blob[2] is radius, not sigma, to pass to indexes_within_circle

    Returns as a ratio distance/blob_radius (0 to 1)

    image (ndarray): image to detect blobs within
    blobs (ndarray): (N, 4) array of blobs from find_blobs
    positive (bool): search for maxima (uplift). If False, search minima
    sigma (float): optional: used to smooth image before finding the
        extreme value
    """
    if patch is None:
        patch = blob_utils.crop_blob(image, blob, crop_val=None)
    if sigma > 0:
        patch = blob_utils.gaussian_filter_nan(patch, sigma=sigma, mode="nearest")

    # Remove any bias to just look at peak difference
    if positive:
        patch += np.nanmin(patch)
        # Absolute max can fail with nearby stronger blob
        # row, col = np.unravel_index(np.nanargmax(patch), patch.shape)
        local_extreme = peak_local_max(np.nan_to_num(patch))
    else:
        patch -= np.nanmax(patch)
        # row, col = np.unravel_index(np.nanargmin(patch), patch.shape)
        local_extreme = peak_local_max(-1 * np.nan_to_num(patch))

    # print(local_extreme)
    nrows, _ = patch.shape  # circle, will have same rows and cols
    midpoint = nrows // 2

    if not local_extreme.size:
        return 1  # No extreme: output max distance possible

    # print(local_extreme)
    rows = local_extreme[:, 0]
    cols = local_extreme[:, 1]
    dist_arr = np.sqrt((rows - midpoint) ** 2 + (cols - midpoint) ** 2)
    return np.min(dist_arr / blob[2])


def prune_border_blobs(im_shape, blobs, border=2):
    """Takes output of find_blobs, removes those within `border` pixels of edges

    Args:
        im_shape (tuple[int, int]): size of total image where blobs found
        blobs (ndarray): (row, col, sigma, amp) from find_blobs
        border (int): pixels to pad edges and remove blobs from

    Returns:
        mid_blobs: rows of `blobs` that are inside `border` pixels of edge

    Examples:
        >>> blobs = np.array([[1, 1, 10, 10], [3, 5, 10, 10]])
        >>> print(len(prune_border_blobs((20, 20), blobs, 1)))
        2
        >>> print(len(prune_border_blobs((20, 20), blobs, 2)))
        1
        >>> print(len(prune_border_blobs((7, 7), blobs, 2)))
        0
    """
    nrows, ncols = im_shape
    mid_idxs = blobs[:, 0] >= border
    mid_idxs = np.logical_and(mid_idxs, blobs[:, 1] >= border)
    mid_idxs = np.logical_and(mid_idxs, blobs[:, 0] <= (nrows - border - 1))
    mid_idxs = np.logical_and(mid_idxs, blobs[:, 1] <= (ncols - border - 1))

    if mid_idxs.size:
        return np.array(blobs[mid_idxs, :])
    else:
        return np.empty((0, blobs.shape[1]))


def bin_blobs(blobs_array, num_radius_bands):
    """Group the blobs array by its sigma values into bins

    Args:
        blobs_array (ndarray): rows are (row, col, radius)
        num_radius_bands (int): number of distinct groups of radius to divide blobs
            to remove overlaps

    Returns:
        list[ndarray]: blobs binned by their sigma value into `num_radius_bands` groups

    Example:
        >>> import numpy as np; np.set_printoptions(legacy="1.13")
        >>> blobs = np.array([[ 0, 0,  1], [1, 1, 2], [2, 2, 20]])
        >>> print(bin_blobs(blobs, 2)[0])
        [[0 0 1]
         [1 1 2]]
        >>> print(bin_blobs(blobs, 2)[1])
        [[ 2  2 20]]

    """

    def _bin_by_radius(radius_list, num_bins):
        bins = np.histogram_bin_edges(radius_list, bins=num_bins)
        bin_idxs = np.digitize(radius_list, bins[1:], right=True)
        return bin_idxs

    num_radius_bands = int(num_radius_bands)
    bin_idxs = _bin_by_radius(blobs_array[:, 2], num_radius_bands)

    out_list = []
    for cur_idx in range(num_radius_bands):
        out_list.append(blobs_array[bin_idxs == cur_idx])
    return out_list


def peak_local_max(
    image,
    min_distance=1,
    threshold_abs=None,
    threshold_rel=None,
    exclude_border=True,
    num_peaks=np.inf,
    footprint=None,
):
    """Find peaks in an image as coordinate list or boolean mask.

    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).

    If peaks are flat (i.e. multiple adjacent pixels have identical
    intensities), the coordinates of all such pixels are returned.

    If both `threshold_abs` and `threshold_rel` are provided, the maximum
    of the two is chosen as the minimum intensity threshold of peaks.

    Args:
    image : ndarray
        Input image.
    min_distance : int, optional
        Minimum number of pixels separating peaks in a region of `2 *
        min_distance + 1` (i.e. peaks are separated by at least
        `min_distance`).
        To find the maximum number of peaks, use `min_distance=1`.
    threshold_abs : float, optional
        Minimum intensity of peaks. By default, the absolute threshold is
        the minimum intensity of the image.
    threshold_rel : float, optional
        Minimum intensity of peaks, calculated as `max(image) * threshold_rel`.
    exclude_border : int, optional
        If nonzero, `exclude_border` excludes peaks from
        within `exclude_border`-pixels of the border of the image.
    num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` peaks based on highest peak intensity.
    footprint : ndarray of bools, optional
        If provided, `footprint == 1` represents the local region within which
        to search for peaks at every point in `image`.  Overrides
        `min_distance` (also for `exclude_border`).

    Returns:
        ndarray or ndarray of bools: (row, column, ...) coordinates of peaks.

    Notes:
    The peak local maximum function returns the coordinates of local peaks
    (maxima) in an image. A maximum filter is used for finding local maxima.
    This operation dilates the original image. After comparison of the dilated
    and original image, this function returns the coordinates or a mask of the
    peaks where the dilated image equals the original image.

    Examples:
    >>> img1 = np.zeros((7, 7))
    >>> img1[3, 4] = 1
    >>> img1[3, 2] = 1.5
    >>> img1
    array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  1.5,  0. ,  1. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])

    >>> peak_local_max(img1, min_distance=1)
    array([[3, 4],
           [3, 2]])

    >>> peak_local_max(img1, min_distance=2)
    array([[3, 2]])

    >>> img2 = np.zeros((20, 20, 20))
    >>> img2[10, 10, 10] = 1
    >>> peak_local_max(img2, exclude_border=0)
    array([[10, 10, 10]])
    """
    if type(exclude_border) == bool:
        exclude_border = min_distance if exclude_border else 0

    if np.all(image == image.flat[0]):
        return np.empty((0, 2), np.int)

    # Non maximum filter
    # Note: constant here is fine since we're looking for extreme values
    if footprint is not None:
        image_max = maximum_filter(image, footprint=footprint, mode="constant")
    else:
        size = 2 * min_distance + 1
        image_max = maximum_filter(image, size=size, mode="constant")
    mask = image == image_max

    if exclude_border:
        # zero out the image borders
        for i in range(mask.ndim):
            mask = mask.swapaxes(0, i)
            remove = footprint.shape[i] if footprint is not None else 2 * exclude_border
            mask[: remove // 2] = mask[-remove // 2 :] = False
            mask = mask.swapaxes(0, i)

    # find top peak candidates above a threshold
    thresholds = []
    if threshold_abs is None:
        threshold_abs = image.min()
    thresholds.append(threshold_abs)
    if threshold_rel is not None:
        thresholds.append(threshold_rel * image.max())
    if thresholds:
        mask &= image > max(thresholds)

    # Select highest intensities (num_peaks)
    coordinates = _get_high_intensity_peaks(image, mask, num_peaks)

    return coordinates


def _get_high_intensity_peaks(image, mask, num_peaks):
    """
    Return the highest intensity peak coordinates.
    """
    # get coordinates of peaks
    coord = np.nonzero(mask)
    # select num_peaks peaks
    if len(coord[0]) > num_peaks:
        intensities = image[coord]
        idx_maxsort = np.argsort(intensities)
        coord = np.transpose(coord)[idx_maxsort][-num_peaks:]
    else:
        coord = np.column_stack(coord)
    # Higest peak first
    return coord[::-1]


def shape_index(image, sigma=1, mode="nearest", cval=0, eps=1e-16):
    """Compute the shape index.

    The shape index, as defined by Koenderink & van Doorn [1]_, is a
    single valued measure of local curvature, assuming the image as a 3D plane
    with intensities representing heights.

    It is derived from the eigen values of the Hessian, and its
    value ranges from -1 to 1 (and is undefined (=NaN) in *flat* regions),
    with following ranges representing following shapes:

    .. table:: Ranges of the shape index and corresponding shapes.

      Interval (s in ...)  Shape
      ===================  =============
      [  -1, -7/8)         Spherical cup
      [-7/8, -5/8)         Through
      [-5/8, -3/8)         Rut
      [-3/8, -1/8)         Saddle rut
      [-1/8, +1/8)         Saddle
      [+1/8, +3/8)         Saddle ridge
      [+3/8, +5/8)         Ridge
      [+5/8, +7/8)         Dome
      [+7/8,   +1]         Spherical cap
      ===================  =============

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float, optional
        Standard deviation used for the Gaussian kernel, which is used for
        smoothing the input data before Hessian eigen value calculation.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    eps : float, optional
        Padding to the smaller hessian eigenvalue to avoid division by 0

    Returns
    -------
    s : ndarray
        Shape index

    References
    ----------
    .. [1] Koenderink, J. J. & van Doorn, A. J.,
           "Surface shape and curvature scales",
           Image and Vision Computing, 1992, 10, 557-564.
           DOI:10.1016/0262-8856(92)90076-F

    Examples
    --------
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> s = shape_index(square, sigma=0.1)
    >>> s
    array([[ nan,  nan, -0.5,  nan,  nan],
           [ nan, -0. ,  nan, -0. ,  nan],
           [-0.5,  nan, -1. ,  nan, -0.5],
           [ nan, -0. ,  nan, -0. ,  nan],
           [ nan,  nan, -0.5,  nan,  nan]])
    """

    H = hessian_matrix(image, sigma=sigma, mode=mode, cval=cval, order="rc")
    l1, l2 = hessian_matrix_eigvals(H)
    l2_safe = l2 + eps
    num = l2 + l1
    denom = l2 - l1
    arg = num / denom

    out = (2.0 / np.pi) * np.arctan(arg)
    # import ipdb; ipdb.set_trace()
    return out
