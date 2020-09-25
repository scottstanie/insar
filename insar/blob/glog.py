import numpy as np
import multiprocessing as mp
import scipy.ndimage as ndi
from skimage import morphology


def glog(
    im_input,
    alpha=1,
    sigma_range=np.linspace(1.5, 3, np.round((3 - 1.5) / 0.2) + 1),
    dtheta=np.pi / 4,
    tau=0.6,
    eps=0.6,
):
    """Performs generalized Laplacian of Gaussian blob detection.

    Parameters
    ----------
    im_input : array_like
        A hematoxylin intensity image obtained from ColorDeconvolution.
    alpha : double
        A positive scalar used to normalize the gLoG filter responses. Controls
        the blob-center detection and eccentricities of detected blobs. Larger
        values emphasize more eccentric blobs. Default value = 1.
    sigma_range : array_like
        Scale range
    dtheta : double
        Angular increment for rotating gLoG filters. Default value = np.pi / 6.
    tau : double
        Tolerance for counting pixels in determining optimal scale SigmaC
    eps : double
        range to define SigmaX surrounding SigmaC

    Returns
    -------
    Rsum : array_like
        Sum of filter responses at specified scales and orientations
    Maxima: : array_like
        A binary mask highlighting maxima pixels

    Notes
    -----
    Return values are returned as a namedtuple

    References
    ----------
    .. [#] H. Kong, H.C. Akakin, S.E. Sarma, "A Generalized Laplacian of
       Gaussian Filter for Blob Detection and Its Applications," in IEEE
       Transactions on Cybernetics, vol.43,no.6,pp.1719-33, 2013.

    """

    # initialize sigma
    Sigma = np.exp(sigma_range)

    # generate circular LoG scale-space to determine range of SigmaX
    l_g = 0
    H = []
    Bins = []
    Min = np.zeros((len(Sigma), 1))
    Max = np.zeros((len(Sigma), 1))
    for i, s in enumerate(Sigma):
        Response = s ** 2 * ndi.filters.gaussian_laplace(
            im_input, s, output=None, mode="constant", cval=0.0
        )
        Min[i] = Response.min()
        Max[i] = Response.max()
        Bins.append(
            np.arange(
                0.01 * np.floor(Min[i] / 0.01),
                0.01 * np.ceil(Max[i] / 0.01) + 0.01,
                0.01,
            )
        )
        Hist = np.histogram(Response, Bins[i])
        H.append(Hist[0])
        if Max[i] > l_g:
            l_g = Max[i]

    # re-normalized based on global max and local min, count threshold pixels
    Zeta = np.zeros((len(Sigma), 1))
    for i, s in enumerate(Sigma):
        Bins[i] = (Bins[i] - Min[i]) / (l_g - Min[i])
        Zeta[i] = np.sum(H[i][Bins[i][0:-1] > tau])

    # identify best scale SigmaC based on maximum circular response
    Index = np.argmax(Zeta)

    # define range for SigmaX
    XRange = range(max(Index - 2, 0), min(len(sigma_range), Index + 2))
    # import pdb
    # pdb.set_trace()
    SigmaX = np.exp(sigma_range[XRange])

    # define rotation angles
    Thetas = np.linspace(0, np.pi - dtheta, np.round(np.pi / dtheta))

    max_procs = mp.cpu_count()
    pool = mp.Pool(processes=max_procs)
    results = []
    # loop over SigmaX, SigmaY and then angle, summing up filter responses
    Rsum = np.zeros(im_input.shape)
    for i, sx in enumerate(SigmaX):
        YRange = range(0, XRange[i])
        SigmaY = np.exp(sigma_range[YRange])
        for sy in SigmaY:
            for theta in Thetas:
                # Rsum += compute_rsumi(im_input, sx, sy, theta, alpha)
                results.append(
                    pool.apply_async(compute_rsumi, (im_input, sx, sy, theta, alpha))
                )
        # Extra with sx, sx
        results.append(
            pool.apply_async(compute_rsumi, (im_input, sx, sx, theta, alpha))
        )
        # Rsum += compute_rsumi(im_input, sx, sx, theta, alpha)

    for res in results:
        Rsum += res.get()
    # detect local maxima
    Disk = morphology.disk(3 * np.exp(sigma_range[Index]))
    Maxima = ndi.filters.maximum_filter(Rsum, footprint=Disk)
    Maxima = Rsum == Maxima

    return Rsum, Maxima


def compute_rsumi(im, sx, sy, theta, alpha, verbose=True):
    if verbose:
        print(sx, sy, theta)
    kernel = glogkernel(sx, sy, theta)
    kernel *= (1 + np.log(sx) ** alpha) * (1 + np.log(sy) ** alpha)
    return ndi.convolve(im, kernel, mode="constant", cval=0.0)


def glogkernel(sigma_x, sigma_y, theta):

    N = np.ceil(2 * 3 * sigma_x)
    X, Y = np.meshgrid(
        np.linspace(0, N, N + 1) - N / 2, np.linspace(0, N, N + 1) - N / 2
    )
    a = np.cos(theta) ** 2 / (2 * sigma_x ** 2) + np.sin(theta) ** 2 / (
        2 * sigma_y ** 2
    )
    b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + np.sin(2 * theta) / (4 * sigma_y ** 2)
    c = np.sin(theta) ** 2 / (2 * sigma_x ** 2) + np.cos(theta) ** 2 / (
        2 * sigma_y ** 2
    )
    D2Gxx = ((2 * a * X + 2 * b * Y) ** 2 - 2 * a) * np.exp(
        -(a * X ** 2 + 2 * b * X * Y + c * Y ** 2)
    )
    D2Gyy = ((2 * b * X + 2 * c * Y) ** 2 - 2 * c) * np.exp(
        -(a * X ** 2 + 2 * b * X * Y + c * Y ** 2)
    )
    Gaussian = np.exp(-(a * X ** 2 + 2 * b * X * Y + c * Y ** 2))
    kernel = (D2Gxx + D2Gyy) / np.sum(Gaussian.flatten())
    return kernel
