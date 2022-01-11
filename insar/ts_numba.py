import numpy as np
from math import ceil
import numba
from numba import njit, cuda

from apertools.log import get_log

log = get_log()


@njit
# @cuda.jit
def build_A_matrix(sar_dates, ifg_dates):
    """Takes the list of igram dates and builds the SBAS A matrix

    Args:
        sar_dates (ndarray[int]): num2date(date) of the acquisitions
        ifg_dates (ndarray[tuple(int, int)])

    Returns:
        np.array 2D: the incident-like matrix from the SBAS paper: A*phi = dphi
            Each row corresponds to an igram, each column to a SAR date
            value will be -1 on the early (reference) igrams, +1 on later (secondary)
    """
    # We take the first .geo to be time 0, leave out of matrix
    # Match on date (not time) to find indices
    M = len(ifg_dates)  # Number of igrams, number of rows
    N = len(sar_dates) - 1
    A = np.zeros((M, N))
    for j in range(M):
        early_igram, late_igram = ifg_dates[j]

        for idx in range(N):
            sd = sar_dates[idx + 1]
            if early_igram == sd:
                A[j, idx] = -1
            elif late_igram == sd:
                A[j, idx] = 1

    return A


@njit
def build_B_matrix(sar_dates, ifg_dates, model=None):
    """Takes the list of igram dates and builds the SBAS B (velocity coeff) matrix

    Args:
        sar_dates (ndarray[int]): num2date(date) of the acquisitions
        ifg_dates (ndarray[tuple(int, int)])
        model (str): If 'linear', creates the M x 1 matrix for linear velo model

    Returns:
        np.array: 2D array of the velocity coefficient matrix from the SBAS paper:
                Bv = dphi
            Each row corresponds to an igram, each column to a SAR date
            value will be t_k+1 - t_k for columns after the -1 in A,
            up to and including the +1 entry
    """
    timediffs = np.diff(sar_dates)

    A = build_A_matrix(sar_dates, ifg_dates)
    B = np.zeros_like(A)

    for j, row in enumerate(A):
        # if no -1 entry, start at index 0. Otherwise, add 1 to exclude the -1 index
        start_idx = 0
        for idx, item in enumerate(row):
            if item == -1:
                start_idx = idx + 1
            elif item == 1:
                end_idx = idx + 1

        # Now only fill in the time diffs in the range from the early igram index
        # to the later igram index
        B[j][start_idx:end_idx] = timediffs[start_idx:end_idx]

    if model == "linear":
        BB = B.sum(axis=1)
        return BB.reshape((-1, 1))
    else:
        return B


@njit
def integrate_velocities(velocity_array, timediffs):
    """Takes SBAS velocity output and finds phases

    Args:
        velocity_array (ndarray): output of run_sbas, velocities at
            each point in time
        timediffs (np.array): dtype=int, days between each SAR acquisitions
            length will be 1 less than num SAR acquisitions

    Returns:
        ndarray: integrated phase array

    """
    nrows, ncols = velocity_array.shape
    # Add 0 as first entry of phase array to match slclist length on each col
    phi_arr = np.zeros((nrows + 1, ncols))
    # multiply each column of vel array: each col is a separate solution
    for j in range(velocity_array.shape[1]):
        # the final phase results are the cumulative sum of delta phis
        phi_arr[1:, j] = np.cumsum(velocity_array[:, j] * timediffs)

    return phi_arr


# TODO: jit...
# @njit
def subset_A(A, full_slclist, full_ifglist, slclist, ifglist, valid_ifg):
    # TODO: valid_ifg in here...
    valid_slc = np.searchsorted(full_slclist[1:], slclist)
    return A[valid_ifg][:, valid_slc][:, 1:]


# def run_lowess_gpu(da, frac=0.7, n_iter=2):
#     from matplotlib.dates import date2num

#     x = date2num(da["date"].values)

#     stack = da.data
#     rows, cols = stack.shape[-2:]
#     # TODO: launch kernel
#     threadsperblock = (16, 16)
#     blockspergrid_x = ceil(rows / threadsperblock[0])
#     blockspergrid_y = ceil(cols / threadsperblock[1])
#     blockspergrid = (blockspergrid_x, blockspergrid_y)
#     log.info("Lowess smooothing on GPU.")
#     log.info(
#         "(blocks per grid, threads per block) = ((%s, %s), (%s, %s))",
#         *blockspergrid,
#         *threadsperblock
#     )

#     n = len(x)
#     h = np.zeros(n, dtype=np.float64)
#     w = np.zeros((n, n), dtype=np.float64)
#     yest = np.zeros(n)
#     delta = np.ones(n)
#     b = np.zeros(2, dtype=np.float64)
#     A = np.zeros((2, 2), dtype=np.float64)
#     weights = np.zeros(2, dtype=np.float64)

#     out = np.zeros(stack.shape, dtype=stack.dtype)
#     lowess[blockspergrid, threadsperblock](
#         stack, x, frac, n_iter, out, n, h, w, yest, delta, b, A, weights
#     )
#     return out


# import math
# import cupy
# from cupy import linalg
# # import cupy


# @cuda.jit
# def lowess(stack, x, frac, n_iter, out, n, h, w, yest, delta, b, A, weights):
#     i, j = cuda.grid(2)
#     if not (0 <= i < stack.shape[1] and 0 <= j < stack.shape[2]):
#         return
#     y = stack[:, i, j]

#     # Check whether any nans exist, or if the entire column is zero
#     all0 = True
#     for idx in range(len(y)):
#         if math.isnan(y[idx]):
#             return
#         if y[idx] != 0:
#             all0 = False
#     if all0:
#         return

#     n = len(x)
#     r = int(math.ceil(frac * n))

#     # h = np.array([np.sort(np.abs(x - x[i]))[r] for i in range(n)])
#     for i in range(n):
#         h[i] = cupy.sort(np.abs(x - x[i]))[r]

#     w = np.minimum(
#         1.0, np.maximum(np.abs((x.reshape((-1, 1)) - x.reshape((1, -1))) / h), 0.0)
#     )
#     w = (1 - w ** 3) ** 3
#     yest = 0.0
#     delta = 1.0

#     for _ in range(n_iter):
#         for i in range(n):
#             weights = delta * w[:, i]
#             # b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
#             b[0] = np.sum(weights * y)
#             b[1] = np.sum(weights * y * x)
#             A[0, 0] = np.sum(weights)
#             A[0, 1] = np.sum(weights * x)
#             A[1, 0] = A[0, 1]
#             A[1, 1] = np.sum(weights * x * x)

#             beta = linalg.lstsq(A, b)[0]
#             yest[i] = beta[0] + beta[1] * x[i]

#         residuals = y - yest
#         s = np.median(np.abs(residuals))
#         # delta = np.clip(residuals / (6.0 * s), -1.0, 1.0)
#         delta = np.minimum(1.0, np.maximum(residuals / (6.0 * s), -1.0))
#         delta = (1 - delta ** 2) ** 2

#     out[:, i, j] = yest

# from apertools import lowess

# # def _lowess(y, x, f=2.0 / 3.0, n_iter=3):  # pragma: no cover
# from numba import guvectorize


# @guvectorize(
#     "(float64[:], float64[:], float64, int64, float64[:])",
#     "(n),(n),(),()->(n)",
#     nopython=True,
#     # parallel=True,
# )
# def _run_pixel(y, x, frac, it, out):
#     if not (np.any(np.isnan(y)) or np.all(y == 0)):
#         out[:] = lowess._lowess(y, x, frac, it)
