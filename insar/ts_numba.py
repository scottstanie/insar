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
def integrate_velocities(velocity_cols, timediffs):
    """Takes SBAS velocity output and finds phases

    Args:
        velocity_cols (ndarray): output of run_sbas, velocities at
            each point in time, where each column is one pixel
        timediffs (np.array): dtype=int, days between each SAR acquisitions
            length will be 1 less than num SAR acquisitions

    Returns:
        ndarray: integrated phase array

    """
    nrows, ncols = velocity_cols.shape
    # Add 0 as first entry of phase array to match slclist length on each col
    phi_arr = np.zeros((nrows + 1, ncols))
    # multiply each column of vel array: each col is a separate solution
    for j in range(velocity_cols.shape[1]):
        # the final phase results are the cumulative sum of delta phis
        phi_arr[1:, j] = np.cumsum(velocity_cols[:, j] * timediffs)

    return phi_arr


# TODO: jit...
# @njit
def subset_A(A, full_slclist, full_ifglist, slclist, ifglist, valid_ifg):
    # TODO: valid_ifg in here...
    valid_slc = np.searchsorted(full_slclist[1:], slclist)
    return A[valid_ifg][:, valid_slc][:, 1:]
