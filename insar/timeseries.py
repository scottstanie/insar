"""Functions for performing time series analysis of unwrapped interferograms

files in the igrams folder:
    geolist, intlist, sbas_list
scott@lidar igrams]$ head geolist
../S1A_IW_SLC__1SDV_20180420T043026_20180420T043054_021546_025211_81BE.SAFE.geo
../S1A_IW_SLC__1SDV_20180502T043026_20180502T043054_021721_025793_5C18.SAFE.geo
[scott@lidar igrams]$ head sbas_list
../S1A_IW_SLC__1SDV_20180420T043026_20180420T043054_021546_025211_81BE.SAFE.geo ../S1A_IW_SLC__1SDV_20180502T043026_20180502T043054_021721_025793_5C18.SAFE.geo 12.0   -16.733327776024169
[scott@lidar igrams]$ head intlist
20180420_20180502.int

"""

import os
import datetime
import numpy as np
from insar.parsers import Sentinel
from insar import sario

SENTINEL_WAVELENGTH = 5.5465763  # cm
PHASE_TO_CM = SENTINEL_WAVELENGTH / (-4 * np.pi)


def read_geolist(filepath="./geolist"):
    """Reads in the list of .geo files used, in time order

    Args:
        filepath (str): path to the intlist file

    Returns:
        list[date]: the parse dates of each .geo used, in date order

    """
    with open(filepath) as f:
        geolist = [os.path.split(geoname)[1] for geoname in f.read().splitlines()]
    return sorted([Sentinel(geo).start_time().date() for geo in geolist])


def read_intlist(filepath="./intlist", parse=True):
    """Reads the list of igrams to return dates of images as a tuple

    Args:
        filepath (str): path to the intlist file
        parse (bool): output the intlist as parsed datetime tuples

    Returns:
        tuple(date, date) of master, slave dates for all igrams (if parse=True)
            if parse=False: returns list[str], filenames of the igrams

    """

    def _parse(datestr):
        return datetime.datetime.strptime(datestr, "%Y%m%d").date()

    with open(filepath) as f:
        intlist = f.read().splitlines()

    if parse:
        intlist = [intname.strip('.int').split('_') for intname in intlist]
        return [(_parse(master), _parse(slave)) for master, slave in intlist]
    else:
        dirname = os.path.dirname(filepath)
        return [os.path.join(dirname, igram) for igram in intlist]


def build_A_matrix(geolist, intlist):
    """Takes the list of igram dates and builds the SBAS A matrix

    Args:
        geolist (list[date]): datetimes of the .geo acquisitions
        intlist (list[tuple(date, date)])

    Returns:
        np.array 2D: the incident-like matrix from the SBAS paper: A*phi = dphi
            Each row corresponds to an igram, each column to a .geo
            value will be -1 on the early (slave) igrams, +1 on later (master)
    """
    # We take the first .geo to be time 0, leave out of matrix
    # Match on date (not time) to find indices
    geolist = geolist[1:]
    M = len(intlist)  # Number of igrams, number of rows
    N = len(geolist)
    A = np.zeros((M, N))
    for j in range(M):
        early_igram, late_igram = intlist[j]

        try:
            idx_early = geolist.index(early_igram)
            A[j, idx_early] = -1
        except ValueError:  # The first SLC will not be in the matrix
            pass

        idx_late = geolist.index(late_igram)
        A[j, idx_late] = 1

    return A


def find_time_diffs(geolist):
    """Finds the number of days between successive .geo files

    Output length is a np.array of length len(geolist) - 1"""
    return np.array([difference.days for difference in np.diff(geolist)])


def build_B_matrix(geolist, intlist):
    """Takes the list of igram dates and builds the SBAS B (velocity coeff) matrix

    Args:
        geolist (list[date]): dates of the .geo acquisitions
        intlist (list[tuple(date, date)])

    Returns:
        np.array 2D: the velocity coefficient matrix from the SBAS paper: Bv = dphi
            Each row corresponds to an igram, each column to a .geo
            value will be t_k+1 - t_k for columns after the -1 in A,
            up to and including the +1 entry
    """
    timediffs = find_time_diffs(geolist)

    A = build_A_matrix(geolist, intlist)
    B = np.zeros_like(A)

    for j, row in enumerate(A):
        # if no -1 entry, start at index 0. Otherwise, add 1 so exclude the -1 index
        start_idx = list(row).index(-1) + 1 if (-1 in row) else 0
        # End index is inclusive of the +1
        end_idx = np.where(row == 1)[0][0] + 1

        # Now only fill in the time diffs in the range from the early igram index
        # to the later igram index
        B[j][start_idx:end_idx] = timediffs[start_idx:end_idx]

    return B


def invert_sbas(geolist, intlist, dphi_array):
    B = build_B_matrix(geolist, intlist)
    # Velocity will be result of the inversion
    velocity_array, _, rank_B, sing_vals_B = np.linalg.lstsq(B, dphi_array, rcond=None)
    # velocity array entries: v_j = (phi_j - phi_j-1)/(t_j - t_j-1)
    velocity_array = np.squeeze(velocity_array)  # Remove singleton dim

    # Now integrate to get back to phases
    timediffs = find_time_diffs(geolist)
    phi_diffs = timediffs * velocity_array
    return velocity_array, np.cumsum(phi_diffs)


def read_unw_list(intlist, row, column):
    # row 283, col 493 looks like a good test
    igrams = read_intlist(intlist, parse=False)
    num_ints = len(igrams)
    pixel_phase_arr = np.zeros((num_ints, 1))
    for idx, igram_file in enumerate(igrams):
        unw_file = igram_file.replace('.int', '.unw')
        pixel_phase = sario.load_file(unw_file)[row, column]
        pixel_phase_arr[idx] = pixel_phase
    return pixel_phase_arr
