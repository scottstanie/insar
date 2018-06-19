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


def read_geolist(filepath="./geolist"):
    """Reads in the list of .geo files used, in time order

    Args:
        filepath (str): path to the intlist file

    Returns:
        list[datetime]: the parse dates of each .geo used, in date order

    """
    with open(filepath) as f:
        geolist = [os.path.split(geoname)[1] for geoname in f.read().splitlines()]
    return sorted([Sentinel(geo).start_time() for geo in geolist])


def read_intlist(filepath="./intlist"):
    """Reads the list of igrams to return dates of images as a tuple

    Args:
        filepath (str): path to the intlist file

    Returns:
        tuple(datetime, datetime) of master, slave dates for all igrams

    """

    def _parse(datestr):
        return datetime.datetime.strptime(datestr, "%Y%m%d")

    with open(filepath) as f:
        intlist = [intname.strip('.int').split('_') for intname in f.read().splitlines()]

    return [(_parse(master), _parse(slave)) for master, slave in intlist]


def build_A_matrix(geolist, intlist):
    """Takes the list of igram dates and builds the SBAS A matrix

    Args:
        geolist (list[datetime]): datetimes of the .geo acquisitions
        intlist (list[tuple(datetime, datetime)])

    Returns:
        np.array 2D: the incident-like matrix from the SBAS paper: A*phi = dphi
            Each row corresponds to an igram, each column to a .geo
            value will be -1 on the early (slave) igrams, +1 on later (master)
    """
    # We take the first .geo to be time 0, leave out of matrix, and only
    # Only match on date (not time) to find indices
    geolist = [g.date() for g in geolist[1:]]
    M = len(intlist)  # Number of igrams, number of rows
    N = len(geolist)
    A = np.zeros((M, N))
    for j in range(M):
        early_igram, late_igram = intlist[j]

        try:
            idx_early = geolist.index(early_igram.date())
            A[j, idx_early] = -1
        except ValueError:  # The first SLC will not be in the matrix
            pass

        idx_late = geolist.index(late_igram.date())
        A[j, idx_late] = 1

    return A


def build_B_matrix(geolist, intlist):
    """Takes the list of igram dates and builds the SBAS B (velocity coeff) matrix

    Args:
        geolist (list[datetime]): datetimes of the .geo acquisitions
        intlist (list[tuple(datetime, datetime)])

    Returns:
        np.array 2D: the velocity coefficient matrix from the SBAS paper: Bv = dphi
            Each row corresponds to an igram, each column to a .geo
            value will be -1 on the early (slave) igrams, +1 on later (master)
    """
