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
from insar.parsers import Sentinel


def read_geolist(filepath="./geolist"):
    """Reads in the list of .geo files used, in time order
 
    Args:
        filepath (str): path to the intlist file

    Returns:
        list[datetime]: the parse dates of each .geo used, in date order

    """
    with open(filepath) as f:
        geolist = [os.path.split(geoname)[1] for geoname in f.readlines()]
    return [Sentinel(geo).start_time() for geo in geolist]


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
        intlist = [intname.strip('.int').split('_') for intname in f.readlines()]
    return [(_parse(master), _parse(slave)) for master, slave in intlist]
