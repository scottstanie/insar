"""
Utilities for parsing file names of SAR products for relevant info.

Sentinel 1 reference:
https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/naming-conventions
or https://sentinel.esa.int/documents/247904/349449/Sentinel-1_Product_Specification

Example:
    S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70.zip

File name format:
    MMM_BB_TTTR_LFPP_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_OOOOOO_DDDDDD_CCCC.EEEE

MMM: mission/satellite S1A or S1B
BB: Mode/beam identifier. The S1-S6 beams apply to SM products
    IW, EW and WV identifiers appply to products from the respective modes.
TTT: Product Type: RAW, SLC, GRD, OCN
R: Resolution class: F,M, or _ (N/A)
L: Processing Level: 0, 1, 2
F: Product class: S (standard), A (annotation, only used internally)
PP: Polarization: SH (single HH), SV (single VV), DH (dual HH+HV), DV (dual VV+VH)
Start date + time (date/time separated by T)
Stop date + time
OOOOOO: absolute orbit number: 000001-999999
DDDDDD: mission data-take identifier: 000001-FFFFFF.
CCCC: product unique identifier: hexadecimal string from CRC-16 the manifest file using CRC-CCITT.

Once unzipped, the folder extension is always "SAFE".
"""
import re
from datetime import datetime

SENTINEL_PARTS = r'([\w\d]{3})_([\w\d]{2})_([\w_]{4})_(\d[SA])([SDHV]{2})_([T\d]{15})_([T\d]{15})_([\d]{6})_([\d\w]{6})_([\d\w]{4})'


def parse_start_stop(sentinel_filename):
    """Returns start datetime and stop datetime from a sentinel file name

    Args:
        sentinel_filename (str): filename of a sentinel 1 product

    Returns:
        (datetime, datetime): start date and time, stop datetime

    Raises:
        ValueError: if sentinel_filename string is invalid

    """

    fileparts = sentinel_filename.split('_')[4:6]  #4th, 5th parts are dates
    start_time = datetime.strptime(fileparts[0])
    start_time = parse(fileparts[1])

    return start_time, stop_time
