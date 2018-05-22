"""
Utilities for parsing file names of SAR products for relevant info.

"""
import re
from datetime import datetime


class Sentinel:
    """
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
    R: Resolution class: F, H, M, or _ (N/A)
    L: Processing Level: 0, 1, 2
    F: Product class: S (standard), A (annotation, only used internally)
    PP: Polarization: SH (single HH), SV (single VV), DH (dual HH+HV), DV (dual VV+VH)
    Start date + time (date/time separated by T)
    Stop date + time
    OOOOOO: absolute orbit number: 000001-999999
    DDDDDD: mission data-take identifier: 000001-FFFFFF.
    CCCC: product unique identifier: hexadecimal string from CRC-16 the manifest file using CRC-CCITT.

    Once unzipped, the folder extension is always "SAFE".

    Attributes:
        filename (str) name of the sentinel data product
    """
    FILE_REGEX = r'([\w\d]{3})_([\w\d]{2})_([\w_]{3})([FHM_])_(\d)([SA])([SDHV]{2})_([T\d]{15})_([T\d]{15})_([\d]{6})_([\d\w]{6})_([\d\w]{4})'

    def __init__(self, filename):
        self.filename = filename

    def full_parse(self):
        """Returns all parts of the sentinel data contained in filename

        Args:
            self

        Returns:
            tuple: parsed file data. Entry order will match `field_meanings()`

        Raises:
            ValueError: if filename string is invalid
        """
        try:
            return re.match(self.FILE_REGEX, self.filename).groups()
        except AttributeError:  # Nonetype has no attribute 'groups'
            raise ValueError('Invalid sentinel product filename: {}'.format(self.filename))

    @staticmethod
    def field_meanings():
        return ('Mission', 'Beam', 'Product type', 'Resolution class', 'Product level',
                'Product class', 'Polarization', 'Start datetime', 'Stop datetime', 'Orbit number',
                'data-take identified', 'product unique id')

    def start_stop_time(self):
        """Returns start datetime and stop datetime from a sentinel file name

        Args:
            sentinel_filename (str): filename of a sentinel 1 product

        Returns:
            (datetime, datetime): start datetime, stop datetime


        """
        time_format = '%Y%m%dT%H%M%S'
        start_time_str, stop_time_str = self.full_parse()[7:9]
        start_time = datetime.strptime(start_time_str, time_format)
        stop_time = datetime.strptime(start_time_str, time_format)

        return start_time, stop_time
