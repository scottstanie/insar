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
    FILE_REGEX = r'(S1A|S1B)_([\w\d]{2})_([\w_]{3})([FHM_])_(\d)([SA])([SDHV]{2})_([T\d]{15})_([T\d]{15})_([\d]{6})_([\d\w]{6})_([\d\w]{4})'
    TIME_FMT = '%Y%m%dT%H%M%S'

    def __init__(self, filename):
        self.filename = filename
        self.full_parse()  # Run a parse to check validity of filename

    def full_parse(self):
        """Returns all parts of the sentinel data contained in filename

        Args:
            self

        Returns:
            tuple: parsed file data. Entry order will match `field_meanings()`

        Raises:
            ValueError: if filename string is invalid
        """
        match = re.search(self.FILE_REGEX, self.filename)
        if not match:
            raise ValueError('Invalid sentinel product filename: {}'.format(self.filename))
        else:
            return match.groups()

    @staticmethod
    def field_meanings():
        """List the fields returned by full_parse()"""
        return ('Mission', 'Beam', 'Product type', 'Resolution class', 'Product level',
                'Product class', 'Polarization', 'Start datetime', 'Stop datetime', 'Orbit number',
                'data-take identified', 'product unique id')

    def start_time(self):
        """Returns start datetime and stop datetime from a sentinel file name

        Args:
            sentinel_filename (str): filename of a sentinel 1 product

        Returns:
            datetime: start datetime of mission


        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.start_time())
            2018-04-08 04:30:25
        """
        start_time_str = self.full_parse()[7]
        return datetime.strptime(start_time_str, self.TIME_FMT)

    def stop_time(self):
        """Returns stop datetime from a sentinel file name

        Args:
            sentinel_filename (str): filename of a sentinel 1 product

        Returns:
            datetime: stop datetime

        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.stop_time())
            2018-04-08 04:30:53
        """
        stop_time_str = self.full_parse()[8]
        return datetime.strptime(stop_time_str, self.TIME_FMT)

    def polarization(self):
        """Returns type of polarization of product

        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.polarization())
            DV
        """
        polarization_index = 6
        return self.full_parse()[polarization_index]

    def mission(self):
        """Returns satellite/mission of product (S1A/S1B)

        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.mission())
            S1A
        """
        mission_index = 0
        return self.full_parse()[mission_index]

    def absolute_orbit(self):
        """Absolute orbit of data, included in file name

        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.absolute_orbit())
            21371
        """
        abs_orbit_index = 9
        return int(self.full_parse()[abs_orbit_index])

    def relative_orbit(self):
        """Relative orbit number/ path

        Formulas for relative orbit from absolute come from:
        https://forum.step.esa.int/t/sentinel-1-relative-orbit-from-filename/7042

        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.relative_orbit())
            124
            >>> s = Sentinel('S1B_WV_OCN__2SSV_20180522T161319_20180522T164846_011036_014389_67D8')
            >>> print(s.relative_orbit())
            160
        """
        if self.mission() == 'S1A':
            return ((self.absolute_orbit() - 73) % 175) + 1
        elif self.mission() == 'S1B':
            return ((self.absolute_orbit() - 27) % 175) + 1

    def path(self):
        """Alias for relative orbit number"""
        return self.relative_orbit()
