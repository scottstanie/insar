"""
Utilities for parsing file names of SAR products for relevant info.

"""
import re
from datetime import datetime
from insar import sario


class Base(object):
    """Base parser to illustrate expected interface/ minimum data available
    """
    FILE_REGEX = None
    TIME_FMT = None
    _FIELD_MEANINGS = None

    def __init__(self, filename):
        self.filename = filename
        self.full_parse()  # Run a parse to check validity of filename

    def __str__(self):
        return "{} product: {}".format(self.__class__.__name__, self.filename)

    def __repr__(self):
        return str(self)

    def full_parse(self):
        """Returns all parts of the data contained in filename

        Returns:
            tuple: parsed file data. Entry order will match `field_meanings`

        Raises:
            ValueError: if filename string is invalid
        """
        if not self.FILE_REGEX:
            raise NotImplementedError("Must define class FILE_REGEX to parse")

        match = re.search(self.FILE_REGEX, self.filename)
        if not match:
            raise ValueError('Invalid {} filename: {}'.format(self.__class__.__name__,
                                                              self.filename))
        else:
            return match.groups()

    @property
    def field_meanings(self):
        """List the fields returned by full_parse()"""
        return self._FIELD_MEANINGS

    def _get_field(self, fieldname):
        """Pick a specific field based on its name"""
        idx = self.field_meanings.index(fieldname)
        return self.full_parse()[idx]


class Sentinel(Base):
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
    FILE_REGEX = r'(S1A|S1B)_([\w\d]{2})_([\w_]{3})([FHM_])_(\d)([SA])([SDHV]{2})_([T\d]{15})_([T\d]{15})_(\d{6})_([\d\w]{6})_([\d\w]{4})'
    TIME_FMT = '%Y%m%dT%H%M%S'
    _FIELD_MEANINGS = ('mission', 'beam', 'product type', 'resolution class', 'product level',
                       'product class', 'polarization', 'start datetime', 'stop datetime',
                       'orbit number', 'data-take identified', 'product unique id')

    def __str__(self):
        return "{} {}, path {} from {}".format(self.__class__.__name__, self.mission, self.path,
                                               self.date)

    @property
    def start_time(self):
        """Returns start datetime from a sentinel file name

        Args:
            filename (str): filename of a sentinel 1 product

        Returns:
            datetime: start datetime of mission


        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.start_time)
            2018-04-08 04:30:25
        """
        start_time_str = self._get_field('start datetime')
        return datetime.strptime(start_time_str, self.TIME_FMT)

    @property
    def stop_time(self):
        """Returns stop datetime from a sentinel file name

        Args:
            filename (str): filename of a sentinel 1 product

        Returns:
            datetime: stop datetime

        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.stop_time)
            2018-04-08 04:30:53
        """
        stop_time_str = self._get_field('stop datetime')
        return datetime.strptime(stop_time_str, self.TIME_FMT)

    @property
    def polarization(self):
        """Returns type of polarization of product

        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.polarization)
            DV
        """
        return self._get_field('polarization')

    @property
    def mission(self):
        """Returns satellite/mission of product (S1A/S1B)

        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.mission)
            S1A
        """
        return self._get_field('mission')

    @property
    def absolute_orbit(self):
        """Absolute orbit of data, included in file name

        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.absolute_orbit)
            21371
        """
        return int(self._get_field('orbit number'))

    @property
    def relative_orbit(self):
        """Relative orbit number/ path

        Formulas for relative orbit from absolute come from:
        https://forum.step.esa.int/t/sentinel-1-relative-orbit-from-filename/7042

        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.relative_orbit)
            124
            >>> s = Sentinel('S1B_WV_OCN__2SSV_20180522T161319_20180522T164846_011036_014389_67D8')
            >>> print(s.relative_orbit)
            160
        """
        if self.mission == 'S1A':
            return ((self.absolute_orbit - 73) % 175) + 1
        elif self.mission == 'S1B':
            return ((self.absolute_orbit - 27) % 175) + 1

    @property
    def path(self):
        """Alias for relative orbit number"""
        return self.relative_orbit


class Uavsar(Base):
    """Uavsar reference for Polsar:
    https://uavsar.jpl.nasa.gov/science/documents/polsar-format.html

    RPI/ InSAR format reference:
    https://uavsar.jpl.nasa.gov/science/documents/rpi-format-browse.html

    Naming example:
    Dthvly_34501_08038_006_080731_L090HH_XX_01.slc

    Dthvly is the site name, 345 degrees is the heading of UAVSAR in flight,
    with a counter of 01, the flight was the thirty-eighth flight by UAVSAR in
    2008,this data take was the sixth data take during the flight, the data was
    acquired on July 31, 2008 (UTC), the frequency band was L-band, pointing at
    perpendicular to the flight heading (90 degrees counterclockwise), this
    file contains the HH data, this is the first interation of processing,
    cross talk calibration has not been applied, and the data type is SLC.

    For downsampled products (3x3 and 5x5), there is an optional extension
    of _ML3X3 and _ML5X5 tacked onto the end

    Examples:
        >>> fname = 'Dthvly_34501_08038_006_080731_L090HH_XX_01.slc'
        >>> parser = Uavsar(fname)

    """
    FILE_REGEX = r'([\w\d]{6})_(\d{3})(\w+)_(\d{2})(\d{3})_(\d{3})_(\d{6})_(\w)(\d{3})(\w{0,4})_(XX|CX)_(\w{2})(_ML\dX\d)?'
    TIME_FMT = '%y%m%d'
    _FIELD_MEANINGS = (
        'target site',
        'heading',
        'counter',
        'flight year',
        'flight number',
        'flight line',
        'date',
        'frequency band',
        'steering angle',
        'polarization',
        'cross talk',
        'version number',
        'downsampling',
    )

    def __str__(self):
        return "{} {} from {}".format(self.__class__.__name__, self.polarization, self.date)

    @property
    def date(self):
        """Returns date of flight from file name
        Args:
            filename (str): filename of a product from self

        Returns:
            datetime.date: date mission

        Examples:
            >>> parser = Uavsar('Dthvly_34501_08038_006_080731_L090HH_XX_01.slc')
            >>> parser.date
            datetime.date(2008, 7, 31)
        """
        date_str = self._get_field('date')
        return datetime.strptime(date_str, self.TIME_FMT).date()

    @property
    def polarization(self):
        """Polarization of the product, if any

        May be between 2 and 4 chars, though .zip files don't have one

        Examples:
            >>> Uavsar('brazos_14938_17087_004_170831_L090HH_CX_01.slc').polarization
            'HH'
            >>> Uavsar('brazos_14938_17087_004_170831_L090HHHV_CX_01.mlc').polarization
            'HHHV'
            >>> Uavsar('brazos_14938_17087_004_170831_L090_CX_01_grd.zip').polarization
            ''
        """
        return self._get_field('polarization')

    @property
    def downsampling(self):
        """Amount of downsampling of product, if any

        Examples:
        >>> print(Uavsar('brazos_14938_17087_004_170831_L090_CX_01_grd.zip').downsampling)
        None
        >>> Uavsar('brazos_14938_17087_004_170831_L090_CX_01_ML5X5_grd.zip').downsampling
        '5X5'
        >>> Uavsar('brazos_14938_17087_004_170831_L090HHHV_CX_01_ML3X3.grd').downsampling
        '3X3'
        """
        sample_str = self._get_field('downsampling')
        return sample_str.replace('_ML', '') if sample_str else None

    def _make_ann_filename(self):
        """Take the name of a data file and return corresponding .ann name

        Examples:
        >>> u = Uavsar('brazos_14938_17087_004_170831_L090HHHV_CX_01_ML3X3.grd')
        >>> print(u._make_ann_filename('brazos.cor'))
        brazos.ann
        >>> print(u._make_ann_filename('brazos.1.int'))
        brazos.ann
        >>> print(u._make_ann_filename('brazos_090HHHV_CX_01.mlc'))
        brazos_090_CX_01.ann
        >>> print(u._make_ann_filename('brazos_090HHVV_CX_01.mlc'))
        brazos_090_CX_01.ann
        >>> print(u._make_ann_filename('brazos_090HHVV_CX_01.grd'))
        brazos_090_CX_01.ann
        >>> print(u._make_ann_filename('brazos_090HHVV_CX_01_ML5X5.grd'))
        brazos_090_CX_01_ML5X5.ann
        """

        # The .mlc and .grd files have polarization added to filename, .ann files don't
        shortname = filename
        for p in sario.POLARIZATIONS:
            shortname = shortname.replace(p, '')
        # If this is a block we split up and names .1.int, remove that since
        # all have the same .ann file

        # TODO: figure out where to get this list from
        ext = get_file_ext(filename)
        shortname = re.sub('\.\d' + ext, ext, shortname)

        return shortname.replace(ext, '.ann')

    def parse_ann_file(self, verbose=False):
        """Returns the requested data from the UAVSAR annotation in ann_filename

        Args:
            ann_data (dict): key-values of requested data from .ann file
            verbose (bool): print extra logging into about file loading

        Returns:
            dict: the annotation file parsed into a dict. If no annotation file
                can be found, None is returned
        """

        def _parse_line(line):
            wordlist = line.split()
            # Pick the entry after the equal sign when splitting the line
            return wordlist[wordlist.index('=') + 1]

        def _parse_int(line):
            return int(_parse_line(line))

        def _parse_float(line):
            return float(_parse_line(line))

        def _make_line_regex(ext, field):
            return r'{}.{}'.format(line_keywords.get(ext), field)

        ext = ext or get_file_ext(self.filename)  # Use what's passed by default
        ann_filename = _make_ann_filename(self.filename)
        if verbose:
            logger.info("Trying to load ann_data from %s", ann_filename)
        if not os.path.exists(ann_filename):
            if verbose:
                logger.info("No file found: returning None")
            return None

        # Taken from a .ann file: (need to check if this is always true?)
        # SLC Data Units = linear amplitude
        # MLC Data Units = linear power
        # GRD Data Units = linear power
        ann_data = {}
        line_keywords = {
            # ext: line start term
            '.slc': 'slc_mag',
            '.mlc': 'mlc_mag',
            '.int': 'slt',
            '.cor': 'slt',
            '.amp': 'slt',
            '.grd': 'grd_mag'
        }
        row_key = line_keywords.get(ext) + '.set_rows'
        col_key = line_keywords.get(ext) + '.set_cols'

        # Peg position the nadir position of aircraft at middle of datatake
        with open(ann_filename, 'r') as f:
            for line in f.readlines():
                # TODO: disambiguate which ones to use, and when
                if line.startswith(row_key):
                    ann_data['rows'] = _parse_int(line)
                elif line.startswith(col_key):
                    ann_data['cols'] = _parse_int(line)
                # Center Latitude of Upper Left Pixel of GRD image, or
                # range Offset(R0) from Peg in meters
                # Note: using convention of .rsc files for consitency
                # I.E. x_first, x_step, y_first, y_step
                elif re.match(_make_line_regex(ext, 'row_addr'), line):
                    ann_data['y_first'] = _parse_float(line)
                # Center Longitude of Upper Left Pixel
                elif re.match(_make_line_regex(ext, 'col_addr'), line):
                    ann_data['x_first'] = _parse_float(line)
                # GRD Latitude Pixel Spacing
                # the step is negative in the y (row) direction
                elif re.match(_make_line_regex(ext, 'row_mult'), line):
                    ann_data['y_step'] = _parse_float(line)
                # GRD Longitude Pixel Spacing or SLC R (range) Slant Post Spacing
                elif re.match(_make_line_regex(ext, 'col_mult'), line):
                    ann_data['x_step'] = _parse_float(line)
                # TODO: Add more parsing! whatever is useful from .ann file

        if verbose:
            logger.info(pprint.pformat(ann_data))
        return ann_data
