"""Digital Elevation Map (DEM) downloading/stitching/upsampling

Module contains utilities for downloading all necessary .hgt files
for a lon/lat rectangle, stiches them into one DEM, and creates a
.dem.rsc file for SAR processing.

Note: NASA Earthdata requires a signup: https://urs.earthdata.nasa.gov/users/new
Once you have signed up, to avoid a username password prompt create/add to a .netrc
file in your home directory:

machine urs.earthdata.nasa.gov
    login yourusername
    password yourpassword

This will be handled if you run download_all by handle_credentials.
You can choose to save you username in a netrc file for future use

NASA MEaSUREs SRTM Version 3 (SRTMGL1) houses the data
    See https://lpdaac.usgs.gov/dataset_discovery/measures/measures_products_table/srtmgl3s_v003
    more info on SRTMGL1: https://cmr.earthdata.nasa.gov/search/concepts/C1000000240-LPDAAC_ECS.html

Example url: "http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/N06W001.SRTMGL1.hgt.zip"

Other option is to download from Mapzen's tile set on AWS:
    https://mapzen.com/documentation/terrain-tiles/formats/#skadi
These do not require a username and password.
They use the SRTM dataset within the US, but combine other sources to produce
1 arcsecond (30 m) resolution world wide.
    Example url: https://s3.amazonaws.com/elevation-tiles-prod/skadi/N19/N19W156.hgt

Example .dem.rsc (for N19W156.hgt and N19W155.hgt stitched horizontally):
        WIDTH         7201
        FILE_LENGTH   3601
        X_FIRST       -156.0
        Y_FIRST       20.0
        X_STEP        0.000277777777
        Y_STEP        -0.000277777777
        X_UNIT        degrees
        Y_UNIT        degrees
        Z_OFFSET      0
        Z_SCALE       1
        PROJECTION    LL

Made for python3, compatible with python2
"""
from __future__ import division
try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    PARALLEL = True
except ImportError:  # Python 2 doesn't have this :(
    PARALLEL = False
import collections
import getpass
import math
import netrc
import os
import re
import requests
import subprocess
import numpy as np

from insar.log import get_log
from insar.utils import floor_float
from insar import sario

try:
    input = raw_input  # Check for python 2
except NameError:
    pass
logger = get_log()
RSC_KEYS = [
    'WIDTH',
    'FILE_LENGTH',
    'X_FIRST',
    'Y_FIRST',
    'X_STEP',
    'Y_STEP',
    'X_UNIT',
    'Y_UNIT',
    'Z_OFFSET',
    'Z_SCALE',
    'PROJECTION',
]


def _get_cache_dir():
    """Find location of directory to store .hgt downloads

    Assuming linux, uses ~/.cache/insar/

    """
    path = os.getenv('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
    path = os.path.join(path, 'insar')  # Make subfolder for our downloads
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class Downloader:
    """Class to download and save SRTM1 tiles to create DEMs

    Attributes:
        bounds (tuple): lon, lat boundaries of a rectangle to download
        data_url (str): Base url where .hgt tiles are stored
        compress_type (str): format .hgt files are stored in online
        data_source (str): choices: NASA, AWS. See module docstring for explanation of sources
        parallel_ok (bool): true if using python3 or concurrent.futures installed

    Raises:
        ValueError: if data_source not a valid source string

    """
    VALID_SOURCES = ('NASA', 'AWS')
    DATA_URLS = {
        'NASA': "http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11",
        'AWS': "https://s3.amazonaws.com/elevation-tiles-prod/skadi"
    }
    COMPRESS_TYPES = {'NASA': 'zip', 'AWS': 'gz'}
    NASAHOST = 'urs.earthdata.nasa.gov'

    def __init__(self,
                 left,
                 bottom,
                 right,
                 top,
                 data_source='NASA',
                 netrc_file='~/.netrc',
                 parallel_ok=PARALLEL):
        self.bounds = (left, bottom, right, top)
        # AWS format for downloading SRTM1 .hgt tiles
        self.data_source = data_source
        if data_source not in self.VALID_SOURCES:
            raise ValueError('data_source must be one of: {}'.format(','.join(self.VALID_SOURCES)))
        self.data_url = self.DATA_URLS[data_source]
        self.compress_type = self.COMPRESS_TYPES[data_source]
        self.netrc_file = os.path.expanduser(netrc_file)
        self.parallel_ok = parallel_ok

    def _get_netrc_file(self):
        return netrc.netrc(self.netrc_file)

    def _has_nasa_netrc(self):
        try:
            n = self._get_netrc_file()
            # Check account exists, as well is having username and password
            return (self.NASAHOST in n.hosts and n.authenticators(self.NASAHOST)[0]
                    and n.authenticators(self.NASAHOST)[2])
        except (OSError, IOError):
            return False

    def _get_username_pass(self):
        """If netrc is not set up, get command line username and password"""
        print("Please enter NASA Earthdata credentials to download NASA hosted STRM.")
        print("See https://urs.earthdata.nasa.gov/users/new for signup info.")
        print("Or choose data_source=AWS for Mapzen tiles.")
        username = input("Username: ")
        password = getpass.getpass(prompt="Password (will not be displayed): ")
        save_to_netrc = input(
            "Would you like to save these to ~/.netrc (machine={}) for future use (y/n)?".format(
                self.NASAHOST))

        return username, password, save_to_netrc.lower().startswith('y')

    @staticmethod
    def _format_netrc(username, password):
        outstring = "machine {}\n".format(Downloader.NASAHOST)
        outstring += "\tlogin {}\n".format(username)
        outstring += "\tpassword {}\n".format(password)
        return outstring

    def handle_credentials(self):
        username, pw, do_save = self._get_username_pass()
        if do_save:
            try:
                n = self._get_netrc_file()
                n.hosts[self.NASAHOST] = (username, None, pw)
                outstring = n.__repr__()
            except (OSError, IOError):
                outstring = self._format_netrc(username, pw)

            with open(self.netrc_file, 'w') as f:
                f.write(outstring)

    @staticmethod
    def srtm1_tile_corner(lon, lat):
        """Integers for the bottom right corner of requested lon/lat

        Examples:
            >>> Downloader.srtm1_tile_corner(3.5, 5.6)
            (3, 5)
            >>> Downloader.srtm1_tile_corner(-3.5, -5.6)
            (-4, -6)
        """
        return int(math.floor(lon)), int(math.floor(lat))

    def srtm1_tile_names(self):
        """Iterator over all tiles needed to cover the requested bounds

        Args:
            None: bounds provided to Downloader __init__()

        Yields:
            str: tile names to fit into data_url to be downloaded
                yielded in order of top left to bottom right

        Examples:
            >>> bounds = (-155.7, 19.1, -154.7, 19.7)
            >>> d = Downloader(*bounds, data_source='NASA')
            >>> from types import GeneratorType  # To check the type
            >>> type(d.srtm1_tile_names()) == GeneratorType
            True
            >>> list(d.srtm1_tile_names())
            ['N19W156.SRTMGL1.hgt', 'N19W155.SRTMGL1.hgt']
            >>> bounds = [-156.0, 19.0, -154.0, 20.0]  # Show int bounds
            >>> list(d.srtm1_tile_names())
            ['N19W156.SRTMGL1.hgt', 'N19W155.SRTMGL1.hgt']
            >>> list(Downloader(*(10.1, -44.9, 10.1, -44.9), data_source='AWS').srtm1_tile_names())
            ['S45/S45E010.hgt']


        """

        left, bottom, right, top = self.bounds
        left_int, top_int = self.srtm1_tile_corner(left, top)
        right_int, bot_int = self.srtm1_tile_corner(right, bottom)
        # If exact integer was requested for top/right, assume tile with that number
        # at the top/right is acceptable (dont download the one above that)
        if isinstance(top, int) or int(top) == top:
            top_int -= 1
        if isinstance(right, int) or int(right) == right:
            right_int -= 1

        if self.data_source == 'AWS':
            tile_name_template = '{lat_str}/{lat_str}{lon_str}.hgt'
        elif self.data_source == 'NASA':
            tile_name_template = '{lat_str}{lon_str}.SRTMGL1.hgt'

        # Now iterate in same order in which they'll be stithced together
        for ilat in range(top_int, bot_int - 1, -1):  # north to south
            hemi_ns = 'N' if ilat >= 0 else 'S'
            lat_str = '{}{:02d}'.format(hemi_ns, abs(ilat))
            for ilon in range(left_int, right_int + 1):  # West to east
                hemi_ew = 'E' if ilon >= 0 else 'W'
                lon_str = '{}{:03d}'.format(hemi_ew, abs(ilon))

                yield tile_name_template.format(lat_str=lat_str, lon_str=lon_str)

    def _form_tile_url(self, tile_name_str):
        """Downloads a singles from AWS

        Args:
            tile_name_str (str): string name of tile
            e.g. N06W001.SRTMGL1.hgt.zip (usgs) or N19/N19W156.hgt.gz (aws)

        Returns:
            url: formatted url string with tile name

        Examples:
            >>> bounds = (-155.7, 19.1, -154.7, 19.7)
            >>> d = Downloader(*bounds, data_source='NASA')
            >>> d._form_tile_url('N19W155.SRTMGL1.hgt')
            ['N19W156.SRTMGL1.hgt', 'N19W155.SRTMGL1.hgt']
            >>> d = Downloader(*bounds, data_source='AWS')
            >>> d._form_tile_url('N19/N19W155.hgt')
            ['N19W156.SRTMGL1.hgt', 'N19W155.SRTMGL1.hgt']
        """
        if self.data_source == 'AWS':
            url = '{base}/{tile}.{ext}'.format(
                base=self.data_url, tile=tile_name_str, ext=self.compress_type)
        elif self.data_source == 'NASA':
            url = '{base}/{tile}.{ext}'.format(
                base=self.data_url, tile=tile_name_str, ext=self.compress_type)
        return url

    def _download_hgt_tile(self, url):
        """Example from https://lpdaac.usgs.gov/data_access/daac2disk "command line tips" """
        with requests.Session() as session:
            session.auth = (self.username, self.password)
            r1 = session.request('get', url)
            r = session.get(r1.url, auth=(self.username, self.password))
            if r.ok:
                logger.info("Downloading {}".format(url))
                return r

    @staticmethod
    def _unzip_file(filepath):
        """Unzips in place the .hgt files downloaded"""
        ext = sario.get_file_ext(filepath)
        if ext == '.gz':
            unzip_cmd = ['gunzip']
        elif ext == '.zip':
            # -o forces overwrite without prompt, -d specifices unzip directory
            unzip_cmd = 'unzip -o -d {}'.format(_get_cache_dir()).split(' ')
        subprocess.check_call(unzip_cmd + [filepath])

    def download_and_save(self, tile_name_str):
        """Download and save one single tile

        Args:
            tile_name_str (str): string name of tile
            e.g. N06W001.SRTMGL1.hgt.zip (usgs) or N19/N19W156.gz (aws)

        Returns:
            None
        """
        # Remove extra latitude portion N19: keep all in one folder, compressed
        local_filename = os.path.join(_get_cache_dir(), tile_name_str.split('/')[-1])
        if os.path.exists(local_filename):
            logger.info("{} already exists, skipping.".format(local_filename))
        else:
            # On AWS these are gzipped: download, then unzip
            local_filename += '.{}'.format(self.compress_type)
            with open(local_filename, 'wb') as f:
                response = self._download_hgt_tile(tile_name_str)
                f.write(response.content)
                logger.info("Writing to {}".format(local_filename))
            logger.info("Unzipping {}".format(local_filename))
            self._unzip_file(local_filename)

    def download_all(self):
        """Downloads and saves all tiles from tile list"""
        if self.data_source == 'NASA' and not self._has_nasa_netrc():
            self.handle_credentials()

        if self.parallel_ok:
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_tile = {
                    executor.submit(self.download_and_save, tile): tile
                    for tile in self.srtm1_tile_names()
                }
                for future in as_completed(future_to_tile):
                    future.result()
                    logger.info('Finished {}'.format(future_to_tile[future]))

        else:
            for tile_name_str in self.srtm1_tile_names():
                self.download_and_save(tile_name_str)


class Stitcher:
    """Class to combine separate .hgt tiles into one .dem file

    Attributes:
        tile_file_list (list[str]) names of .hgt tiles as saved from download
            E.g.: ['N19W156.hgt', 'N19W155.hgt'] (not ['N19/N19W156.hgt',...])
        num_pixels (int): size of the squares of the .hgt files
            Assumes 3601 for SRTM1 (SRTM3 not yet implemented)

    """

    def __init__(self, tile_file_list, num_pixels=3601):
        """List should come from Downloader.srtm1_tile_names()"""
        # Remove extra stuff from either NASA or AWS
        self.tile_file_list = [t.replace('SRTMGL1.', '').split('/')[-1] for t in tile_file_list]
        # Assuming SRTM1: 3601 x 3601 squares
        self.num_pixels = num_pixels

    @property
    def shape(self):
        """Number of rows/columns in pixels for stitched .dem

        Uses the blockshape property, along with num_pixels property
        Returned as a tuple

        Examples:
            >>> s = Stitcher(['N19/N19W156.hgt', 'N19/N19W155.hgt'])
            >>> s.shape
            (3601, 7201)
        """
        blockrows, blockcols = self.blockshape
        return (self._total_length(blockrows), self._total_length(blockcols))

    def _total_length(self, numblocks):
        """Computes the total number of pixels in one dem from numblocks"""
        return numblocks * (self.num_pixels - 1) + 1

    @property
    def blockshape(self):
        """Number of tile in rows cols"""
        return self._compute_shape()

    def _compute_shape(self):
        """Takes the tile list and computes the number of tile rows and tile cols

        Figures out how many lons wide and lats tall the tile array spans
        Note: This is not the total number of pixels, which can be found in .shape

        Examples:
            >>> s = Stitcher(['N19/N19W156.hgt', 'N19/N19W155.hgt'])
            >>> s._compute_shape()
            (1, 2)
            >>> s = Stitcher(['N19W156.SRTMGL1.hgt', 'N19W155.SRTMGL1.hgt'])
            >>> s._compute_shape()
            (1, 2)
        """
        lon_lat_tups = [start_lon_lat(t) for t in self.tile_file_list]
        # Unique each lat/lon: length of lats = num rows, lons = cols
        num_lons = len(set(tup[0] for tup in lon_lat_tups))
        num_lats = len(set(tup[1] for tup in lon_lat_tups))
        return (num_lats, num_lons)

    def _create_file_array(self):
        """Finds filenames and reshapes into numpy.array matching DEM shape

        Examples:
            >>> s = Stitcher(['N19/N19W156.hgt', 'N19/N19W155.hgt', 'N18/N18W156.hgt', 'N18/N18W155.hgt'])
            >>> print(s._create_file_array())
            [['N19W156.hgt' 'N19W155.hgt']
             ['N18W156.hgt' 'N18W155.hgt']]
            >>> s = Stitcher(['N19W156.SRTMGL1.hgt', 'N19W155.SRTMGL1.hgt', 'N18W156.SRTMGL1.hgt', 'N18W155.SRTMGL1.hgt'])
            >>> print(s._create_file_array())
            [['N19W156.hgt' 'N19W155.hgt']
             ['N18W156.hgt' 'N18W155.hgt']]
        """
        nrows, ncols = self.blockshape
        return np.array(self.tile_file_list).reshape((nrows, ncols))

    def load_and_stitch(self):
        """Function to load combine .hgt tiles

        Uses hstack first on rows, then vstacks rows together.
        Also handles the deleting of overlapped rows/columns of SRTM tiles
        TODO: break this up to more testable chunks

        Returns:
            numpy.array: the stitched .hgt tiles in 2D np.array
        """
        row_list = []
        # ncols in the number of .hgt blocks wide
        _, ncols = self.blockshape
        flist = self._create_file_array()
        for idx, row in enumerate(flist):
            cur_row = np.hstack(sario.load_file(os.path.join(_get_cache_dir(), f)) for f in row)
            # Delete columns: 3601*[1, 2,... not-including last column]
            delete_cols = self.num_pixels * np.arange(1, ncols)
            cur_row = np.delete(cur_row, delete_cols, axis=1)
            if idx > 0:
                # For all except first block-row, delete repeated first row of data
                cur_row = np.delete(cur_row, 0, axis=0)
            row_list.append(cur_row)
        return np.vstack(row_list)

    def _find_step_sizes(self, ndigits=12):
        """Calculates the step size for the dem.rsc

        Note: assuming same step size in x and y direction

        Args:
            ndigits (int) default=12 because that's what was given

        Returns:
            (float, float): x_step, y_step

        Example:
            >>> s = Stitcher(['N19/N19W156.hgt', 'N19/N19W155.hgt'])
            >>> print(s._find_step_sizes())
            (0.000277777777, -0.000277777777)
        """
        step_size = floor_float(1 / (self.num_pixels - 1), ndigits)
        return (step_size, -1 * step_size)

    def create_dem_rsc(self):
        """Takes a list of the SRTM1 tile names and outputs .dem.rsc file values

        See module docstring for example .dem.rsc file.

        Args:
            srtm1_tile_list (list[str]): names of tiles (e.g. N19W156)
                must be sorted with top-left tile first, as in from
                output of Downloader.srtm1_tile_names

        Returns:
            OrderedDict: key/value pairs in order to write to a .dem.rsc file

        Examples:
            >>> s = Stitcher(['N19/N19W156.hgt', 'N19/N19W155.hgt'])
            >>> s.create_dem_rsc()
            OrderedDict([('WIDTH', 7201), ('FILE_LENGTH', 3601), ('X_FIRST', -156.0), ('Y_FIRST', 20.0), ('X_STEP', 0.000277777777), ('Y_STEP', -0.000277777777), ('X_UNIT', 'degrees'), ('Y_UNIT', 'degrees'), ('Z_OFFSET', 0), ('Z_SCALE', 1), ('PROJECTION', 'LL')])
        """

        # Use an OrderedDict for the key/value pairs so writing to file easy
        rsc_dict = collections.OrderedDict.fromkeys(RSC_KEYS)
        rsc_dict.update({
            'X_UNIT': 'degrees',
            'Y_UNIT': 'degrees',
            'Z_OFFSET': 0,
            'Z_SCALE': 1,
            'PROJECTION': 'LL',
        })

        # Remove paths from tile filenames, if they exist
        x_first, y_first = start_lon_lat(self.tile_file_list[0])
        nrows, ncols = self.shape
        # TODO: figure out where to generalize for SRTM3
        rsc_dict.update({'WIDTH': ncols, 'FILE_LENGTH': nrows})
        rsc_dict.update({'X_FIRST': x_first, 'Y_FIRST': y_first})

        x_step, y_step = self._find_step_sizes()
        rsc_dict.update({'X_STEP': x_step, 'Y_STEP': y_step})
        return rsc_dict

    def format_dem_rsc(self, rsc_dict):
        """Creates the .dem.rsc file string from key/value pairs of an OrderedDict

        Output of function can be written to a file as follows
            with open('my.dem.rsc', 'w') as f:
                f.write(outstring)

        Args:
            rsc_dict (OrderedDict): data about dem in ordered key/value format
                See `create_dem_rsc` output for example

        Returns:
            outstring (str) formatting string to be written to .dem.rsc

        Example:
            >>> s = Stitcher(['N19/N19W156.hgt', 'N19/N19W155.hgt'])
            >>> rsc_dict = s.create_dem_rsc()
            >>> print(s.format_dem_rsc(rsc_dict))
            WIDTH         7201
            FILE_LENGTH   3601
            X_FIRST       -156.0
            Y_FIRST       20.0
            X_STEP        0.000277777777
            Y_STEP        -0.000277777777
            X_UNIT        degrees
            Y_UNIT        degrees
            Z_OFFSET      0
            Z_SCALE       1
            PROJECTION    LL
            <BLANKLINE>

        Note: ^^ <BLANKLINE> is doctest's way of saying it ends in newline
        """
        outstring = ""
        for field, value in rsc_dict.items():
            # Files seemed to be left justified with 13 spaces? Not sure why 13
            if field.lower() in ('x_step', 'y_step'):
                # give step floats proper sig figs to not output scientific notation
                outstring += "{field:<14s}{val:0.12f}\n".format(field=field.upper(), val=value)
            else:
                outstring += "{field:<14s}{val}\n".format(field=field.upper(), val=value)

        return outstring


def _up_size(cur_size, rate):
    """Calculates the number of points to be computed in the upsampling

    Example: 3 points at x = (0, 1, 2), rate = 2 becomes 5 points:
        x = (0, .5, 1, 1.5, 2)
        >>> _up_size(3, 2)
        5
    """
    return 1 + (cur_size - 1) * rate


def start_lon_lat(tilename):
    """Takes an SRTM1 data tilename and returns the first (lon, lat) point

    The reverse of Downloader.srtm1_tile_names()

    Used for .rsc file formation to make X_FIRST and Y_FIRST
    The names of individual data tiles refer to the longitude
    and latitude of the lower-left (southwest) corner of the tile.

    Example: N19W156.hgt refers to `bottom left` corner, while data starts
    at top left. This would return (X_FIRST, Y_FIRST) = (-156.0, 20.0)

    Args:
        tilename (str): name of .hgt file for SRTM1 tile

    Returns:
        tuple (float, float) of first (lon, lat) point in .hgt file

    Raises:
        ValueError: if regex match fails on tilename

    Examples:
        >>> start_lon_lat('N19W156.hgt')
        (-156.0, 20.0)
        >>> start_lon_lat('S5E6.hgt')
        (6.0, -4.0)
        >>> start_lon_lat('Notrealname.hgt')
        Traceback (most recent call last):
           ...
        ValueError: Invalid SRTM1 tilename: Notrealname.hgt, must match ([NS])(\d+)([EW])(\d+).hgt

    """
    lon_lat_regex = r'([NS])(\d+)([EW])(\d+).hgt'
    tilename = tilename.replace('SRTMGL1.', '')  # Remove NASA addition, if exists
    match = re.match(lon_lat_regex, tilename)
    if not match:
        raise ValueError('Invalid SRTM1 tilename: {}, must match {}'.format(
            tilename, lon_lat_regex))

    lat_str, lat, lon_str, lon = match.groups()

    # Only lon adjustment is negative it western hemisphere
    left_lon = -1 * float(lon) if lon_str == 'W' else float(lon)
    # No additions needed to lon: bottom left and top left are same
    # Only the lat gets added or subtracted
    top_lat = float(lat) + 1 if lat_str == 'N' else -float(lat) + 1
    return (left_lon, top_lat)


def upsample_dem_rsc(rate=None, rsc_dict=None, rsc_filepath=None):
    """Creates a new .dem.rsc file for upsampled version

    Adjusts the FILE_LENGTH, WIDTH, X_STEP, Y_STEP for new rate

    Args:
        rate (int): rate by which to upsample the DEM
        rsc_dict (str): Optional, the rsc data from Stitcher.create_dem_rsc()
        filepath (str): Optional, location of .dem.rsc file

    Note: Must supply only one of rsc_dict or rsc_filepath

    Returns:
        str: file same as original with upsample adjusted numbers

    Raises:
        TypeError: if neither (or both) rsc_filepath and rsc_dict are given

    """
    if rsc_dict and rsc_filepath:
        raise TypeError("Can only give one of rsc_dict or rsc_filepath")
    elif not rsc_dict and not rsc_filepath:
        raise TypeError("Must give at least one of rsc_dict or rsc_filepath")
    elif not rate:
        raise TypeError("Must supply rate for upsampling")

    if rsc_filepath:
        rsc_dict = sario.load_dem_rsc(rsc_filepath)

    outstring = ""
    for field, value in rsc_dict.items():
        # Files seemed to be left justified with 13 spaces? Not sure why 13
        # TODO: its 14- but fix this and previous formatting to be DRY
        if field.lower() in ('width', 'file_length'):
            new_size = _up_size(value, rate)
            outstring += "{field:<14s}{val}\n".format(field=field.upper(), val=new_size)
        elif field.lower() in ('x_step', 'y_step'):
            # New is 1 + (size - 1) * rate, old is size, old rate is 1/(size-1)
            value /= rate
            # Also give step floats proper sig figs to not output scientific notation
            outstring += "{field:<14s}{val:0.12f}\n".format(field=field.upper(), val=value)
        else:
            outstring += "{field:<14s}{val}\n".format(field=field.upper(), val=value)

    return outstring


def find_bounding_idxs(bounds, x_step, y_step, x_first, y_first):
    """Finds the indices of stitched dem to crop bounding box
    Also finds the new x_start and y_start after cropping.

    Note: x_start, y_start could be different from bounds
    if steps didnt exactly match, but should be further up and left

    Takes the desired bounds, .rsc data from stitched dem,
    Examples:
        >>> bounds = (-155.49, 19.0, -154.5, 19.51)
        >>> x_step = 0.1
        >>> y_step = -0.1
        >>> x_first = -156
        >>> y_first = 20.0
        >>> print(find_bounding_idxs(bounds, x_step, y_step, x_first, y_first))
        ((5, 10, 15, 4), (-155.5, 19.6))
    """

    left, bot, right, top = bounds
    left_idx = int(math.floor((left - x_first) / x_step))
    right_idx = int(math.ceil((right - x_first) / x_step))
    # Note: y_step will be negative for these
    top_idx = int(math.floor((top - y_first) / y_step))
    bot_idx = int(math.ceil((bot - y_first) / y_step))
    new_x_first = x_first + x_step * left_idx
    new_y_first = y_first + y_step * top_idx  # Again: y_step negative
    return (left_idx, bot_idx, right_idx, top_idx), (new_x_first, new_y_first)


def crop_stitched_dem(bounds, stitched_dem, rsc_data):
    """Takes the output of Stitcher.load_and_stitch, crops to bounds

    Args:
        bounds (tuple[float]): (left, bot, right, top) lats and lons of
            desired bounding box for the DEM
        stitched_dem (numpy.array, 2D): result from .hgt files
            through Stitcher.load_and_stitch()
        rsc_data (dict): data from .dem.rsc file, from Stitcher.create_dem_rsc

    Returns:
        numpy.array: a cropped version of the bigger stitched_dem
    """
    indexes, new_starts = find_bounding_idxs(
        bounds,
        rsc_data['X_STEP'],
        rsc_data['Y_STEP'],
        rsc_data['X_FIRST'],
        rsc_data['Y_FIRST'],
    )
    left_idx, bot_idx, right_idx, top_idx = indexes
    cropped_dem = stitched_dem[top_idx:bot_idx, left_idx:right_idx]
    new_sizes = cropped_dem.shape
    return cropped_dem, new_starts, new_sizes


def rsc_bounds(rsc_data):
    """Uses the x/y and step data from a .rsc file to generate LatLonBox for .kml"""
    north = rsc_data['Y_FIRST']
    west = rsc_data['X_FIRST']
    east = west + rsc_data['WIDTH'] * rsc_data['X_STEP']
    south = north + rsc_data['FILE_LENGTH'] * rsc_data['Y_STEP']
    return {'north': north, 'south': south, 'east': east, 'west': west}


def create_kml(rsc_data, tif_filename, title="Title", desc="Description"):
    """Make a simply kml file to display a tif in Google Earth from rsc_data"""
    north, south, east, west = rsc_bounds(rsc_data)
    template = """\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://earth.google.com/kml/2.2">
<GroundOverlay>
    <name> {title} </name>
    <description> {description} </description>
    <Icon>
          <href> {tif_filename} </href>
    </Icon>
    <LatLonBox>
        <north> {north} </north>
        <south> {south} </south>
        <east> {east} </east>
        <west> {west} </west>
    </LatLonBox>
</GroundOverlay>
</kml>
"""
    output = template.format(
        title=title, description=desc, tif_filename=tif_filename, **rsc_bounds(rsc_data))

    return output
