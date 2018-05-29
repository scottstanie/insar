"""Digital Elevation Map (DEM) downloading/stitching/upsampling

Module contains utilities for downloading all necessary .hgt files
for a lon/lat rectangle, stiches them into one DEM, and creates a
.dem.rsc file for SAR processing.

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

Make for python3, compatible with python2
"""
from __future__ import division
try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    PARALLEL = True
except ImportError:  # Python 2 doesn't have this :(
    PARALLEL = False
import collections
import math
import os
import re
import requests
import subprocess
import numpy as np

from insar.log import get_log
from insar.utils import floor_float
from insar import sario

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
        compressed_ext (str): format .hgt files are stored in online
        parallel_ok (bool): true if using python3 or concurrent.futures installed

    """

    def __init__(self, left, bottom, right, top, parallel_ok=PARALLEL):
        self.bounds = (left, bottom, right, top)
        # AWS format for downloading SRTM1 .hgt tiles
        self.data_url = 'https://s3.amazonaws.com/elevation-tiles-prod/skadi'
        self.compressed_ext = '.gz'
        self.parallel_ok = parallel_ok

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
            >>> d = Downloader(*bounds)
            >>> type(d.srtm1_tile_names())
            <class 'generator'>
            >>> list(d.srtm1_tile_names())
            ['N19/N19W156.hgt', 'N19/N19W155.hgt']
            >>> list(Downloader(*(10.1, -44.9, 10.1, -44.9)).srtm1_tile_names())
            ['S45/S45E010.hgt']

            >>> bounds = [-156.0, 19.0, -154.0, 20.0]  # Show int bounds
            >>> list(d.srtm1_tile_names())
            ['N19/N19W156.hgt', 'N19/N19W155.hgt']

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

        tile_name_template = '{lat_str}/{lat_str}{lon_str}.hgt'

        # Now iterate in same order in which they'll be stithced together
        for ilat in range(top_int, bot_int - 1, -1):  # north to south
            hemi_ns = 'N' if ilat >= 0 else 'S'
            lat_str = '{}{:02d}'.format(hemi_ns, abs(ilat))
            for ilon in range(left_int, right_int + 1):  # West to east
                hemi_ew = 'E' if ilon >= 0 else 'W'
                lon_str = '{}{:03d}'.format(hemi_ew, abs(ilon))

                yield tile_name_template.format(lat_str=lat_str, lon_str=lon_str)

    def _download_hgt_tile(self, tile_name_str):
        """Downloads a singles from AWS

        Args:
            tile_name_str (str): string name of tile on AWS (e.g. N19/N19W156.hgt)

        Returns:
            None
        """
        url = '{base}/{tile}{ext}'.format(
            base=self.data_url, tile=tile_name_str, ext=self.compressed_ext)
        logger.info("Downloading {}".format(url))
        return requests.get(url)

    @staticmethod
    def _unzip_file(filepath):
        """Unzips in place the .hgt files downloaded"""
        ext = sario.get_file_ext(filepath)
        if ext == '.gz':
            unzip_cmd = 'gunzip'
        elif ext == '.zip':
            unzip_cmd = 'unzip'
        subprocess.check_call([unzip_cmd, filepath])

    def download_and_save(self, tile_name_str):
        """Download and save one single tile

        Args:
            tile_name_str (str): string name of tile on AWS (e.g. N19/N19W156.hgt)

        Returns:
            None
        """
        # Remove extra latitude portion N19: keep all in one folder, gzipped
        local_filename = os.path.join(_get_cache_dir(), tile_name_str.split('/')[1])
        if os.path.exists(local_filename):
            logger.info("{} already exists, skipping.".format(local_filename))
        else:
            # On AWS these are gzipped: download, then unzip
            local_filename += self.compressed_ext
            with open(local_filename, 'wb') as f:
                response = self._download_hgt_tile(tile_name_str)
                f.write(response.content)
                logger.info("Writing to {}".format(local_filename))
            logger.info("Unzipping {}".format(local_filename))
            self._unzip_file(local_filename)

    def download_all(self):
        """Downloads and saves all tiles from tile list"""
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
            Assumes 3601 fo SRTM1 (SRTM3 not yet implemented)

    """

    def __init__(self, tile_file_list, num_pixels=3601):
        """List should come from Downloader.srtm1_tile_names()"""
        self.tile_file_list = [t.split('/')[1] for t in tile_file_list]
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
        """
        lon_lat_tups = [start_lon_lat(t) for t in self.tile_file_list]
        # Unique each lat/lon: length of lats = num rows, lons = cols
        num_lons = len(set(tup[0] for tup in lon_lat_tups))
        num_lats = len(set(tup[1] for tup in lon_lat_tups))
        return (num_lats, num_lons)

    def _create_file_array(self):
        """Finds filenames and reshapes into numpy.array matching DEM shape

        Examples:
            >>> s2 = Stitcher(['N19/N19W156.hgt', 'N19/N19W155.hgt', 'N18/N18W156.hgt', 'N18/N18W155.hgt'])
            >>> print(s2._create_file_array())
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
        ndigits = 12
        step_size = floor_float(1 / (self.num_pixels - 1), ndigits)
        rsc_dict.update({'X_STEP': step_size, 'Y_STEP': -1 * step_size})
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

        Examples:
            >>> s = Stitcher(['N19/N19W156.hgt', 'N19/N19W155.hgt'])
            >>> rsc_dict = s.create_dem_rsc()
            >>> print(s.format_dem_rsc(rsc_dict))
            WIDTH        7201
            FILE_LENGTH  3601
            X_FIRST      -156.0
            Y_FIRST      20.0
            X_STEP       0.000277777777
            Y_STEP       -0.000277777777
            X_UNIT       degrees
            Y_UNIT       degrees
            Z_OFFSET     0
            Z_SCALE      1
            PROJECTION   LL
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
        ValueError: Invalid SRTM1 tilename: must match ([NS])(\d+)([EW])(\d+).hgt

    """
    lon_lat_regex = r'([NS])(\d+)([EW])(\d+).hgt'
    match = re.match(lon_lat_regex, tilename)
    if not match:
        raise ValueError('Invalid SRTM1 tilename: must match {}'.format(lon_lat_regex))

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
