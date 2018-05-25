"""Digital Elevation Map (DEM) downloading/stitching/upsampling

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

"""
try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    PARALLEL = True
except ImportError:  # Python 2 doesn't have this :(
    PARALLEL = False
import collections
import math
import json
import os
import re
import sys
import requests
import subprocess
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from insar.geojson import geojson_to_bounds
from insar.log import get_log, log_runtime
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
    """Find location of directory to store .hgt downloads"""
    # TODO: Decide assumption that we're on linux (should I just get appdirs?)
    path = os.getenv('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
    path = os.path.join(path, 'insar')  # Make subfolder for our downloads
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class Downloader:
    def __init__(self, left, bottom, right, top, margin=0, parallel_ok=PARALLEL):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.margin = margin
        # AWS format for downloading SRTM1 .hgt tiles
        self.data_url = 'https://s3.amazonaws.com/elevation-tiles-prod/skadi'
        self.compressed_ext = '.gz'
        self.parallel_ok = parallel_ok

    @staticmethod
    def srtm1_tile_corner(lon, lat):
        """Integers for the bottom right corner of requested lon/lat"""
        return int(math.floor(lon)), int(math.floor(lat))

    def srtm1_tile_names(self):
        """Iterator over all tiles needed to cover the requested bounds

        yields:
            str: tile names to fit into data_url to be downloaded
                yielded in order of top left to bottom right
        """
        left_int, top_int = self.srtm1_tile_corner(self.left, self.top)
        right_int, bot_int = self.srtm1_tile_corner(self.right, self.bottom)
        # If exact integer was requested for top/right, assume tile with that number
        # at the top/right is acceptable (dont download the one above that)
        if isinstance(self.top, int):
            top_int -= 1
        if isinstance(self.right, int):
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
    def unzip_file(filepath):
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
            logger.info("{} alread exists, skipping.".format(local_filename))
        else:
            # On AWS these are gzipped: download, then unzip
            local_filename += self.compressed_ext
            with open(local_filename, 'wb') as f:
                response = self._download_hgt_tile(tile_name_str)
                f.write(response.content)
                logger.info("Writing to {}".format(local_filename))
            logger.info("Unzipping {}".format(local_filename))
            self.unzip_file(local_filename)

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
    """Class to combine separate .hgt tiles into one .dem file"""

    def __init__(self, tile_file_list):
        """List should come from Downloader.srtm1_tile_list()"""
        self.tile_file_list = [t.split('/')[1] for t in tile_file_list]

    def compute_shape(self):
        """Takes the tile list and computes the number of rows and columns"""
        lon_lat_tups = [start_lon_lat(t) for t in self.tile_file_list]
        # Unique each lat/lon: length of lats = num rows, lons = cols
        num_lons = len(set(tup[0] for tup in lon_lat_tups))
        num_lats = len(set(tup[1] for tup in lon_lat_tups))
        return (num_lats, num_lons)

    def load_tiles(self):
        file_list = [os.path.join(_get_cache_dir(), tile) for tile in self.tile_file_list]
        nrows, ncols = self.compute_shape()
        flist = np.array(file_list).reshape((nrows, ncols))
        row_list = []
        for idx, row in enumerate(flist):
            cur_row = np.hstack(sario.load_file(f) for f in row)
            # TODO: where to get 3601 from, magic number now
            cur_row = np.delete(cur_row, 3601 * list(range(1, ncols)), axis=1)
            if idx > 0:
                # For all except first block-row, delete repeated first row of data
                cur_row = np.delete(cur_row, 0, axis=0)
            row_list.append(cur_row)
        return np.vstack(row_list)

    def reshape(self):
        pass


def _up_size(cur_size, rate):
    """Calculates the number of points to be computed in the upsampling

    Example: 3 points at x = (0, 1, 2), rate = 2 becomes 5 points:
        x = (0, .5, 1, 1.5, 2)
    """
    return 1 + (cur_size - 1) * rate


def start_lon_lat(tilename):
    """Takes an SRTM1 data tilename and returns the first (lon, lat) point

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
    """
    lon_lat_regex = r'([NS])(\d+)([EW])(\d+)'
    match = re.match(lon_lat_regex, tilename)
    if not match:
        raise ValueError('Invalid SRTM1 tilename: must match {}'.format(lon_lat_regex))

    lat_str, lat, lon_str, lon = match.groups()

    # Only lon adjustment is negative it western hemisphere
    left_lon = -1 * float(lon) if lon_str == 'W' else float(lon)
    # No additions needed to lon: bottom left and top left are same
    # Only the lat gets added or subtracted
    top_lat = float(lat) + 1 if lat_str == 'N' else float(lat) - 1
    return (left_lon, top_lat)


def mosaic_dem(d1, d2):
    """Joins two .hgt files side by side, d1 left, d2 right"""
    D = np.concatenate((d1, d2), axis=1)
    nrows, ncols = d1.shape
    D = np.delete(D, nrows, axis=1)
    return D


def create_dem_rsc(srtm1_tile_list):
    """Takes a list of the SRTM1 tile names and outputs .dem.rsc file values

    See module docstring for example .dem.rsc file.

    Args:
        srtm1_tile_list (list[str]): names of tiles (e.g. N19W156)

    Returns:
        OrderedDict: key/value pairs in order to write to a .dem.rsc file
    """

    def _calc_x_y_firsts(tile_list):
        x_first = np.inf  # Start all the way east, take min (west)
        y_first = -np.inf  # Start all the way south, max is north
        for tile in tile_list:
            lon, lat = start_lon_lat(tile)
            x_first = min(x_first, lon)
            y_first = max(y_first, lat)
        return x_first, y_first

    # Use an OrderedDict for the key/value pairs so writing to file easy
    rsc_data = collections.OrderedDict.fromkeys(RSC_KEYS)
    rsc_data.update({
        'X_UNIT': 'degrees',
        'Y_UNIT': 'degrees',
        'Z_OFFSET': 0,
        'Z_SCALE': 1,
        'PROJECTION': 'LL',
    })

    # Remove paths from tile filenames, if they exist
    tile_names = [os.path.split(t)[1] for t in srtm1_tile_list]
    x_first, y_first = _calc_x_y_firsts(tile_names)
    # TODO: first out generalized way to get nx, ny.
    # Only using one pair left/right for now
    nx = 2
    ny = 1
    # TODO: figure out where to generalize for SRTM3
    num_pixels = 3601
    rsc_data.update({
        'WIDTH': nx * num_pixels - (nx - 1),
        'FILE_LENGTH': ny * num_pixels - (ny - 1)
    })
    rsc_data.update({'X_FIRST': x_first, 'Y_FIRST': y_first})
    rsc_data.update({'X_STEP': 1 / (num_pixels - 1), 'Y_STEP': -1 / (num_pixels - 1)})
    return rsc_data


def format_dem_rsc(rsc_data):
    """Creates the .dem.rsc file string from key/value pairs of an OrderedDict

    Output of function can be written to a file as follows
        with open('my.dem.rsc', 'w') as f:
            f.write(outstring)

    Args:
        rsc_data (OrderedDict): data about dem in ordered key/value format
            See `create_dem_rsc` output for example

    Returns:
        outstring (str) formatting string to be written to .dem.rsc

    """
    outstring = ""
    for field, value in rsc_data.items():
        # Files seemed to be left justified with 13 spaces? Not sure why 13
        if field.lower() in ('x_step', 'y_step'):
            # give step floats proper sig figs to not output scientific notation
            outstring += "{field:<13s}{val:0.12f}\n".format(field=field.upper(), val=value)
        else:
            outstring += "{field:<13s}{val}\n".format(field=field.upper(), val=value)

    return outstring


def upsample_dem_rsc(filepath, rate):
    """Creates a new .dem.rsc file for upsampled version

    Adjusts the FILE_LENGTH, WIDTH, X_STEP, Y_STEP for new rate

    Args:
        filepath (str) location of .dem.rsc file
        rate (int)

    Returns:
        str: file same as original with upsample adjusted numbers

    """
    outstring = ""
    rsc_data = sario.load_dem_rsc(filepath)
    for field, value in rsc_data.items():
        # Files seemed to be left justified with 13 spaces? Not sure why 13
        if field.lower() in ('width', 'file_length'):
            new_size = _up_size(value, rate)
            outstring += "{field:<13s}{val}\n".format(field=field.upper(), val=new_size)
        elif field.lower() in ('x_step', 'y_step'):
            # New is 1 + (size - 1) * rate, old is size, old rate is 1/(size-1)
            value /= rate
            # Also give step floats proper sig figs to not output scientific notation
            outstring += "{field:<13s}{val:0.12f}\n".format(field=field.upper(), val=value)
        else:
            outstring += "{field:<13s}{val}\n".format(field=field.upper(), val=value)

    return outstring


@log_runtime
def upsample_dem(dem_img, rate=3):
    """Interpolates a DEM to higher resolution for better InSAR quality

    TOO SLOW: scipy's interp for some reason isn't great
    Use upsample.c instead

    Args:
        dem_img: numpy.ndarray (int16)
        rate: int, default = 3

    Returns:
        numpy.ndarray (int16): original dem_img upsampled by `rate`. Needs
            to return same type since downstream scripts expect int16 DEMs

    """

    s1, s2 = dem_img.shape
    orig_points = (np.arange(1, s1 + 1), np.arange(1, s2 + 1))

    rgi = RegularGridInterpolator(points=orig_points, values=dem_img)

    # Make a grid from 1 to size (inclusive for mgrid), in both directions
    # 1j used by mgrid: makes numx/numy number of points exactly (like linspace)
    numx = _up_size(s1, rate)
    numy = _up_size(s2, rate)
    X, Y = np.mgrid[1:s1:(numx * 1j), 1:s2:(numy * 1j)]

    # vstack makes 2xN, num_pixels=(numx*numy): new_points will be a Nx2 matrix
    new_points = np.vstack([X.ravel(), Y.ravel()]).T

    # rgi expects Nx2 as input, and will output as a 1D vector
    # Should be same dtype (int16), and round used to not truncate 2.9 to 2
    return rgi(new_points).reshape(numx, numy).round().astype(dem_img.dtype)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            geojson = json.load(f)
    else:
        geojson = json.load(sys.stdin)

    bounds = geojson_to_bounds(geojson)
    logger.info("Bounds: %s", " ".join(str(b) for b in bounds))
    d = Downloader(*bounds)
    d.download_all()
