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
import collections
import os
import re
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from insar import sario

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


def upsample_dem(dem_img, rate=3):
    """Interpolates a DEM to higher resolution for better InSAR quality


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


def mosaic_dem(d1, d2):
    """Joins two .hgt files side by side, d1 left, d2 right"""
    D = np.concatenate((d1, d2), axis=1)
    nrows, ncols = d1.shape
    D = np.delete(D, nrows, axis=1)
    return D


def create_dem_rsc(SRTM1_tile_list):
    """Takes a list of the SRTM1 tile names and outputs .dem.rsc file values

    See module docstring for example .dem.rsc file.

    Args:
        SRTM1_tile_list (list[str]): names of tiles (e.g. N19W156)

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
    tile_names = [os.path.split(t)[1] for t in SRTM1_tile_list]
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
