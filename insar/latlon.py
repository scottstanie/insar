from __future__ import division
import copy
from math import sin, cos, sqrt, atan2, radians
import os
import numpy as np
from insar import sario
from insar.log import get_log

logger = get_log()


class LatlonImage(object):
    def __init__(self, filename=None, data=None, dem_rsc_file=None, dem_rsc=None):
        """Can pass in either filenames to load, or 2D arrays/dem_rsc dicts"""
        # TODO: do we need to check that the rsc info matches the data?
        self.filename = filename
        if data is not None:
            self.data = data
        elif filename:
            self.data = sario.load(filename)
        else:
            raise ValueError("Need data or filename")

        if dem_rsc_file:
            self.dem_rsc_file = dem_rsc_file
        else:
            self.dem_rsc_file = sario.find_rsc_file(filename) if filename else None

        if dem_rsc:
            self.dem_rsc = dem_rsc
        elif self.dem_rsc_file:
            self.dem_rsc = sario.load(self.dem_rsc_file)
        else:
            self.dem_rsc = None

    def __str__(self):
        return "<LatlonImage %s>" % (self.filename or str(self.data.shape))

    def __repr__(self):
        return str(self)

    @property
    def shape(self):
        """Shape of Image ndarray"""
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, item):
        return self.data[item]

    def rowcol_to_latlon(self, row, col):
        return rowcol_to_latlon(row, col, self.dem_rsc)

    def distance(self, row_col1, row_col2):
        """Find the distance in km between two points on the image

        Args:
            row_col1 (tuple[int, int]): starting (row, col)
            row_col2 (tuple[int, int]): ending (row, col)

        Returns:
            float: distance in km between two points on LatlonImage
        """
        latlon1 = self.rowcol_to_latlon(*row_col1)
        latlon2 = self.rowcol_to_latlon(*row_col2)
        return latlon_to_dist(latlon1, latlon2)

    def crop(self, start_row, end_row, start_col, end_col):
        """Adjusts the old dem_rsc for a cropped data

        Takes the 'file_length' and 'width' keys for a cropped data
        and adjusts for the smaller size with a new dict

        Example:
        >>> im_test = np.arange(20).reshape((4, 5))
        >>> rsc_info = {'x_first': 1.0, 'y_first': 2.0, 'x_step': 0.1, 'y_step': 0.2, 'file_length': 1325,'width': 1000}
        >>> im = LatlonImage(data=im_test, dem_rsc=rsc_info)
        >>> im.crop(0, 3, 0, 2)
        >>> print(sorted(im.dem_rsc.items()))
        [('file_length', 3), ('width', 2), ('x_first', 1.0), ('x_step', 0.1), ('y_first', 2.0), ('y_step', 0.2)]
        >>> im2 = LatlonImage(data=im_test, dem_rsc=rsc_info)
        >>> im2.crop(1, 4, 2, 5)
        >>> print(sorted(im2.dem_rsc.items()))
        [('file_length', 3), ('width', 3), ('x_first', 1.1), ('x_step', 0.1), ('y_first', 2.4), ('y_step', 0.2)]

        """
        # Note: this will overwrite the old self.data
        # Do we want some copy version option?
        rsc_copy = copy.copy(self.dem_rsc)
        self.data = self.data[..., start_row:end_row, start_col:end_col]
        if self.data.ndim > 2:
            nlayers, nrows, ncols = self.data.shape
        else:
            nrows, ncols = self.data.shape

        rsc_copy['x_first'] = rsc_copy['x_first'] + rsc_copy['x_step'] * start_row
        rsc_copy['y_first'] = rsc_copy['y_first'] + rsc_copy['y_step'] * start_col

        rsc_copy['width'] = ncols
        rsc_copy['file_length'] = nrows
        self.dem_rsc = rsc_copy

    def blob_size(self, radius):
        """Finds the radius of a circle/blob on the LatlonImage in km"""
        nrows, ncols = self.shape
        midrow, midcol = nrows // 2, ncols // 2
        return self.distance((midrow, midcol), (midrow + radius, midcol + radius))


def LatlonStack(LatlonImage):
    """3D stack version of LatlonImage"""

    def __init__(self, stack_path=None, ext=None, data=None, dem_rsc_file=None, dem_rsc=None):
        """Can pass in either filenames to load, or 2D arrays/dem_rsc dicts"""
        # TODO: extract common stuff from this and Latlon Image
        self.stack_path = stack_path
        if data:
            self.data = data
        elif stack_path and ext:
            self.data = sario.load_stack(stack_path, ext)
        else:
            raise ValueError("Need `data` or `stack_path` and `ext`")

        if dem_rsc_file:
            self.dem_rsc_file = dem_rsc_file
        else:
            self.dem_rsc_file = os.path.join(stack_path, 'dem.rsc') if stack_path else None

        if dem_rsc:
            self.dem_rsc = dem_rsc
        elif self.dem_rsc_file:
            self.dem_rsc = sario.load(self.dem_rsc_file)
        else:
            self.dem_rsc = None


class DemTile(object):
    def __init__(self, rsc_file=None, rsc_data=None):
        if rsc_data:
            self.rsc_data = rsc_data
        elif rsc_file:
            self.rsc_data = sario.load(rsc_file)


def rowcol_to_latlon(row, col, rsc_data):
    """ Takes the row, col of a pixel and finds its lat/lon

    Can also pass numpy arrays of row, col.
    row, col must match size

    Args:
        row (int or ndarray): row number
        col (int or ndarray): col number
        rsc_data (dict): data output from load_dem_rsc

    Returns:
        tuple[float, float]: lat, lon for the pixel

    Example:
        >>> rsc_data = {"x_first": 1.0, "y_first": 2.0, "x_step": 0.2, "y_step": -0.1}
        >>> rowcol_to_latlon(7, 3, rsc_data)
        (1.4, 1.4)
    """
    # Force keys to lowercase
    rsc_data = {k.lower(): v for k, v in rsc_data.items()}
    start_lon = rsc_data["x_first"]
    start_lat = rsc_data["y_first"]
    lon_step, lat_step = rsc_data["x_step"], rsc_data["y_step"]
    lat = start_lat + (row - 1) * lat_step
    lon = start_lon + (col - 1) * lon_step

    return lat, lon


def latlon_to_dist(lat_lon_start, lat_lon_end, R=6378):
    """Find the distance between two lat/lon points on Earth

    Uses the haversine formula: https://en.wikipedia.org/wiki/Haversine_formula
    so it does not account for the ellopsoidal Earth shape. Will be with about
    0.5-1% of the correct value.

    Notes: lats and lons are in degrees, and the values used for R Earth
    (6373 km) are optimized for locations around 39 degrees from the equator

    Reference: https://andrew.hedges.name/experiments/haversine/

    Args:
        lat_lon_start (tuple[int, int]): (lat, lon) in degrees of start
        lat_lon_end (tuple[int, int]): (lat, lon) in degrees of end
        R (float): Radius of earth

    Returns:
        float: distance between two points in km

    Examples:
        >>> round(latlon_to_dist((38.8, -77.0), (38.9, -77.1)), 1)
        14.1
    """
    lat1, lon1 = lat_lon_start
    lat2, lon2 = lat_lon_end
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = (sin(dlat / 2)**2) + (cos(lat1) * cos(lat2) * sin(dlon / 2)**2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def grid(rows=None,
         cols=None,
         y_step=None,
         x_step=None,
         y_first=None,
         x_first=None,
         width=None,
         file_length=None,
         sparse=False,
         **kwargs):
    """Takes sizes and spacing info, creates a grid of values

    Args:
        rows (int): number of rows
        cols (int): number of cols
        y_step (float): spacing between rows
        x_step (float): spacing between cols
        y_first (float): starting location of first row at top
        x_first (float): starting location of first col on left
        sparse (bool): Optional (default False). Passed through to
            np.meshgrid to optionally conserve memory

    Returns:
        tuple[ndarray, ndarray]: the XX, YY grids of longitudes and lats

    Examples:
    >>> test_grid_data = {'cols': 2, 'rows': 3, 'x_first': -155.0, 'x_step': 0.01, 'y_first': 19.5, 'y_step': -0.2}
    >>> lons, lats = grid(**test_grid_data)
    >>> np.set_printoptions(legacy="1.13")
    >>> print(lons)
    [[-155.   -154.99]
     [-155.   -154.99]
     [-155.   -154.99]]
    >>> print(lats)
    [[ 19.5  19.5]
     [ 19.3  19.3]
     [ 19.1  19.1]]
    """
    rows = rows or file_length
    cols = cols or width
    x = np.linspace(x_first, x_first + (cols - 1) * x_step, cols).reshape((1, cols))
    y = np.linspace(y_first, y_first + (rows - 1) * y_step, rows).reshape((rows, 1))
    return np.meshgrid(x, y, sparse=sparse)


def grid_extent(rows=None,
                cols=None,
                y_step=None,
                x_step=None,
                y_first=None,
                x_first=None,
                file_length=None,
                width=None,
                **kwargs):
    """Takes sizes and spacing info, finds boundaries

    Used for `matplotlib.pyplot.imshow` keyword arg `extent`:
    extent : scalars (left, right, bottom, top)

    Args:
        rows (int): number of rows
        cols (int): number of cols
        y_step (float): spacing between rows
        x_step (float): spacing between cols
        y_first (float): starting location of first row at top
        x_first (float): starting location of first col on left
        file_length (int): alias for number of rows (used in dem.rsc)
            Not needed if `rows` is supplied
        width (int): alias for number of cols (used in dem.rsc)
            Not needed if `cols` is supplied

    Returns:
        tuple[float]: the boundaries of the latlon grid in order:
        (lon_left,lon_right,lat_bottom,lat_top)

    Examples:
    >>> test_grid_data = {'cols': 2, 'rows': 3, 'x_first': -155.0, 'x_step': 0.01, 'y_first': 19.5, 'y_step': -0.2}
    >>> print(grid_extent(**test_grid_data))
    (-155.0, -154.99, 19.1, 19.5)
    """
    rows = rows or file_length
    cols = cols or width
    return (x_first, x_first + x_step * (cols - 1), y_first + y_step * (rows - 1), y_first)


def grid_corners(**kwargs):
    """Takes sizes and spacing info, finds corner points in (x, y) form

    Returns:
        list[tuple[float]]: the corners of the latlon grid in order:
        (top right, top left, bottom left, bottom right)
    """
    left, right, bot, top = grid_extent(**kwargs)
    return [(right, top), (left, top), (left, bot), (right, bot)]


def grid_midpoint(**kwargs):
    """Takes sizes and spacing info, finds midpoint in (x, y) form

    Returns:
        tuple[float]: midpoint of the latlon grid
    """
    left, right, bot, top = grid_extent(**kwargs)
    return (left + right) / 2, (top + bot) / 2


def grid_size(**kwargs):
    """Takes rsc_data and gives width and height of box in km

    Returns:
        tupls[float, float]: width, height in km
    """
    left, right, bot, top = grid_extent(**kwargs)
    width = latlon_to_dist((top, left), (top, right))
    height = latlon_to_dist((top, left), (bot, right))
    return width, height


def grid_bounds(**kwargs):
    """Same to grid_extent, but in the order (left, bottom, right, top)"""
    left, right, bot, top = grid_extent(**kwargs)
    return left, bot, right, top


def grid_width_height(**kwargs):
    """Finds the width and height in deg of the latlon grid"""
    left, right, bot, top = grid_extent(**kwargs)
    return (right - left, top - bot)


def rot(angle, axis):
    """
    Find a 3x3 euler rotation matrix given an angle and axis.

    Rotation matrix used for rotating a vector about a single axis.

    Args:
        angle (float): angle in degrees to rotate
        axis (int): 1, 2 or 3
    """
    R = np.eye(3)
    cang = cos(np.deg2rad(angle))
    sang = sin(np.deg2rad(angle))
    if (axis == 1):
        R[1, 1] = cang
        R[2, 2] = cang
        R[1, 2] = sang
        R[2, 1] = -sang
    elif (axis == 2):
        R[0, 0] = cang
        R[2, 2] = cang
        R[0, 2] = -sang
        R[2, 0] = sang
    elif (axis == 3):
        R[0, 0] = cang
        R[1, 1] = cang
        R[1, 0] = -sang
        R[0, 1] = sang
    else:
        raise ValueError("axis must be 1, 2 or 2")
    return R


def rotate_xyz_to_enu(xyz, lat, lon):
    """Rotates a vector in XYZ coords to ENU

    Args:
        xyz (list[float], ndarray[float]): length 3 x, y, z coordinates
        lat (float): latitude of point to rotate into
        lon (float): longitude of point to rotate into
    """
    # Rotate about axis 3 with longitude, then axis 1 with latitude
    R3 = rot(90 + lon, 3)
    R1 = rot(90 - lat, 1)
    R = np.matmul(R3, R1)
    return np.matmul(R, xyz)


def convert_xyz_latlon_to_enu(lat_lons, xyz_array):
    return [rotate_xyz_to_enu(xyz, lat, lon) for (lat, lon), xyz in zip(lat_lons, xyz_array)]


def intersects1d(low1, high1, low2, high2):
    """Checks if two line segments intersect

    Example:
    >>> low1, high1 = [1, 5]
    >>> low2, high2 = [4, 6]
    >>> print(intersects1d(low1, high1, low2, high2))
    True
    >>> low2 = 5.5
    >>> print(intersects1d(low1, high1, low2, high2))
    False
    >>> high1 = 7
    >>> print(intersects1d(low1, high1, low2, high2))
    True
    """
    # Is this easier?
    # return not (high2 <= low1 or high2 <= low1)
    return high1 >= low2 and high2 >= low1


def intersects(box1, box2):
    """Returns true if box1 intersects box2

    box = (left, right, bot, top), same as matplotlib `extent` format

    Example:
    >>> box1 = (-105.1, -102.2, 31.4, 33.4)
    >>> box2 = (-103.4, -102.7, 30.9, 31.8)
    >>> print(intersects(box1, box2))
    True
    >>> box2 = (-103.4, -102.7, 30.9, 31.0)
    >>> print(intersects(box2, box1))
    False
    """
    left1, right1, bot1, top1 = box1
    left2, right2, bot2, top2 = box2
    return (intersects1d(left1, right1, left2, right2) and intersects1d(bot1, top1, bot2, top2))
