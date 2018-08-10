from __future__ import division
import copy
from math import sin, cos, sqrt, atan2, radians
# import os
import numpy as np
from insar import sario
from insar.log import get_log

logger = get_log()


class LatlonImage(object):
    def __init__(self, filename=None, image=None, dem_rsc_file=None, dem_rsc=None):
        """Can pass in either filenames to load, or 2D arrays/dem_rsc dicts"""
        # TODO: do we need to check that the rsc info matches the image?
        self.filename = filename
        if filename and not image:
            self.image = sario.load(filename)
        else:
            self.image = image

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
        return "<LatlonImage (%s)>" % self.filename or self.image.shape

    def __repr__(self):
        return str(self)

    @property
    def shape(self):
        return self.image.shape

    def __getitem__(self, item):
        return self.image[item]

    def crop(self, start_row, end_row, start_col, end_col):
        """Adjusts the old dem_rsc for a cropped image

        Takes the 'file_length' and 'width' keys for a cropped image
        and adjusts for the smaller size with a new dict

        Example:
        >>> im_test = np.arange(20).reshape((4, 5))
        >>> rsc_info = {'x_first': 1.0, 'y_first': 2.0, 'x_step': 0.1, 'y_step': 0.2, 'file_length': 1325,'width': 1000}
        >>> im = LatlonImage(image=im_test, dem_rsc=rsc_info)
        >>> im.crop(0, 3, 0, 2)
        >>> print(sorted(im.dem_rsc.items()))
        [('file_length', 3), ('width', 2), ('x_first', 1.0), ('x_step', 0.1), ('y_first', 2.0), ('y_step', 0.2)]
        >>> im2 = LatlonImage(image=im_test, dem_rsc=rsc_info)
        >>> im2.crop(1, 4, 2, 5)
        >>> print(sorted(im2.dem_rsc.items()))
        [('file_length', 3), ('width', 3), ('x_first', 1.1), ('x_step', 0.1), ('y_first', 2.4), ('y_step', 0.2)]

        """
        # Note: this will overwrite the old self.image
        # Do we want some copy version option?
        rsc_copy = copy.copy(self.dem_rsc)
        self.image = self.image[start_row:end_row, start_col:end_col]
        nrows, ncols = self.image.shape

        rsc_copy['x_first'] = rsc_copy['x_first'] + rsc_copy['x_step'] * start_row
        rsc_copy['y_first'] = rsc_copy['y_first'] + rsc_copy['y_step'] * start_col

        rsc_copy['width'] = ncols
        rsc_copy['file_length'] = nrows
        self.dem_rsc = rsc_copy


def rowcol_to_latlon(row, col, rsc_data=None):
    """ Takes the row, col of a pixel and finds its lat/lon

    Can also pass numpy arrays of row, col.
    row, col must match size

    Args:
        row (int or ndarray): row number
        col (int or ndarray): col number
        rsc_data (dict): data output from sario.load_dem_rsc

    Returns:
        tuple[float, float]: lat, lon for the pixel

    Example:
        >>> rsc_data = {"X_FIRST": 1.0, "Y_FIRST": 2.0, "X_STEP": 0.2, "Y_STEP": -0.1}
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


def latlon_grid(rows=None,
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
    >>> lons, lats = latlon_grid(**test_grid_data)
    >>> lons
    array([[-155.  , -154.99],
           [-155.  , -154.99],
           [-155.  , -154.99]])
    >>> lats
    array([[19.5, 19.5],
           [19.3, 19.3],
           [19.1, 19.1]])
    """
    rows = rows or file_length
    cols = cols or width
    x = np.linspace(x_first, x_first + (cols - 1) * x_step, cols).reshape((1, cols))
    y = np.linspace(y_first, y_first + (rows - 1) * y_step, rows).reshape((rows, 1))
    return np.meshgrid(x, y, sparse=sparse)


def latlon_grid_extent(rows=None,
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
    >>> print(latlon_grid_extent(**test_grid_data))
    (-155.0, -154.99, 19.1, 19.5)
    """
    rows = rows or file_length
    cols = cols or width
    return (x_first, x_first + x_step * (cols - 1), y_first + y_step * (rows - 1), y_first)


def latlon_grid_corners(**kwargs):
    """Takes sizes and spacing info, finds corner points in (x, y) form

    Returns:
        list[tuple[float]]: the corners of the latlon grid in order:
        (top right, top left, bottom left, bottom right)
    """
    left, right, bot, top = latlon_grid_extent(**kwargs)
    return [(right, top), (left, top), (left, bot), (right, bot)]


def latlon_grid_midpoint(**kwargs):
    """Takes sizes and spacing info, finds midpoint in (x, y) form

    Returns:
        tuple[float]: midpoint of the latlon grid
    """
    left, right, bot, top = latlon_grid_extent(**kwargs)
    return (left + right) / 2, (top + bot) / 2
