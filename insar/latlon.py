from __future__ import division
import copy
from numpy import sin, cos, sqrt, arctan2, radians
# import os
import numpy as np
from insar import sario, utils, kml
from insar.log import get_log

logger = get_log()


class LatlonImage(np.ndarray):
    def __new__(cls, data=None, filename=None, dem_rsc_file=None, dem_rsc=None):
        """Can pass in either filenames to load, or 2D arrays/dem_rsc dicts

        https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        """
        # TODO: do we need to check that the rsc info matches the data?
        if data is None and filename is None:
            raise ValueError("Need data or filename")
        elif data is None and filename is not None:
            data = sario.load(filename)

        obj = np.asarray(data).view(cls)
        obj.filename = filename

        if dem_rsc_file:
            obj.dem_rsc_file = utils.fullpath(dem_rsc_file)
        else:
            obj.dem_rsc_file = sario.find_rsc_file(filename) if filename else None

        if dem_rsc:
            obj.dem_rsc = dem_rsc
        elif obj.dem_rsc_file:
            obj.dem_rsc = sario.load(obj.dem_rsc_file)
        else:
            obj.dem_rsc = None

        # Also make each element in the dem_rsc dict an object attr
        # e.g. : img.x_first
        if obj.dem_rsc:
            for k, v in obj.dem_rsc.items():
                setattr(obj, k, v)

        if obj.dem_rsc is not None:
            if (obj.file_length, obj.width) != obj.shape:
                raise ValueError("Shape %s does not equal dem_rsc data (%s, %s)" %
                                 (obj.shape, obj.file_length, obj.width))

        if not hasattr(obj, 'points'):
            obj.points = []

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.filename = getattr(obj, 'filename', None)
        self.dem_rsc_file = getattr(obj, 'dem_rsc_file', None)
        self.dem_rsc = getattr(obj, 'dem_rsc', None)
        self.points = getattr(obj, 'points', None)

    def __getitem__(self, items):
        """Runs on access/slicing: we want to adjust the dem_rsc

        Will get the right starts and steps to pass to `crop_rsc_data`
        """
        sliced_out = super(LatlonImage, self).__getitem__(items)
        # __getitem__ called multiple times: only do extra on first
        if not isinstance(sliced_out, LatlonImage):
            return sliced_out

        try:
            row_slice, col_slice = items
        except TypeError:
            # Note: this means they did A[100] or A[2:4], not A[2:4,:]
            # We need this to stay 2D!
            raise ValueError("Can only do 2D slices on %s" % self.__class__.__name__)
        except ValueError:
            # Here they passed, for instance, an ndarray to index
            # We'll assume that this indexing will destroy the dem_rsc data
            # Since mask will destroy the shape to 1D
            sliced_out.dem_rsc = None
            return sliced_out

        # print(sliced_out.shape)
        nrows, ncols = sliced_out.shape

        row_start, row_step = row_slice.start, row_slice.step
        col_start, col_step = col_slice.start, col_slice.step
        new_rsc_data = self.crop_rsc_data(
            self.dem_rsc,
            row_start,
            col_start,
            nrows,
            ncols,
            row_step,
            col_step,
        )

        sliced_out.dem_rsc = new_rsc_data
        return sliced_out

    @staticmethod
    def crop_rsc_data(dem_rsc, row_start, col_start, nrows, ncols, row_step=1, col_step=1):
        """Adjusts the old dem_rsc for a cropped data

        Takes the 'file_length' and 'width' keys for a cropped data
        and adjusts for the smaller size with a new dict

        Returns:
            dict: copy of original rsc dict with items modified for crop

        Example:
        >>> im_test = np.arange(30).reshape((6, 5))
        >>> rsc_info = {'x_first': 1.0, 'y_first': 2.0, 'x_step': 0.1, 'y_step': -0.2, 'file_length': 6,'width': 5}
        >>> im = LatlonImage(data=im_test, dem_rsc=rsc_info)
        >>> out = im.crop_rsc_data(rsc_info, None, None, 2, 2)
        >>> print(out['width'], out['file_length'])
        2 2
        >>> out = im.crop_rsc_data(rsc_info, 1, 1, 2, 2)
        >>> print(out['x_first'], out['y_first'])
        1.1 1.8
        >>> out = im.crop_rsc_data(rsc_info, None, None, 2, 2, 2, 2)
        >>> print(out['x_step'], out['y_step'])
        0.2 -0.4
        >>> im2 = LatlonImage(data=im_test, dem_rsc=None)
        >>> print(im.crop_rsc_data(None, 1, 4, 2, 5))
        None
        """
        if dem_rsc is None:
            return None

        # Adjust and Nones from the slice object
        row_start = row_start or 0
        col_start = col_start or 0
        row_step = row_step or 1
        col_step = col_step or 1

        rsc_copy = copy.copy(dem_rsc)
        # Move forward the starting row/col from where it used to be
        rsc_copy['x_first'] = rsc_copy['x_first'] + rsc_copy['x_step'] * col_start
        rsc_copy['y_first'] = rsc_copy['y_first'] + rsc_copy['y_step'] * row_start

        rsc_copy['width'] = ncols
        rsc_copy['file_length'] = nrows
        # After moving start, now adjust step sizes
        rsc_copy['x_step'] *= col_step
        rsc_copy['y_step'] *= row_step
        return rsc_copy

    def rowcol_to_latlon(self, row, col):
        return rowcol_to_latlon(row, col, self.dem_rsc)

    @property
    def extent(self):
        if self.dem_rsc:
            return grid_extent(**self.dem_rsc)

    @property
    def top_left(self):
        """Returns the (lat, lon) of the top left corner"""
        if self.dem_rsc:
            return self.x_first, self.y_first

    @property
    def last_lat(self):
        """The latitude of the last row of the image"""
        if self.dem_rsc:
            return self.y_first + self.y_step * (self.shape[0] - 1)

    @property
    def last_lon(self):
        """The longitude of the last column of the image"""
        if self.dem_rsc:
            return self.x_first + self.x_step * (self.shape[1] - 1)

    @property
    def lat_step(self):
        """The latitude increment for each pixel"""
        if self.dem_rsc:
            return self.y_step

    @property
    def lon_step(self):
        """The longitude increment of each pixel"""
        if self.dem_rsc:
            return self.x_step

    def nearest_pixel(self, lat=None, lon=None):
        """Find the nearest row or col number to a given lat or lon

        Returns (tuple[int, int]): If both given, a pixel (row, col) is returned
            Otherwise if only one, it is (None, col) or (row, None)
        """
        out_row_col = [None, None]
        if lat:
            row_idx = (lat - self.y_first) / self.y_step
            if row_idx >= 0 and row_idx < self.shape[0]:
                out_row_col[0] = int(round(row_idx))
        if lon:
            col_idx = (lon - self.x_first) / self.x_step
            if col_idx >= 0 and col_idx < self.shape[1]:
                out_row_col[1] = int(round(col_idx))

        return tuple(out_row_col)

    def to_kml(self, tif_filename, title=None, desc="Description", kml_out=None):
        """Convert the dem.rsc data into a kml string"""
        return kml.create_kml(self.dem_rsc, tif_filename, title=title, desc=desc, kml_out=kml_out)

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

    def blob_size(self, radius):
        """Finds the radius of a circles/blobs on the LatlonImage in km

        Can also pass array of radii to get multiple distances
        """
        radius = np.array(radius)
        # Use the center of the image as dummy center for circle
        # (really only sigma/radius matters)
        nrows, ncols = self.shape
        midrow, midcol = nrows // 2, ncols // 2
        return self.distance((midrow, midcol), (midrow + radius, midcol + radius))

    def km_to_pixels(self, km):
        """Convert a km distance into number of pixels across"""
        deg_per_pixel = self.x_step  # assume x_step = y_step
        return km_to_pixels(km, deg_per_pixel)


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
    start_lon = rsc_data["x_first"]
    start_lat = rsc_data["y_first"]
    lon_step, lat_step = rsc_data["x_step"], rsc_data["y_step"]
    lat = start_lat + (row - 1) * lat_step
    lon = start_lon + (col - 1) * lon_step

    return lat, lon


def latlon_to_rowcol(lat, lon, rsc_data):
    """Takes latitude, longitude and finds pixel location.

    Inverse of rowcol_to_latlon function

    Args:
        lat (float): latitude
        lon (float): longitude
        rsc_data (dict): data output from load_dem_rsc

    Returns:
        tuple[int, int]: row, col for the pixel

    Example:
        >>> rsc_data = {"x_first": 1.0, "y_first": 2.0, "x_step": 0.2, "y_step": -0.1}
        >>> latlon_to_rowcol(1.4, 1.4, rsc_data)
        (7, 3)
    """
    start_lon = rsc_data["x_first"]
    start_lat = rsc_data["y_first"]
    lon_step, lat_step = rsc_data["x_step"], rsc_data["y_step"]
    row = 1 + (lat - start_lat) / lat_step
    col = 1 + (lon - start_lon) / lon_step
    return int(round(row)), int(round(col))


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
    c = 2 * arctan2(sqrt(a), sqrt(1 - a))
    return R * c


def km_to_deg(km, R=6378):
    """Find the degrees separation for distance km

    Assumes distance along great circle arc

    Args:
        km (float): distance in degrees
        R (float): default 6378, Radius of Earth in km

    Returns:
        float: distance in degrees
    """
    return 360 * km / (2 * np.pi * R)


# Alias to match latlon_to_dist
dist_to_deg = km_to_deg


def km_to_pixels(km, step, R=6378):
    """Convert km distance to pixel size

    Note: This assumes x_step, y_step are equal, and
    calculates the distance in a vertical direction
    (which is more pixels than in the diagonal direction)

    Args:
        km (float): distance in degrees
        step (float): number of degrees per pixel step
        R (float): default 6378, Radius of Earth in km

    Returns:
        float: distance in number of pixels
    """
    degrees = km_to_deg(km, R)
    return degrees / step


dist_to_pixel = km_to_pixels


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


def grid_contains(point, **kwargs):
    """Returns true if point (x, y) or (lon, lat) is within the grid"""
    point_x, point_y = point
    left, right, bot, top = grid_extent(**kwargs)
    return (left < point_x < right) and (bot < point_y < top)


def rot(angle, axis, in_degrees=True):
    """
    Find a 3x3 euler rotation matrix given an angle and axis.

    Rotation matrix used for rotating a vector about a single axis.

    Args:
        angle (float): angle in degrees to rotate
        axis (int): 1, 2 or 3
        in_degrees (bool): specify the angle in degrees. if false, using
            radians for `angle`
    """
    R = np.eye(3)
    if in_degrees:
        angle = np.deg2rad(angle)
    cang = cos(angle)
    sang = sin(angle)
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


def sort_by_lat(latlon_img_list):
    """Sorts a list of LatlonImages by latitude, north to south"""
    return sorted(latlon_img_list, key=lambda img: img.y_first, reverse=True)


def sort_by_lon(latlon_img_list):
    """Sorts a list of LatlonImages by longitude, west to east"""
    return sorted(latlon_img_list, key=lambda img: img.x_first)


def find_img_intersections(image1, image2):
    """Takes two LatlonImages, finds which pixels mark their intersection
    """
    if not intersects(image1.extent, image2.extent):
        return (None, None)

    # TODO: i'm messing up the order between bottom right/ left overlap, e.g.
    im1_lat, im2_lat = sort_by_lat([image1, image2])
    im1_lon, im2_lon = sort_by_lon([image1, image2])

    lat_tup = im1_lat.nearest_pixel(lat=im2_lat.y_first)
    lon_tup = im1_lon.nearest_pixel(lon=im2_lon.x_first)

    # Now combine by ignoring the Nones for each that don't matter
    return (lat_tup[0], lon_tup[1])


def find_total_pixels(image_list):
    """Get the total number of rows and columns for overlapping images
    """
    # TODO: + 1 needed?
    if any(img.dem_rsc is None for img in image_list):
        raise ValueError("All images must have dem_rsc provided")
    elif any(img.x_step != image_list[0].x_step for img in image_list):
        raise ValueError("All images must have same x_step in dem_rsc")
    elif any(img.y_step != image_list[0].y_step for img in image_list):
        raise ValueError("All images must have same y_step in dem_rsc")

    images_sorted = sort_by_lat(image_list)
    im_first = images_sorted[0]
    im_last = images_sorted[-1]
    row_increments = int(round((im_last.last_lat - im_first.y_first) / im_first.y_step))

    images_sorted = sort_by_lon(image_list)
    im_first = images_sorted[0]
    im_last = images_sorted[-1]
    col_increments = int(round((im_last.last_lon - im_first.x_first) / im_first.x_step))
    # Add 1 to count the starting row
    return (1 + row_increments, 1 + col_increments)


def stitch_images(image_list):
    total_rows, total_cols = find_total_pixels(image_list)
    # out = np.zero((total_rows, total_cols))
    # img1 = image_list[0]
    # out[:rows, :cols] = img1
