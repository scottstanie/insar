"""Tools for dividing large area into tiles to process
"""
from __future__ import division
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import insar.geojson
from insar import latlon, parsers, sario, plotting
from insar.log import get_log

logger = get_log()


class Tile(object):
    """Class holding one lat/lon area to process

    Note: height and width depend on the total area size, so are optional

    Attributes:
        lat (float): bottom (southern) latitude
        lon (float): leftmost (western) longitude
        name (str): Representation of lat/lon to name
            the directory holding processing results.
            Example: N30.1W104.1
    """

    def __init__(self, lat, lon, height=None, width=None):
        self.lat = lat
        self.lon = lon
        self.height = height
        self.width = width
        self.name = self._form_tilename(lat, lon)

    def __str__(self):
        return "<Tile %s>" % self.name

    def __repr__(self):
        return str(self)

    def _form_tilename(self, lat, lon):
        hemi_ns = 'N' if lat >= 0 else 'S'
        hemi_ew = 'E' if lon >= 0 else 'W'
        # Format to one decimal
        lat_str = '{}{:02.1f}'.format(hemi_ns, abs(lat))
        lon_str = '{}{:03.1f}'.format(hemi_ew, abs(lon))
        latlon_str = '{lat_str}{lon_str}'.format(lat_str=lat_str, lon_str=lon_str)
        # Use underscores in the name instead of decimal
        # return latlon_str.replace('.', '_')
        return latlon_str

    @staticmethod
    def parse_tilename(tilename):
        """Parses the lat/lon info from a tilename into (lon, lat)

        Returns:
            tuple[float, float]: (longitude, latitude)
                lon is negative in west, lat negative in south

        Examples:
            >>> Tile.parse_tilename('./N30.8W103.7')
            (-103.7, 30.8)
            >>> Tile.parse_tilename('/home/scott/N30.8W103.7/')
            (-103.7, 30.8)

        """
        name_regex = r'([NS])([0-9\.]{1,5})([EW])([0-9\.]{1,5})'
        # Now strip path dependencies and parse
        _, name_clean = os.path.split(os.path.abspath(tilename))
        re_match = re.match(name_regex, name_clean.strip('./'))
        if re_match is None:
            raise ValueError("%s is not a valid tile name" % tilename)
        hemi_ns, lat_str, hemi_ew, lon_str = re_match.groups()
        lon_out = float(lon_str) if hemi_ew == 'E' else -1 * float(lon_str)
        lat_out = float(lat_str) if hemi_ns == 'N' else -1 * float(lat_str)
        return lon_out, lat_out

    def to_geojson(self):
        """Converts a lat/lon Tile to a geojson object

        Tiles have a (lat, lon) start point at the bottom left of the tile,
        along with a height and width

        Args:
            height (float): height of the tile (size in lat)
            width (float): width of the tile (lon size)
        """
        corners = insar.geojson.corner_coords(
            bot_corner=(self.lon, self.lat),
            dlon=self.width,
            dlat=self.height,
        )
        return insar.geojson.corners_to_geojson(corners)

    @property
    def geojson(self):
        return self.to_geojson()

    @property
    def extent(self):
        """Boundaries of tile: (lon_left,lon_right,lat_bottom,lat_top)"""
        return (self.lon, self.lon + self.width, self.lat, self.lat + self.height)

    def overlaps_with(self, sentinel=None, extent=None):
        """Returns True if Tile's area overlaps the sentinel extent
        """
        extent = sentinel.extent if sentinel else extent
        return latlon.intersects(self.extent, extent)


class TileGrid(object):
    """
    Class handling making tiles out of list of Sentinel objects

    Attributes:
        sentinel_list (list[Sentinel]): Sentinel objects covering one area
        tile_size (float): default 0.5, degrees of tile size to aim for
            Note: This will be adjusted to make even tiles
        overlap (float): default 0.1, overlap size between adjacent blocks
    """

    def __init__(self, sentinel_list, tile_size=0.5, overlap=0.1):
        self.sentinel_list = sentinel_list
        if not sentinel_list:
            raise ValueError("sentinel_list must be non-empty")
        self.tile_size = tile_size
        self.overlap = overlap

    @property
    def extent(self):
        return self.total_swath_extent()

    def total_swath_extent(self):
        """Take a list of Sentinels, and find total extent of area covered

        Returns:
            tuple[float]: (min_lon, max_lon, min_lat, max_lat)
        """
        swath_extents = np.array([s.swath_extent for s in self.sentinel_list])
        min_lon, _, min_lat, _ = np.min(swath_extents, axis=0)
        _, max_lon, _, max_lat = np.max(swath_extents, axis=0)
        return min_lon, max_lon, min_lat, max_lat

    @property
    def total_width_height(self):
        """Width and height in deg across all swath areas in sentinel_list

        Width is longitude extent, height is latitude extent

        Returns:
            ndarray[float, float]: (size_lon, size_lat) in degrees
        """
        left, right, bot, top = self.extent
        return np.array([right - left, top - bot])

    @staticmethod
    def calc_num_tiles(length, tile_size, overlap):
        """Given an edge length (of a swath), find number of tiles in a direction

        Illustration: With tile_size=0.5 and overlap=0.1,
        dividing 1.3 total length into 3 overlap blocks:

        |     | |   | |    |
        -----------------------
          .4  .1 .3 .1  .4

        Examples:
        >>> TileGrid.calc_num_tiles(1.3, tile_size=0.5, overlap=0.1)
        3
        >>> TileGrid.calc_num_tiles(np.array([1.3, 1.3]), tile_size=0.5, overlap=0.1)
        array([3, 3])
        >>> TileGrid.calc_num_tiles(np.array([1.4, 1.5, 1.6]), tile_size=0.5, overlap=0.1)
        array([3, 3, 4])
        """
        covered_width = tile_size - overlap
        length_to_divide = length - overlap
        return np.round((length_to_divide) / covered_width).astype(int)

    @property
    def num_tiles(self):
        return self.calc_num_tiles(self.total_width_height, self.tile_size, self.overlap)

    @staticmethod
    def calc_tile_dims(length_arr, tile_size, overlap):
        """Given a swath (size_lon, width), find sizes of tiles

        If tile_size = 0.5 and overlap = 0.1, the function will
        try to divide into about 0.5 degree tiles, rounding up or down

        Args:
            length_arr (iterable[float, float]): sizes in degrees of swath
            tile_size (float): degrees of tile size to aim for
                Note: This will be adjusted to make even tiles
            overlap (float): overlap size between adjacent blocks

        Examples:
        >>> print(TileGrid.calc_tile_dims(1.3, 0.5, 0.1))
        0.5
        >>> test_lengths = np.linspace(1, 2, 11)
        >>> print(TileGrid.calc_tile_dims(test_lengths, 0.5, 0.1))
        [ 0.55        0.6         0.46666667  0.5         0.53333333  0.56666667
          0.475       0.5         0.525       0.55        0.48      ]
        """
        num_tiles_arr = TileGrid.calc_num_tiles(length_arr, tile_size, overlap)
        # how much length is covered if we spread out overlap
        size_unoverlapped = length_arr + (num_tiles_arr - 1) * overlap
        # Use this to get each tile's size
        return size_unoverlapped / num_tiles_arr

    @property
    def tile_dims(self):
        return self.calc_tile_dims(self.total_width_height, self.tile_size, self.overlap)

    def make_tiles(self, verbose=False):
        """Divide the extent from the sentinel_list into Tiles

        Returns:
            list[Tile]: list of tiles in ordered bot to top, left to right
        """
        tiles_out = []

        min_lon, _, min_lat, _ = self.extent
        cur_lat, cur_lon = min_lat, min_lon

        # Iterate bot to top, left to right
        num_lat_tiles, num_lon_tiles = self.num_tiles
        lon_tile_size, lat_tile_size = self.tile_dims
        if verbose:
            self._log_tile_info()
        for latidx in range(num_lat_tiles):
            for lonidx in range(num_lon_tiles):
                t = Tile(cur_lat, cur_lon, height=lat_tile_size, width=lon_tile_size)
                tiles_out.append(t)
                cur_lon = cur_lon + lon_tile_size - self.overlap
            cur_lon = min_lon  # Reset after each left-to-right
            cur_lat = cur_lat + lat_tile_size - self.overlap

        return tiles_out

    def _log_tile_info(self):
        logger.info("Tiles in (lon, lat) directions: (%d, %d)", *self.num_tiles)
        logger.info(
            "Dimensions of tile in (lon, lat) directions: ({:.2f}, {:.2f})".format(*self.tile_dims))
        logger.info("Total number of tiles: %d", np.prod(self.num_tiles))
        logger.info(
            "Total area covered in (lon, lat): ({:.2f}, {:.2f})".format(*self.total_width_height))
        logger.info("Total extent covered: {:.2f} {:.2f} {:.2f} {:.2f} ".format(*self.extent))


def find_sentinels(data_path, path_num=None, ending='.SAFE'):
    """Find sentinel products in data_path

    Optionally filter by ending (.SAFE for directory, .zip for zipped product)
    or by a path_number (relative orbit)

    Args:
        data_path (str): location to look for products
        path_num (int): path number/ relative orbit to filter products by
        ending (str): ending of filename to looks for. Default = '.SAFE', unzipped dirs
    Returns:
        list[Sentinel]: list of the parsed Sentinel instances
    """
    search_results = glob.glob(os.path.join(data_path, "*"))
    sents = [parsers.Sentinel(f) for f in search_results if f.endswith(ending)]
    if path_num:
        sents = [s for s in sents if s.path == path_num]
    return sents


def create_tiles(data_path=None,
                 path_num=None,
                 sentinel_list=None,
                 tile_size=0.5,
                 overlap=0.1,
                 verbose=False):
    """Find tiles over a sentinel area, form the tiles/geojsons

    Args:
        data_path (str): path to .zips of .SAFEs of sentinel products
        path_num (int): Optional- A relative orbit/ path number to filter products
        sentinel_list (list[Sentinel]): Sentinel objects covering one area,
            alternative to data_path + path_num args
        tile_size (float): default 0.5, degrees of tile size to aim for
            Note: This will be adjusted to make even tiles
        overlap (float): default 0.1, overlap size between adjacent blocks
    """
    # TODO: figure out how to find/ symlink .EOFS if they are there
    if not sentinel_list:
        sentinel_list = find_sentinels(data_path, path_num)
    tile_grid = TileGrid(sentinel_list, tile_size=tile_size, overlap=overlap)

    return tile_grid.make_tiles(verbose=verbose)


def plot_tiles(dirlist, gps_station_list=None):
    """Takes a list of tile directories and plots the deformation result

    Args:
        dirlist (str): file containing the list of tile directories

    Examples dirlist:
        $ cat dirlist.txt
        ./N31.4W103.7
        ./N30.8W103.7
    """

    def count_row_col(lon_lat_list):
        """Returns num_rows, num_cols in the lon_lat_list"""
        # Count uniqe lons, lats by unpacking
        lons, lats = zip(*lon_lat_list)
        return len(set(lats)), len(set(lons))

    def _read_dirname(dirname):
        igram_dir = os.path.join(dirname, 'igrams')
        defo_file = os.path.join(igram_dir, 'deformation.npy')
        img_data_file = os.path.join(igram_dir, 'dem.rsc')

        print('reading in %s' % defo_file)
        defo_img = np.mean(np.load(defo_file)[-3:], axis=0)
        img_data = sario.load(img_data_file)
        return defo_img, img_data

    with open(dirlist) as f:
        directory_names = f.read().splitlines()

    lon_lat_list = [Tile.parse_tilename(d) for d in directory_names]

    num_rows, num_cols = count_row_col(lon_lat_list)

    # Sort these in a lat/lon grid from top left to bottom right, row order
    sorted_dirs = sorted(
        zip(directory_names, lon_lat_list), key=lambda tup: (-tup[1][1], tup[1][0]))

    defo_img_list = []
    img_data_list = []
    for dirname, lon_lat_tup in sorted_dirs:
        defo_img, img_data = _read_dirname(dirname)
        defo_img_list.append(defo_img)
        img_data_list.append(img_data)

    vmax = np.nanmax(np.stack(defo_img_list, axis=0))
    vmin = np.nanmin(np.stack(defo_img_list, axis=0))
    print('vmin, vmax', vmin, vmax)
    cmap_name = 'seismic'

    fig, axes = plt.subplots(num_rows, num_cols)
    for idx, (dirname, lon_lat_tup) in enumerate(sorted_dirs):
        cur_ax = axes.flat[idx]
        defo_img = defo_img_list[idx]
        # plotting.plot_image_shifted(
        #     defo_img,
        #     fig=fig,
        #     ax=cur_ax,
        #     img_data=img_data,
        #     title=dirname,
        #     perform_shift=True,
        # )

        cur_data = img_data_list[idx]
        extent = latlon.grid_extent(**cur_data)
        points = []
        legends = []
        for name, lon, lat in stations_with_data:
            if latlon.grid_contains((lon, lat), **cur_data):
                points.append((lon, lat))
                legends.append(name)

        shifted_cmap = plotting.make_shifted_cmap(
            cmap_name=cmap_name,
            vmax=vmax,
            vmin=vmin,
        )
        im = cur_ax.imshow(
            defo_img,
            cmap=shifted_cmap,
            extent=extent,
        )
        cur_ax.set_title(dirname)

        cbar = fig.colorbar(im, ax=cur_ax, boundaries=np.arange(vmin, vmax + 1).astype(int))
        cbar.set_clim(vmin, vmax)
        for lon, lat in points:
            cur_ax.plot(lon, lat, 'X', markersize=15)
        cur_ax.legend(legends)

    return fig, axes, defo_img_list, img_data_list


def find_stations_with_data(gps_dir=None):
    # Now also get gps station list
    if not gps_dir:
        gps_dir = '/data1/scott/pecos/gps_station_data'

    all_station_data = read_station_dict(os.path.join(gps_dir, 'texas_stations.csv'))
    station_data_list = find_station_data_files(gps_dir)
    stations_with_data = [tup for tup in all_station_data if tup[0] in station_data_list]
    return stations_with_data


def find_station_data_files(gps_dir):
    station_files = glob.glob(os.path.join(gps_dir, '*.tenv3'))
    station_list = []
    for filename in station_files:
        _, name = os.path.split(filename)
        station_list.append(name.split('.')[0])
    return station_list


def read_station_dict(filename):
    """Reads in GPS station data"""
    with open(filename) as f:
        station_strings = [row for row in f.read().splitlines()]

    all_station_data = []
    for row in station_strings:
        name, lat, lon, _ = row.split(',')  # Ignore altitude
        all_station_data.append((name, float(lon), float(lat)))
    return all_station_data
