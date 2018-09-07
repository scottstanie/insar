"""Tools for dividing large area into tiles to process
"""
import numpy as np
import insar.geojson
from insar import parsers, latlon


class Tile(object):
    """Class holding one lat/lon area to process

    Note: height and width depend on the total area size, so are optional

    Attributes:
        lat (float): bottom (southern) latitude
        lon (float): leftmost (western) longitude
        tilename (str): Representation of lat/lon to name
            the directory holding processing results.
            Example: N30.1W104.1
    """

    def __init__(self, lat, lon, height=None, width=None):
        self.lat = lat
        self.lon = lon
        self.height = height
        self.width = width
        self.tilename = self._form_tilename(lat, lon)

    def __str__(self):
        return "<Tile %s>" % self.tilename

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
        return insar.latlon.intersects(self.extent, extent)


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

    def make_tiles(self):
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
        for latidx in range(num_lat_tiles):
            for lonidx in range(num_lon_tiles):
                t = Tile(cur_lat, cur_lon, height=lat_tile_size, width=lon_tile_size)
                tiles_out.append(t)
                cur_lon = cur_lon + lon_tile_size - self.overlap
            cur_lon = min_lon  # Reset after each left-to-right
            cur_lat = cur_lat + lat_tile_size - self.overlap

        return tiles_out
