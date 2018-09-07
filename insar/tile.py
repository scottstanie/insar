import numpy as np
import insar.geojson

# import insar.parsers


def total_swath_extent(sentinel_list):
    """Take a list of Sentinels, and find total extent of area covered

    Args:
        list[Sentinel]: Sentinel objects covering one area

    Returns:
        tuple[float]: (min_lon, max_lon, min_lat, max_lat)
    """
    swath_extents = np.array([s.swath_extent for s in sentinel_list])
    min_lon, _, min_lat, _ = np.min(swath_extents, axis=0)
    _, max_lon, _, max_lat = np.max(swath_extents, axis=0)
    return min_lon, max_lon, min_lat, max_lat


def total_swath_width_height(sentinel_list):
    """Width and height in deg across all swath areas in sentinel_list

    Width is longitude extent, height is latitude extent

    Args:
        list[Sentinel]: Sentinel objects covering one area

    Returns:
        tuple[float]: (size_lon, size_lat) in degrees
    """
    left, right, bot, top = total_swath_extent(sentinel_list)
    return right - left, top - bot


def num_tiles(length, tile_size=0.5, overlap=0.1):
    """Given an edge length (of a swath), find number of tiles in a direction

    Illustration: Dividing 1.3 total length into 3 overlap blocks
    |     | |   | |    |
    -----------------------
      .4  .1 .3 .1  .4

    Examples:
    >>> num_tiles(1.3, tile_size=0.5, overlap=0.1)
    3
    >>> num_tiles(np.array([1.3, 1.3]), tile_size=0.5, overlap=0.1)
    array([3, 3])
    >>> num_tiles(np.array([1.4, 1.5, 1.6]), tile_size=0.5, overlap=0.1)
    array([3, 3, 4])
    """
    covered_width = tile_size - overlap
    length_to_divide = length - overlap
    return np.round((length_to_divide) / covered_width).astype(int)


def tile_dims(swath_length_width, tile_size=0.5, overlap=0.1):
    """Given a swath (size_lon, width), find sizes of tiles

    Will try to divide into about 0.5 degree tiles, rounding up or down

    Args:
        swath_length_width (iterable[float, float]): sizes in degrees of swath
        tile_size (float): default 0.5, degrees of tile size to aim for
            Note: This will be adjusted to make even tiles
        overlap (float): default 0.1, overlap size between adjacent blocks

    Examples:
    >>> print(tile_dims(1.3, tile_size=0.5, overlap=0.1))
    0.5
    >>> test_lengths = np.linspace(1, 2, 11)
    >>> print(tile_dims(test_lengths))
    [ 0.55        0.6         0.46666667  0.5         0.53333333  0.56666667
      0.475       0.5         0.525       0.55        0.48      ]

    """
    swath_lw_arr = np.array(swath_length_width)
    num_tiles_arr = num_tiles(swath_lw_arr, tile_size, overlap)

    # how much length is covered if we spread out overlap
    size_unoverlapped = swath_lw_arr + (num_tiles_arr - 1) * overlap
    # Use this to get each tile's size
    return size_unoverlapped / num_tiles_arr


def make_tiles(extent, tile_size=0.5, overlap=0.1):
    """Given a swath extent, divide into overlapping tiles

    Args:
        extent (tuple[float, float, float, float]): Sentinel.swath_extent
            (lon_left,lon_right,lat_bottom,lat_top)
        tile_size (float): default 0.5, degrees of tile size to aim for
            Note: This will be adjusted to make even tiles
        overlap (float): default 0.1, overlap size between adjacent blocks
    Returns:
        list[tuple[float]]: list of tiles in order

    TODO: do I want geojson objects?
    """
    min_lon, max_lon, min_lat, max_lat = extent
    height_width_arr = np.array([max_lon - min_lon, max_lat - min_lat])

    num_lat_tiles, num_lon_tiles = num_tiles(height_width_arr, tile_size=tile_size, overlap=overlap)
    print("Num lat tiles: %s" % num_lat_tiles)
    print("Num lon tiles: %s" % num_lon_tiles)
    lat_tile_size, lon_tile_size = tile_dims(height_width_arr, tile_size=tile_size, overlap=overlap)

    tiles = []
    cur_lat, cur_lon = min_lat, min_lon
    # Iterate bot to top, left to right
    for latidx in range(num_lat_tiles):
        for lonidx in range(num_lon_tiles):
            tiles.append((cur_lat, cur_lon))
            cur_lon = cur_lon + lon_tile_size - overlap
        cur_lon = min_lon  # Reset after each left-to-right
        cur_lat = cur_lat + lat_tile_size - overlap

    # TODO: maybe name these? Do I need a class?
    return tiles, (lat_tile_size, lon_tile_size)


def tile_to_geojson(tile, height, width):
    """Converts a lat/lon Tile to a geojson object

    Tiles have a (lat, lon) start point at the bottom left of the tile,
    along with a height and width
    """
    lat, lon = tile
    corners = insar.geojson.corner_coords(
        bot_corner=(lon, lat),
        dlon=width,
        dlat=height,
    )
    return insar.geojson.corners_to_geojson(corners)
