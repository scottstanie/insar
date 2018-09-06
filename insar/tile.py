import numpy as np

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


def num_tiles(length, tile_size=0.5):
    """Given an edge length (of a swath), find how many tiles in a direction

    Examples:
    >>> num_tiles(1, tile_size=0.5)
    2
    >>> num_tiles(3.2, tile_size=0.5)
    6
    >>> num_tiles(3.3, tile_size=0.5)
    7
    """
    return np.round(length / tile_size)


def tile_dims(swath_length_width, tile_size=0.5):
    """Given a swath (size_lon, width), find sizes of tiles

    Will try to divide into about 0.5 degree tiles, rounding up or down
    """
    swath_lw_arr = np.array(swath_length_width)
    num_tiles_arr = num_tiles(swath_lw_arr, tile_size)
    return swath_lw_arr / num_tiles_arr
