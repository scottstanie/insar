"""
Takes in a geojson polygon, outputs bounds to use for dem download
Used with http://geojson.io to get a quick geojson polygon
Coordinates are (lon, lat)
Output: left, bottom, right, top (floats)
"""

import sys
import json
from insar.log import get_log

logger = get_log()


def geojson_to_bounds(geojson):
    """From a geojson object, compute bounding lon/lats

    Valid geojson types: Polygon (necessary at some depth), Feature, FeatureCollection
    """
    # First, if given a deeper object (e.g. from geojson.io), extract just polygon
    try:
        if geojson.get('type') == 'FeatureCollection':
            geojson = geojson['features'][0]['geometry']
        elif geojson.get('type') == 'Feature':
            geojson = geojson['geometry']
    except KeyError:
        print("Invalid geojson")
        raise

    assert geojson['type'] == 'Polygon', 'Must use polygon geojson'
    # Note: we are assuming a simple polygon with no holes
    coordinates = geojson['coordinates'][0]

    left = min(float(lon) for (lon, lat) in coordinates)
    right = max(float(lon) for (lon, lat) in coordinates)

    top = max(float(lat) for (lon, lat) in coordinates)
    bottom = min(float(lat) for (lon, lat) in coordinates)
    return left, bottom, right, top


if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            geojson = json.load(f)
    else:
        geojson = json.load(sys.stdin)

    print(' '.join(str(c) for c in geojson_to_bounds(geojson)))
