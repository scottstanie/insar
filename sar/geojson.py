"""
Takes in a geojson polygon, outputs bounds to use for dem download
Used with http://geojson.io to get a quick geojson polygon
Coordinates are (lon, lat)
Output: left, bottom, right, top (floats)
"""

import sys
import json


def geojson_to_bounds(geojson):
    assert geojson['features'][0]['geometry']['type'] == 'Polygon', 'Must use polygon geojson'
    coordinates = geojson['features'][0]['geometry']['coordinates'][0]

    left = min(float(lon) for (lon, lat) in coordinates)
    right = max(float(lon) for (lon, lat) in coordinates)

    top = max(float(lat) for (lon, lat) in coordinates)
    bottom = min(float(lat) for (lon, lat) in coordinates)
    return left, bottom, right, top


def srtm_download():
    # website given links:
    # http://e4ftl01.cr.usgs.gov//MODV6_Dal_D/SRTM/SRTMGL1N.003/2000.02.11/N19W156.SRTMGL1N.num.zip
    # http://e4ftl01.cr.usgs.gov//MODV6_Dal_D/SRTM/SRTMGL1N.003/2000.02.11/N18W156.SRTMGL1N.num.zip

    # Elevation grabbed:
    # https://dds.cr.usgs.gov/srtm/version2_1/Documentation/SRTM_Topo.pdf
    # https://github.com/bopen/elevation/blob/master/elevation/datasource.py
    # curl -s -o spool/N20/N20W155.hgt.gz.temp https://s3.amazonaws.com/elevation-tiles-prod/skadi/N20/N20W155.hgt.gz && mv spool/N20/N20W155.hgt.gz.temp spool/N20/N20W155.hgt.gz
    pass


if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            geojson = json.load(f)
    else:
        geojson = json.load(sys.stdin)

    print(' '.join(str(c) for c in geojson_to_bounds(geojson)))
