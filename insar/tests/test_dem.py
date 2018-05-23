import unittest

from insar.dem import (start_lon_lat, upsample_dem, upsample_dem_rsc, mosaic_dem)
"""Functions todo:
def upsample_dem(dem_img, rate=3):
def mosaic_dem(d1, d2):
def upsample_dem_rsc(filepath, rate):
"""


class TestStartLatLon(unittest.TestCase):
    def setUp(self):
        self.northwest = 'N19W156.hgt'
        self.southeast = 'S19E156.hgt'
        self.bad_tilename = 'somethingelse.hgt'

    def test_northwest(self):
        lon, lat = start_lon_lat(self.northwest)
        self.assertEqual((lon, lat), (-156.0, 20.0))

    def test_southeast(self):
        lon, lat = start_lon_lat(self.southeast)
        self.assertEqual((lon, lat), (156.0, 18.0))

    def test_fail(self):
        self.assertRaises(ValueError, start_lon_lat, self.bad_tilename)
