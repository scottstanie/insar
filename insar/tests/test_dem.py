import unittest

from insar.dem import upsample_dem_rsc
"""Functions todo:
    todo: mock out responses from url request
          upsample_dem_rsc(filepath, rate):
"""


class TestStartLatLon(unittest.TestCase):
    def setUp(self):
        self.northwest = 'N19W156.hgt'
        self.southeast = 'S19E156.hgt'
        self.bad_tilename = 'somethingelse.hgt'
