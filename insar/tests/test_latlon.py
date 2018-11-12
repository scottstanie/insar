import unittest
# import os
# from os.path import join, dirname, exists
# import tempfile
# import shutil
# import numpy as np
# from numpy.testing import assert_array_almost_equal

from insar import latlon


class TestLatlonConversion(unittest.TestCase):
    def setUp(self):
        # self.datapath = join(dirname(__file__), 'data')
        # self.rsc_path = join(self.datapath, 'elevation.dem.rsc')
        # self.dem_path = join(self.datapath, 'elevation.dem')
        self.rsc_data = {"x_first": 1.0, "y_first": 2.0, "x_step": 0.2, "y_step": -0.1}

    def test_latlon_rowcol(self):
        row, col = (7, 3)
        lat, lon = (1.4, 1.4)

        self.assertEqual((row, col), latlon.latlon_to_rowcol(lat, lon, self.rsc_data))
        self.assertEqual(lat, lon, latlon.rowcol_to_latlon(lat, lon, self.rsc_data))
