import unittest
# import os
# from os.path import join, dirname, exists
# import tempfile
# import shutil
import numpy as np
# from numpy.testing import assert_array_almost_equal

from insar import latlon


class TestLatlonConversion(unittest.TestCase):
    def setUp(self):
        # self.datapath = join(dirname(__file__), 'data')
        # self.rsc_path = join(self.datapath, 'elevation.dem.rsc')
        # self.dem_path = join(self.datapath, 'elevation.dem')

        self.im_test = np.arange(20).reshape((5, 4))
        self.rsc_info1 = {
            'x_first': -5.0,
            'y_first': 4.0,
            'x_step': 0.5,
            'y_step': -0.5,
            'file_length': 5,
            'width': 4
        }
        self.im1 = latlon.LatlonImage(data=self.im_test, dem_rsc=self.rsc_info1)

        self.im_test2 = np.arange(30).reshape((6, 5))
        self.rsc_info2 = {
            'x_first': -4.0,
            'y_first': 2.5,
            'x_step': 0.5,
            'y_step': -0.5,
            'file_length': 6,
            'width': 5
        }
        self.im2 = latlon.LatlonImage(data=self.im_test2, dem_rsc=self.rsc_info2)
        self.image_list = [self.im1, self.im2]

    def test_latlon_rowcol(self):
        rsc_data = {"x_first": 1.0, "y_first": 2.0, "x_step": 0.2, "y_step": -0.1}

        row, col = (7, 3)
        lat, lon = (1.4, 1.4)

        self.assertEqual((row, col), latlon.latlon_to_rowcol(lat, lon, rsc_data))
        self.assertEqual(lat, lon, latlon.rowcol_to_latlon(lat, lon, rsc_data))

    def test_find_total_pixels(self):
        self.assertEqual((9, 7), latlon.find_total_pixels(self.image_list))

    def test_find_img_intersection(self):
        expected = (3, 2)
        self.assertEqual(expected, latlon.find_img_intersections(*self.image_list))

    def test_stitch(self):
        out = np.zeros(latlon.find_total_pixels(self.image_list))
        start_row, start_col = latlon.find_img_intersections(*self.image_list)
        out[start_row:, start_col:] = self.im2
