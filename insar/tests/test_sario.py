import unittest
from collections import OrderedDict
from os.path import join, dirname
import numpy as np

from insar import sario
"""Functions todo:
def load_file(filename, ann_info=None):
def load_elevation(filename):
def upsample_dem_rsc(filepath, rate):
def load_real(filename, ann_info):
def is_complex(filename, ann_info):
def parse_complex_data(complex_data, rows, cols):
def combine_real_imag(real_data, imag_data):
def load_complex(filename, ann_info):
def save_array(filename, amplitude_array):
def make_ann_filename(filename):
def parse_ann_file(filename, ext=None, verbose=False):
"""


class TestLoading(unittest.TestCase):
    def setUp(self):
        # self.jsonfile = tempfile.NamedTemporaryFile(mode='w+')
        self.datapath = join(dirname(__file__), 'data')
        self.rsc_path = join(self.datapath, 'elevation.dem.rsc')
        self.ann_path = join(self.datapath, 'test.ann')

    def test_load_dem_rsc(self):
        expected = OrderedDict(
            [('WIDTH', 7201), ('FILE_LENGTH', 3601), ('X_FIRST', -156.0), ('Y_FIRST', 20.0),
             ('X_STEP', 0.000277777777), ('Y_STEP', -0.000277777777), ('X_UNIT', 'degrees'),
             ('Y_UNIT', 'degrees'), ('Z_OFFSET', 0), ('Z_SCALE', 1), ('PROJECTION', 'LL')])
        rsc_data = sario.load_dem_rsc(self.rsc_path)
        self.assertEqual(expected, rsc_data)

    def test_parse_ann_file(self):
        ann_info = sario.parse_ann_file(self.ann_path, ext='.int', verbose=True)
        expected_ann_info = {'rows': 22826, 'cols': 3300}
        self.assertEqual(expected_ann_info, ann_info)

        # Same path and same name as .ann file
        # Different data for the .slc for same ann
        expected_slc_info = {'rows': 273921, 'cols': 9900}
        fake_slc_path = self.ann_path.replace('.ann', '.slc')
        ann_info = sario.parse_ann_file(fake_slc_path)
        self.assertEqual(expected_slc_info, ann_info)

    def test_get_file_rows_cols(self):
        expected_rows_cols = (3601, 7201)
        test_ann_info = {'rows': 3601, 'cols': 7201}
        output = sario._get_file_rows_cols(ann_info=test_ann_info)
        self.assertEqual(expected_rows_cols, output)

        test_rsc_data = {'FILE_LENGTH': 3601, 'WIDTH': 7201}
        output = sario._get_file_rows_cols(rsc_data=test_rsc_data)
        self.assertEqual(expected_rows_cols, output)
        self.assertRaises(ValueError, sario._get_file_rows_cols)
