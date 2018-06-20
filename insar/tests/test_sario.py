import unittest
from collections import OrderedDict
from os.path import join, dirname
import numpy as np
from numpy.testing import assert_array_almost_equal

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
"""


class TestLoading(unittest.TestCase):
    def setUp(self):
        # self.jsonfile = tempfile.NamedTemporaryFile(mode='w+')
        self.datapath = join(dirname(__file__), 'data')
        self.rsc_path = join(self.datapath, 'elevation.dem.rsc')
        self.ann_path = join(self.datapath, 'test.ann')
        self.rsc_data = OrderedDict(
            [('WIDTH', 2), ('FILE_LENGTH', 3), ('X_FIRST', -155.676388889), ('Y_FIRST',
                                                                             19.5755555567),
             ('X_STEP', 0.000138888888), ('Y_STEP', -0.000138888888), ('X_UNIT', 'degrees'),
             ('Y_UNIT', 'degrees'), ('Z_OFFSET', 0), ('Z_SCALE', 1), ('PROJECTION', 'LL')])

    def test_load_dem_rsc(self):
        rsc_data = sario.load_dem_rsc(self.rsc_path)
        self.assertEqual(self.rsc_data, rsc_data)

    def test_format_dem_rsc(self):
        output = sario.format_dem_rsc(self.rsc_data)
        read_file = open(self.rsc_path).read()
        self.assertEqual(output, read_file)

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

    def test_is_complex(self):
        for ext in sario.COMPLEX_EXTS:
            if ext == '.mlc':
                # Test later, must have a real polarization in name
                continue
            fname = 'test' + ext
            self.assertTrue(sario.is_complex(fname))

        for ext in sario.REAL_EXTS:
            self.assertFalse(sario.is_complex('test' + ext))

        # UAVSAR .mlcs are weird
        mlc_complex = 'brazos_090HHHV_CX_01.mlc'
        self.assertTrue(sario.is_complex(mlc_complex))
        mlc_real = 'brazos_090HHHH_CX_01.mlc'
        self.assertFalse(sario.is_complex(mlc_real))

        self.assertRaises(ValueError, sario.is_complex, 'badext.tif')

    def test_is_combine_real_imag(self):
        real_data = np.array([1, 2, 3], dtype='<f4')
        imag_data = np.array([4, 5, 6], dtype='<f4')
        complex_expected = np.array([1 + 4j, 2 + 5j, 3 + 6j])
        output = sario.combine_real_imag(real_data, imag_data)
        assert_array_almost_equal(complex_expected, output)
        self.assertEqual(output.dtype, np.dtype('complex64'))

    def test_assert_valid_size(self):
        data = np.array([1, 2, 3, 4], '<f4')
        rows = 2
        cols = 1
        self.assertIsNone(sario._assert_valid_size(data, rows, cols))

        self.assertRaises(AssertionError, sario._assert_valid_size, data, rows, 5 * cols)
