import unittest
from collections import OrderedDict
import os
from os.path import join, dirname, exists
import shutil
import numpy as np
from numpy.testing import assert_array_almost_equal

from insar import sario


class TestLoading(unittest.TestCase):
    def setUp(self):
        # self.jsonfile = tempfile.NamedTemporaryFile(mode='w+')
        self.datapath = join(dirname(__file__), 'data')
        self.rsc_path = join(self.datapath, 'elevation.dem.rsc')
        self.ann_path = join(self.datapath, 'test.ann')
        self.dem_path = join(self.datapath, 'elevation.dem')
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
            if ext in sario.UAVSAR_POL_DEPENDENT:
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
        # UAVSAR .grd also weird
        grd_complex = 'brazos_090HHHV_CX_01.grd'
        self.assertTrue(sario.is_complex(mlc_complex))
        grd_real = 'brazos_090HHHH_CX_01.grd'
        self.assertFalse(sario.is_complex(grd_real))
        # Also test the downsampled versions
        grd_complex = 'brazos_090HHHV_CX_01_3x3.grd'
        self.assertTrue(sario.is_complex(grd_complex))

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
        cols = 1
        self.assertIsNone(sario._assert_valid_size(data, cols))

        self.assertRaises(AssertionError, sario._assert_valid_size, data, 5 * cols)

    def test_load_file(self):
        geo_path = join(
            self.datapath,
            'S1A_IW_SLC__1SDV_20180420T043026_20180420T043054_021546_025211_81BE.SAFE.small.geo')
        loaded_geo = sario.load_file(geo_path, verbose=True)
        expected_geo = np.array(
            [[-27.189274 - 60.105267j, -41.34938 + 82.05109j], [
                58.716545 + 13.9955j, 68.892 - 42.065178j
            ], [41.361275 - 152.78986j, -65.905945 - 61.246834j]],
            dtype='complex64')
        assert_array_almost_equal(expected_geo, loaded_geo)

        loaded_dem = sario.load_file(self.dem_path, verbose=True)
        expected_dem = np.array([[1413, 1413], [1414, 1414], [1415, 1415]], dtype='<i2')
        assert_array_almost_equal(expected_dem, loaded_dem)

    def test_save_elevation(self):
        loaded_dem = sario.load_file(self.dem_path)
        save_path = self.dem_path.replace('.dem', '_test.dem')

        # Must copy the .dem.rsc as well
        old_dem_rsc = self.dem_path + '.rsc'
        new_dem_rsc = old_dem_rsc.replace('.dem', '_test.dem')
        shutil.copyfile(old_dem_rsc, new_dem_rsc)

        sario.save(save_path, loaded_dem)
        self.assertTrue(exists(save_path))

        reloaded_dem = sario.load_file(save_path)
        assert_array_almost_equal(reloaded_dem, loaded_dem)

        os.remove(new_dem_rsc)
        os.remove(save_path)
        self.assertFalse(exists(save_path))
        self.assertFalse(exists(new_dem_rsc))
