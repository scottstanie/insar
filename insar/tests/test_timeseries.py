import unittest
import tempfile
from os.path import join, dirname

from datetime import date
import numpy as np
from insar import timeseries


class TestInvertSbas(unittest.TestCase):
    def setUp(self):
        # self.jsonfile = tempfile.NamedTemporaryFile(mode='w+')
        self.datapath = join(dirname(__file__), "data", "sbas_test")
        self.geolist_path = join(self.datapath, 'geolist')
        self.intlist_path = join(self.datapath, 'intlist')

    def test_read_geolist(self):
        geolist = timeseries.read_geolist(self.geolist_path)
        expected = [date(2018, 4, 20), date(2018, 4, 24), date(2018, 4, 28), date(2018, 5, 2)]
        self.assertEqual(geolist, expected)

    def test_read_intlist(self):
        intlist = timeseries.read_intlist(self.intlist_path)
        expected = [
            (date(2018, 4, 20), date(2018, 4, 24)),
            (date(2018, 4, 20), date(2018, 4, 28)),
            (date(2018, 4, 20), date(2018, 5, 2)),
            (date(2018, 4, 24), date(2018, 4, 28)),
            (date(2018, 4, 24), date(2018, 5, 2)),
            (date(2018, 4, 28), date(2018, 5, 2)),
        ]
        self.assertEqual(intlist, expected)

    def test_build_A_matrix(self):
        geolist = timeseries.read_geolist(self.geolist_path)
        intlist = timeseries.read_intlist(self.intlist_path)
        expected_A = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 1, 0],
            [-1, 0, 1],
            [0, -1, 1],
        ])
        A = timeseries.build_A_matrix(geolist, intlist)
        np.testing.assert_array_equal(expected_A, A)

    def test_build_B_matrix(self):
        geolist = timeseries.read_geolist(self.geolist_path)
        intlist = timeseries.read_intlist(self.intlist_path)
        expected_A = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 1, 0],
            [-1, 0, 1],
            [0, -1, 1],
        ])
        A = timeseries.build_A_matrix(geolist, intlist)
        np.testing.assert_array_equal(expected_A, A)
        pass

    def test_invert_sbas(self):
        pass
