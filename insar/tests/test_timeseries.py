import unittest
from os.path import join, dirname

from datetime import date
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from insar import timeseries


class TestInvertSbas(unittest.TestCase):
    def setUp(self):
        # self.jsonfile = tempfile.NamedTemporaryFile(mode='w+')
        self.datapath = join(dirname(__file__), "data", "sbas_test")
        self.geolist_path = join(self.datapath, 'geolist')
        self.intlist_path = join(self.datapath, 'intlist')
        self.actual_time_diffs = np.array([2, 6, 4])

    def test_time_diff(self):
        geolist = timeseries.read_geolist(self.geolist_path)
        time_diffs = timeseries.find_time_diffs(geolist)
        assert_array_equal(self.actual_time_diffs, time_diffs)

    def test_read_geolist(self):
        geolist = timeseries.read_geolist(self.geolist_path)
        expected = [date(2018, 4, 20), date(2018, 4, 22), date(2018, 4, 28), date(2018, 5, 2)]
        self.assertEqual(geolist, expected)

    def test_read_intlist(self):
        intlist = timeseries.read_intlist(self.intlist_path)
        expected = [
            (date(2018, 4, 20), date(2018, 4, 22)),
            (date(2018, 4, 20), date(2018, 4, 28)),
            (date(2018, 4, 22), date(2018, 4, 28)),
            (date(2018, 4, 22), date(2018, 5, 2)),
            (date(2018, 4, 28), date(2018, 5, 2)),
        ]
        self.assertEqual(intlist, expected)

    def test_build_A_matrix(self):
        geolist = timeseries.read_geolist(self.geolist_path)
        intlist = timeseries.read_intlist(self.intlist_path)
        expected_A = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-1, 1, 0],
            [-1, 0, 1],
            [0, -1, 1],
        ])
        A = timeseries.build_A_matrix(geolist, intlist)
        assert_array_equal(expected_A, A)

    def test_build_B_matrix(self):
        geolist = timeseries.read_geolist(self.geolist_path)
        intlist = timeseries.read_intlist(self.intlist_path)
        expected_B = np.array([
            [2, 0, 0],
            [2, 6, 0],
            [0, 6, 0],
            [0, 6, 4],
            [0, 0, 4],
        ])
        B = timeseries.build_B_matrix(geolist, intlist)
        assert_array_equal(expected_B, B)

    def test_invert_sbas(self):
        # Fake pixel phases from unwrapped igrams
        actual_phases = np.array([2.0, 14.0, 16.0])
        actual_velocity_array = np.array([1, 2, .5])
        delta_phis = np.array([2, 14, 12, 14, 2])

        geolist = timeseries.read_geolist(self.geolist_path)
        intlist = timeseries.read_intlist(self.intlist_path)

        velocity_array, phases = timeseries.invert_sbas(geolist, intlist, delta_phis)

        assert_array_almost_equal(velocity_array, actual_velocity_array)
        assert_array_almost_equal(phases, actual_phases)
