import unittest
import glob
import os
from os.path import join, dirname

from datetime import date
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from insar import timeseries


class TestInvertSbas(unittest.TestCase):
    def setUp(self):
        # self.jsonfile = tempfile.NamedTemporaryFile(mode='w+')
        self.igram_path = join(dirname(__file__), "data", "sbas_test")
        self.geolist_path = join(self.igram_path, 'geolist')
        self.intlist_path = join(self.igram_path, 'intlist')
        self.actual_time_diffs = np.array([2, 6, 4])

    def tearDown(self):
        for f in glob.glob(join(self.igram_path, "*flat*")):
            os.remove(f)

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

        expected = [
            'data/sbas_test/20180420_20180422.int', 'data/sbas_test/20180420_20180428.int',
            'data/sbas_test/20180422_20180428.int', 'data/sbas_test/20180422_20180502.int',
            'data/sbas_test/20180428_20180502.int'
        ]

        igram_files = timeseries.read_intlist(self.intlist_path, parse=False)
        # Remove all but last part to ignore where we are running this
        igram_files = [os.sep.join(f.split(os.sep)[-3:]) for f in igram_files]
        self.assertEqual(igram_files, expected)

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

    def test_find_time_diffs(self):
        geolist = [date(2018, 4, 20), date(2018, 4, 22), date(2018, 4, 28), date(2018, 5, 2)]
        expected = np.array([2, 6, 4])
        assert_array_equal(expected, timeseries.find_time_diffs(geolist))

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

    def test_invert_sbas_errors(self):
        B = np.arange(12).reshape((4, 3))
        timediffs = np.arange(4)
        dphis = np.arange(4)
        self.assertRaises(ValueError, timeseries.invert_sbas, dphis, timediffs, B)

        # Should work without error
        timediffs = np.arange(3)
        timeseries.invert_sbas(dphis, timediffs, B)

    def test_invert_sbas(self):
        # Fake pixel phases from unwrapped igrams
        actual_phases = np.array([0.0, 2.0, 14.0, 16.0]).reshape((-1, 1))
        actual_velocity_array = np.array([1, 2, .5]).reshape((-1, 1))

        delta_phis = np.array([2, 14, 12, 14, 2]).reshape((-1, 1))

        geolist = timeseries.read_geolist(self.geolist_path)
        intlist = timeseries.read_intlist(self.intlist_path)

        timediffs = timeseries.find_time_diffs(geolist)
        B = timeseries.build_B_matrix(geolist, intlist)
        velocity_array, phases = timeseries.invert_sbas(delta_phis, timediffs, B)

        assert_array_almost_equal(velocity_array, actual_velocity_array)
        assert_array_almost_equal(phases, actual_phases)

        # Now test multiple phase time series as columns
        # stack is column-wise stack by laying vertical rows, then transpose
        actual_phases = np.hstack((actual_phases, 2 * actual_phases))
        actual_velocity_array = np.hstack((actual_velocity_array, 2 * actual_velocity_array))
        delta_phis = np.hstack((delta_phis, 2 * delta_phis))

        velocity_array, phases = timeseries.invert_sbas(delta_phis, timediffs, B)
        assert_array_almost_equal(velocity_array, actual_velocity_array)
        assert_array_almost_equal(phases, actual_phases)

    def test_run_inverison(self):
        # Fake pixel phases from unwrapped igrams
        # See insar/tests/data/sbas_test/write_unw.py for source of these
        actual_phases = np.array([[[0., 0.], [0., 0.], [0., 0.]], [[2., 2.], [4., 4.], [0., 0.]],
                                  [[14., 14.], [28., 28.], [0., 0.]], [[16., 16.], [32., 32.],
                                                                       [0., 0.]]])
        actual_velocity_array = np.array([[[1., 1.], [2., 2.], [0., 0.]],
                                          [[2., 2.], [4., 4.], [0., 0.]], [[0.5, 0.5], [1., 1.],
                                                                           [0., 0.]]])

        # Check that a bad reference throws exception
        self.assertRaises(
            ValueError,
            timeseries.run_inversion,
            self.igram_path,
            reference=(100, 100),
            verbose=True)

        _, phases, deformation, velocity_array, _ = timeseries.run_inversion(
            self.igram_path,
            reference=(2, 0),
            deramp=False  # For this, dont remove the linear ramp
        )

        assert_array_almost_equal(velocity_array, actual_velocity_array)
        assert_array_almost_equal(phases, actual_phases)

    def test_invert_regularize(self):
        B = np.arange(15).reshape((5, 3))
        dphis = np.arange(10).reshape((5, 2))  # Two fake pixels to invert
        timediffs = np.arange(3)
        # Checks for no errors in shape (todo: get good expected output)
        timeseries.invert_sbas(dphis, timediffs, B, alpha=1)

    def test_remove_ramp(self):
        z = np.arange(1, 9, 2).reshape((4, 1)) + np.arange(4)  # (1-4)*(1-7)
        # First test coefficient extimation for z = c + ax + by
        coeffs = timeseries._estimate_ramp(z, order=1)
        assert_array_almost_equal(coeffs, np.array((1, 1, 2)))

        expected_deramped = np.zeros((4, 4))
        assert_array_almost_equal(expected_deramped, timeseries.remove_ramp(z, order=1))

        coeffs = timeseries._estimate_ramp(z, order=2)
        assert_array_almost_equal(coeffs, np.array((1, 1, 2, 0, 0, 0)))

        expected_deramped = np.zeros((4, 4))
        assert_array_almost_equal(expected_deramped, timeseries.remove_ramp(z, order=2))
