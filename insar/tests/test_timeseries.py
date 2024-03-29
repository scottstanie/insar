import unittest
import glob
import os
from os.path import join, dirname

from datetime import date
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from insar import timeseries
from apertools import sario


class TestInvertSbas(unittest.TestCase):
    def setUp(self):
        # self.jsonfile = tempfile.NamedTemporaryFile(mode='w+')
        self.igram_path = join(dirname(__file__), "data", "sbas_test")
        self.slclist_path = join(self.igram_path, "slclist")
        self.ifglist_path = join(self.igram_path, "ifglist")
        self.actual_time_diffs = np.array([2, 6, 4])

    def tearDown(self):
        for f in glob.glob(join(self.igram_path, "*flat*")):
            os.remove(f)

    def test_time_diff(self):
        slclist = sario.find_geos(self.slclist_path)
        time_diffs = timeseries.find_time_diffs(slclist)
        assert_array_equal(self.actual_time_diffs, time_diffs)

    def test_read_slclist(self):
        slclist = sario.find_geos(self.slclist_path)
        expected = [
            date(2018, 4, 20),
            date(2018, 4, 22),
            date(2018, 4, 28),
            date(2018, 5, 2),
        ]
        self.assertEqual(slclist, expected)

    def test_read_ifglist(self):
        ifglist = sario.find_igrams(self.ifglist_path)
        expected = [
            (date(2018, 4, 20), date(2018, 4, 22)),
            (date(2018, 4, 20), date(2018, 4, 28)),
            (date(2018, 4, 22), date(2018, 4, 28)),
            (date(2018, 4, 22), date(2018, 5, 2)),
            (date(2018, 4, 28), date(2018, 5, 2)),
        ]
        self.assertEqual(ifglist, expected)

        expected = [
            "data/sbas_test/20180420_20180422.int",
            "data/sbas_test/20180420_20180428.int",
            "data/sbas_test/20180422_20180428.int",
            "data/sbas_test/20180422_20180502.int",
            "data/sbas_test/20180428_20180502.int",
        ]

        igram_files = sario.find_igrams(self.ifglist_path, parse=False)
        # Remove all but last part to ignore where we are running this
        igram_files = [os.sep.join(f.split(os.sep)[-3:]) for f in igram_files]
        self.assertEqual(igram_files, expected)

    def test_build_A_matrix(self):
        slclist = sario.find_geos(self.slclist_path)
        ifglist = sario.find_igrams(self.ifglist_path)
        expected_A = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [-1, 1, 0],
                [-1, 0, 1],
                [0, -1, 1],
            ]
        )
        A = timeseries.build_A_matrix(slclist, ifglist)
        assert_array_equal(expected_A, A)

    def test_find_time_diffs(self):
        slclist = [
            date(2018, 4, 20),
            date(2018, 4, 22),
            date(2018, 4, 28),
            date(2018, 5, 2),
        ]
        expected = np.array([2, 6, 4])
        assert_array_equal(expected, timeseries.find_time_diffs(slclist))

    def test_build_B_matrix(self):
        slclist = sario.find_geos(self.slclist_path)
        ifglist = sario.find_igrams(self.ifglist_path)
        expected_B = np.array(
            [
                [2, 0, 0],
                [2, 6, 0],
                [0, 6, 0],
                [0, 6, 4],
                [0, 0, 4],
            ]
        )
        B = timeseries.build_B_matrix(slclist, ifglist)
        assert_array_equal(expected_B, B)

    def test_invert_sbas_errors(self):
        B = np.arange(12).reshape((4, 3))
        dphis = np.arange(4)
        timediffs = np.arange(4)
        vs = timeseries.invert_sbas(dphis, B)
        self.assertRaises(ValueError, timeseries.integrate_velocities, vs, timediffs)

        # Should work without error
        timediffs = np.arange(3)
        vs = timeseries.invert_sbas(dphis, B)
        timeseries.integrate_velocities(vs, timediffs)

    def test_invert_sbas(self):
        # Fake pixel phases from unwrapped igrams
        actual_phases = np.array([0.0, 2.0, 14.0, 16.0]).reshape((-1, 1))
        actual_velocity_array = np.array([1, 2, 0.5]).reshape((-1, 1))

        delta_phis = np.array([2, 14, 12, 14, 2]).reshape((-1, 1))

        slclist = sario.find_geos(self.slclist_path)
        ifglist = sario.find_igrams(self.ifglist_path)

        timediffs = timeseries.find_time_diffs(slclist)
        B = timeseries.build_B_matrix(slclist, ifglist)

        velocity_array = timeseries.invert_sbas(delta_phis, B)
        assert_array_almost_equal(velocity_array, actual_velocity_array)

        phases = timeseries.integrate_velocities(velocity_array, timediffs)
        assert_array_almost_equal(phases, actual_phases)

        # Now test multiple phase time series as columns
        # stack is column-wise stack by laying vertical rows, then transpose
        actual_phases = np.hstack((actual_phases, 2 * actual_phases))
        actual_velocity_array = np.hstack(
            (actual_velocity_array, 2 * actual_velocity_array)
        )
        delta_phis = np.hstack((delta_phis, 2 * delta_phis))

        velocity_array = timeseries.invert_sbas(delta_phis, B)
        assert_array_almost_equal(velocity_array, actual_velocity_array)

        phases = timeseries.integrate_velocities(velocity_array, timediffs)
        assert_array_almost_equal(phases, actual_phases)

    def test_run_inverison(self):
        # Fake pixel phases from unwrapped igrams
        # See insar/tests/data/sbas_test/write_unw.py for source of these
        actual_phases = np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[2.0, 2.0], [4.0, 4.0], [0.0, 0.0]],
                [[14.0, 14.0], [28.0, 28.0], [0.0, 0.0]],
                [[16.0, 16.0], [32.0, 32.0], [0.0, 0.0]],
            ]
        )  # yapf: disable

        # Check that a bad reference throws exception
        self.assertRaises(
            ValueError,
            timeseries.run_inversion,
            self.igram_path,
            reference=(100, 100),
            masking=False,
            verbose=True,
        )

        _, phases, deformation = timeseries.run_inversion(
            self.igram_path,
            reference=(2, 0),
            deramp=False,  # For this, dont remove the linear ramp
            masking=False,
        )

        assert_array_almost_equal(phases, actual_phases)

    def test_invert_regularize(self):
        B = np.arange(15).reshape((5, 3))
        dphis = np.arange(10).reshape((5, 2))  # Two fake pixels to invert
        # Checks for no errors in shape (todo: get good expected output)
        vs = timeseries.invert_sbas(dphis, B, alpha=1)
        timediffs = np.arange(3)
        timeseries.integrate_velocities(vs, timediffs)

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
