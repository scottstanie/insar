import unittest
import numpy as np

from insar import utils


class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.im = np.array([[.1, 0.01, 2], [3, 4, 1 + 1j]])

    def test_clip(self):
        self.assertTrue(np.all(utils.clip(self.im) == np.array([[.1, 0.01, 1], [1, 1, 1]])))

    def test_log(self):
        out = np.array([[-20., -40, 6.020599], [9.542425, 12.041199, 3.010299]])
        self.assertTrue(np.allclose(out, utils.log(np.abs(self.im))))
