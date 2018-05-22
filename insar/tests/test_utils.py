import unittest
import numpy as np

from insar.utils import (
    downsample_im,
    upsample_dem,
    clip,
    log,
    split_array_into_blocks,
    split_and_save,
    combine_cor_amp,
)
"""Functions todo:
def downsample_im(image, rate=10):
def upsample_dem(dem_img, rate=3):
def split_array_into_blocks(data):
def split_and_save(filename):
def combine_cor_amp(corfilename, save=True):

"""


class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.im = np.array([[.1, 0, 2], [3, 4, 1 + 1j]])

    def test_clip(self):
        self.assertTrue(np.all(clip(self.im) == np.array([[.1, 0, 1], [1, 1, 1]])))

    def test_log(self):
        out = np.array([[-20., -np.inf, 6.020599], [9.542425, 12.041199, 3.010299]])
        self.assertTrue(np.allclose(out, log(np.abs(self.im))))
