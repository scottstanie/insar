import unittest
import os
from os.path import join, dirname, exists
import tempfile
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal

from insar import plotting, timeseries


class TestPlotting(unittest.TestCase):
    def setUp(self):
        self.datapath = join(dirname(__file__), 'data')
        igram_path = join(self.datapath, 'sbas_test')
        self.igram_path = igram_path
        self.stack = timeseries.read_unw_stack(igram_path, 2, 0)

    def test_animate_stack(self):
        try:
            # Commands to turn off interactive for travis tests
            plt.ioff()

            temp_dir = tempfile.mkdtemp()
            animate_file = join(temp_dir, 'save.gif')
            plotting.animate_stack(
                self.stack, display=False, save_title=animate_file, writer='ffmpeg')
            self.assertTrue(os.path.exists(animate_file))
            os.remove(animate_file)

            igram_files = timeseries.read_intlist(join(self.igram_path, 'intlist'), parse=False)
            plotting.animate_stack(
                self.stack, display=False, titles=igram_files, save_title=animate_file)
            self.assertTrue(os.path.exists(animate_file))

        finally:
            shutil.rmtree(temp_dir)
