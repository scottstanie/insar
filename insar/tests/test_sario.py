import unittest

from insar.sario import (get_file_ext, load_file, load_real, load_complex, load_elevation,
                         is_complex, parse_ann_file, make_ann_filename)
"""Functions todo:
def get_file_ext(filename):
def load_file(filename, ann_info=None):
def load_elevation(filename):
def load_dem_rsc(filename):
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
    pass
