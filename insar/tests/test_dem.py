import unittest
import tempfile
import shutil
import os
import responses

from insar import dem
from insar import geojson
"""Functions todo:
    todo: mock out responses from url request
          upsample_dem_rsc(filepath, rate):
"""


class TestStartLatLon(unittest.TestCase):
    def setUp(self):
        self.datapath = join(dirname(__file__), 'data')
        self.geojson = 
        self.southeast = 'S19E156.hgt'
        self.bad_tilename = 'somethingelse.hgt'

    @responses.activate
    def test_s(self):
        responses.add(responses.GET, self.download_url, body=self.sample_hgt, status=200)
