import unittest
import json
import tempfile
import shutil
from os.path import join, dirname
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
        self.geojson_path = join(self.datapath, 'hawaii_small.geojson')
        with open(self.geojson_path, 'r') as f:
            self.geojson = json.load(f)
        self.bounds = geojson.bounding_box(self.geojson)
        self.download_url = "http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/N19W156.SRTMGL1.hgt.zip"

    @responses.activate
    def test_s(self):
        responses.add(responses.GET, self.download_url, body=self.sample_hgt, status=200)


class TestTile(unittest.TestCase):
    def setUp(self):
        self.datapath = join(dirname(__file__), 'data')
        self.geojson_path = join(self.datapath, 'hawaii_small.geojson')
        with open(self.geojson_path, 'r') as f:
            self.geojson = json.load(f)
        self.bounds = geojson.bounding_box(self.geojson)

    def test_init(self):
        t = dem.Tile(*self.bounds)
        expected = (-155.63232421875, 19.4303341116379, -155.3192138671875, 19.730512997022263)
        self.assertEqual(expected, t.bounds)
