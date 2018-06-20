import unittest
import json
import tempfile
import shutil
from os.path import join, dirname
import os
import responses

from insar import dem
from insar import geojson

DATAPATH = join(dirname(__file__), 'data')
NETRC_PATH = join(DATAPATH, 'netrc')


class TestNetrc(unittest.TestCase):
    def test_format(self):
        n = dem.Netrc(NETRC_PATH)
        expected = "machine urs.earthdata.nasa.gov\n\tlogin testuser\n\tpassword testpass\n"

        self.assertEqual(n.format(), expected)


class TestTile(unittest.TestCase):
    def setUp(self):
        self.geojson_path = join(DATAPATH, 'hawaii_small.geojson')
        with open(self.geojson_path, 'r') as f:
            self.geojson = json.load(f)
        self.bounds = geojson.bounding_box(self.geojson)

    def test_init(self):
        t = dem.Tile(*self.bounds)
        expected = (-155.49898624420166, 19.741217531292406, -155.497784614563, 19.74218696311137)

        self.assertEqual(expected, t.bounds)


class TestDownload(unittest.TestCase):
    def setUp(self):
        self.geojson_path = join(DATAPATH, 'hawaii_small.geojson')
        with open(self.geojson_path, 'r') as f:
            self.geojson = json.load(f)
        self.bounds = geojson.bounding_box(self.geojson)
        self.test_tile = 'N19W156'
        self.hgt_url = "http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/N19W156.SRTMGL1.hgt.zip"

        sample_hgt_path = join(DATAPATH, 'N19W156.hgt')
        with open(sample_hgt_path, 'rb') as f:
            self.sample_hgt = f.read()

        self.cache_dir = tempfile.mkdtemp()

    def test_init(self):

        d = dem.Downloader([self.test_tile], netrc_file=NETRC_PATH, parallel_ok=False)
        self.assertEqual(d.data_url, "http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11")

    @responses.activate
    def test_s(self):
        responses.add(responses.GET, self.hgt_url, body=self.sample_hgt, status=200)
