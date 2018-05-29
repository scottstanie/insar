import unittest
import tempfile
import json

from insar.geojson import _read_json, _parse_coordinates, bounding_box


class TestGeojson(unittest.TestCase):
    def setUp(self):
        self.geojson = {
            "type":
            "Polygon",
            "coordinates": [[[-156.0, 18.7], [-154.6, 18.7], [-154.6, 20.3], [-156.0, 20.3],
                             [-156.0, 18.7]]]
        }
        self.jsonfile = tempfile.NamedTemporaryFile(mode='w+')
        with open(self.jsonfile.name, 'w+') as f:
            json.dump(self.geojson, f)

        self.bad_geojson = {"Point": 0}

    def tearDown(self):
        self.jsonfile.close()

    def test_read_json(self):
        loaded_json = _read_json(self.jsonfile.name)
        self.assertEqual(loaded_json, self.geojson)

    def test_parse_coordinates(self):
        coords = _parse_coordinates(self.geojson)
        self.assertEqual(coords, self.geojson['coordinates'][0])

    def test_bounding_box(self):
        output = bounding_box(self.geojson)
        self.assertEqual(output, (-156.0, 18.7, -154.6, 20.3))

    def test_fail_format(self):
        self.assertRaises(KeyError, bounding_box, self.bad_geojson)
