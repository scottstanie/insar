import unittest

from insar.geojson import geojson_to_bounds


class TestGeojsonBounds(unittest.TestCase):
    def setUp(self):
        self.geojson = {
            "type":
            "Polygon",
            "coordinates": [[[-156.0, 18.7], [-154.6, 18.7], [-154.6, 20.3], [-156.0, 20.3],
                             [-156.0, 18.7]]]
        }

        self.bad_geojson = {"Point": 0}

    def test_geojson_to_bounds(self):
        output = geojson_to_bounds(self.geojson)
        self.assertEqual(output, (-156.0, 18.7, -154.6, 20.3))

    def test_fail_format(self):
        self.assertRaises(KeyError, geojson_to_bounds, self.bad_geojson)
