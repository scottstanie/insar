import json
import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from insar.geojson import geojson_to_bounds
from insar.dem import Downloader, Stitcher
from insar.log import get_log, log_runtime

logger = get_log()


@log_runtime
def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            geojson = json.load(f)
    else:
        geojson = json.load(sys.stdin)

    bounds = geojson_to_bounds(geojson)
    logger.info("Bounds: %s", " ".join(str(b) for b in bounds))

    d = Downloader(*bounds)
    d.download_all()

    s = Stitcher(d.srtm1_tile_names())
    full_dem = s.load_and_stitch()
    filename = 'elevation.dem'
    full_dem.tofile(filename)

    # Now create corresponding rsc file
    rsc_filename = filename + '.rsc'
    logger.info("Writing to %s", rsc_filename)
    with open(rsc_filename, 'w') as f:
        rsc_string = s.format_dem_rsc(s.create_dem_rsc())
        f.write(rsc_string)


if __name__ == '__main__':
    main()
