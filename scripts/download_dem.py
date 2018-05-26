import json
import sys
from insar.geojson import geojson_to_bounds
from insar.dem import Downloader
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


if __name__ == '__main__':
    main()
