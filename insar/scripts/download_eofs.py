#!/usr/bin/env python
# Make sure the top level is in the pythonpath for insar package import
import argparse
import sys
from os.path import dirname, abspath

import insar.eof
from insar.log import get_log, log_runtime

logger = get_log()


@log_runtime
def main():
    parser = argparse.ArgumentParser(
        description='Download EOFs for specific date, or for Sentinel files in --path. '
        'With arguments, searches current directory for Sentinel 1 products')
    parser.add_argument("--date", "-r", help="Validity date for EOF to download")
    parser.add_argument(
        "--path", "-p", help="Which directory to look for Sentinel products.", default=".")
    parser.add_argument(
        "--mission",
        "-m",
        choices=("S1A", "S1B"),
        help="Specify mission satellite to download (None downloads both S1A and S1B)")
    args = parser.parse_args()

    if (args.mission and not args.date):
        logger.error("Must specify date if specifying mission.")
        sys.exit(1)
    if not args.date:
        # No command line args given: search current directory
        orbit_dates, missions = insar.eof.find_sentinel_products(args.path)
        if not orbit_dates:
            logger.info("No Sentinel products found in current directory. Exiting")
            sys.exit(0)
    if args.date:
        orbit_dates = [args.date]
        missions = list(args.mission) if args.mission else []

    insar.eof.download_eofs(orbit_dates, missions=missions)


if __name__ == '__main__':
    main()
