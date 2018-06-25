#!/usr/bin/env python
# Make sure the top level is in the pythonpath for insar package import
import argparse
import sys
from os.path import dirname, abspath

import insar.eof
from insar.log import get_log, log_runtime

logger = get_log()


@log_runtime
def main(mission=None, date=None):
    if (mission and not date):
        logger.error("Must specify date if specifying mission.")
        sys.exit(1)
    if not date:
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
