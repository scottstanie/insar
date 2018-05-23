#!/usr/bin/env python
# Make sure the top level is in the pythonpath for insar package import
import sys
from os.path import dirname, abspath
from datetime import date
sys.path.insert(0, dirname(dirname(abspath(__file__))))
import insar.eof
from insar.parsers import Sentinel
import glob


def main():
    orbit_dates = []
    for filename in glob.glob("./*.zip"):
        start_date = Sentinel(filename).start_stop_time()[0]
        orbit_dates.append(start_date)

    print("Downloading precise orbits for the following dates:")
    print([d.strftime('%Y-%m-%d') for d in orbit_dates])
    insar.eof.download_eofs(orbit_dates)


if __name__ == '__main__':
    main()
