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
        try:
            parser = Sentinel(filename)
        except ValueError:  # Not a sentinel zip file
            print('Skipping {}'.format(filename))
            continue

        start_date = parser.start_stop_time()[0]
        mission = parser.mission()
        print("Downloading precise orbits for {} on {}".format(
            mission, start_date.strftime('%Y-%m-%d')))
        orbit_dates.append(start_date)

    insar.eof.download_eofs(orbit_dates, mission=mission)


if __name__ == '__main__':
    main()
