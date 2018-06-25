#!/usr/bin/env python
"""Simple script to quickly view one or more elevation dems with matplotlib

    Usage: view_dem.py [demname1.dem [demname2.dem ...]]

As many as you'd like can be shown in different figures.
With no arguments, searches for elevation.dem to show
"""
import sys
from os.path import dirname, abspath
from insar.sario import load_file
import matplotlib.pyplot as plt


def main():
    usage_string = "Usage: view_dem.py [demname1.dem [demname2.dem ...]]"
    fname_list = sys.argv[1:] if len(sys.argv) > 1 else ['elevation.dem']
    for fname in fname_list:
        try:
            dem = load_file(fname)
        except FileNotFoundError:
            print(usage_string)
            raise
        plt.figure()
        plt.imshow(dem)
        plt.colorbar()

    # Wait for windows to close to exit the script
    plt.show(block=True)


if __name__ == '__main__':
    main()
