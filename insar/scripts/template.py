#!/usr/bin/env python
"""Starting script base

    Usage: {{script_name}}.py

"""

import argparse
import sys
import subprocess
from os.path import abspath, dirname, join, exists

from insar.log import get_log


def main():
    logger = get_log()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="filename.out", help="Output filename")
    args = parser.parse_args()


if __name__ == '__main__':
    main()
