#!/usr/bin/env python
"""Starting script base

    Usage: {{script_name}}.py

"""

import argparse
import sys
import subprocess
from os.path import abspath, dirname, join, exists
try:
    import insar
except ImportError:  # add root to pythonpath if import fails
    sys.path.insert(0, dirname(dirname(abspath(__file__))))

from insar.log import get_log


def main():
    logger = get_log()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="filename.out", help="Output filename")
    args = parser.parse_args()


if __name__ == '__main__':
    main()
