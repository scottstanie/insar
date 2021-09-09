import os
from apertools import utils
import subprocess

"""
[scott@grace l1-l0-mountain-test-egm96]$ tree
.
├── elevation.dem
├── elevation.dem.rsc
├── l0
│   └── data
└── l1
    ├── data
"""
for dirname, lvl in zip(["l1/data/", "l0/data/"], ["SLC", "RAW"]):
    utils.mkdir_p(dirname)
    with utils.chdir_then_revert(dirname):
        cmd = f"asfdownload --proce {lvl} --start 2020-05-23 --end 2020-05-24 --dem ../elevation.dem --rel 151"
        print(cmd)
        subprocess.run(cmd, shell=True)
        cmd = f"asfdownload --proce {lvl} --start 2020-06-04 --end 2020-06-05 --dem ../elevation.dem --rel 151"
        print(cmd)
        subprocess.run(cmd, shell=True)