import sys
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
if len(sys.argv) > 1 and sys.argv[1].startswith("--down"):
    for dirname, lvl in zip(["l1/data/", "l0/data/"], ["SLC", "RAW"]):
        utils.mkdir_p(dirname)
        with utils.chdir_then_revert(dirname):
            cmd = f"asfdownload --proce {lvl} --start 2019-12-07 --end 2019-12-08 --dem ../../elevation.dem --rel 151"
            print(cmd)
            subprocess.run(cmd, shell=True)
            cmd = f"asfdownload --proce {lvl} --start 2020-05-23 --end 2020-05-24 --dem ../../elevation.dem --rel 151"
            print(cmd)
            subprocess.run(cmd, shell=True)

for dirname in ["l1/", "l0/"]:
    with utils.chdir_then_revert(dirname):
        subprocess.run("ln -s data/* .", shell=True)


for dirname in ["l1/", "l0/"]:
    with utils.chdir_then_revert(dirname):
        subprocess.run("eof", shell=True)
        subprocess.run("insar process --step 3-9", shell=True)
