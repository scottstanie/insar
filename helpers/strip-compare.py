# coding: utf-8
from apertools import sario, utils
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

try:
    slclist = glob.glob(sys.argv[1])
    print("loading from", slclist)
    g1 = sario.load(slclist[0])
    g2 = sario.load(slclist[1])
except IndexError:
    dpath = "/data3/scott/pecos/stitch-test/"
    print("loading from", dpath)
    g1 = sario.load(
        dpath
        + "S1A_IW_SLC__1SSV_20160930T125457_20160930T125524_013282_015288_4D8C.SAFE.geo"
    )
    g2 = sario.load(
        dpath
        + "S1A_IW_SLC__1SSV_20160930T125522_20160930T125550_013282_015288_EFE5.SAFE.geo"
    )
overlap_idxs = (g1 != 0) & (g2 != 0)

strip1 = np.zeros_like(g1)
strip1[overlap_idxs] = g1[overlap_idxs]

strip2 = np.zeros_like(g2)
strip2[overlap_idxs] = g2[overlap_idxs]

fig, axes = plt.subplots(1, 2)
db_img = axes[0].imshow(utils.db(strip2 - strip1))
axes[0].set_title("magnitude of strip difference")
angle_img = axes[1].imshow(np.angle(strip2 - strip1))
axes[1].set_title("phase of strip difference")
# fig.colorbar(angle_img)
fig.colorbar(db_img)

plt.show()
