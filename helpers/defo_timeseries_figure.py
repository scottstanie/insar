import numpy as np
import matplotlib.pyplot as plt
from apertools import latlon

defo = np.load("deformation.npy")
defo_img = latlon.LatlonImage(data=np.mean(defo[-3:], axis=0), dem_rsc_file="dem.rsc")
row, col = 146, 250
lat, lon = defo_img.rowcol_to_latlon(row, col)
timeline = defo[:, row, col]
geolist = np.load("geolist.npy")

plt.plot(geolist, timeline, marker="o", linestyle="dashed", linewidth=2, markersize=6)
plt.grid("on")
legend = ["Lat {:.3f}, Lon {:.3f}".format(lat, lon)]
plt.xlabel("SAR Image Date")
plt.ylabel("Centimeters of uplift")
plt.title("Deformation timeseries")
plt.savefig("defo_timeseries.png", bbox_inches="tight", transparent=True, dpi=300)
