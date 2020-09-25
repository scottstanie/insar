from apertools import latlon, plotting
from insar import blob
import matplotlib.pyplot as plt
import numpy as np
import skimage
import cv2 as cv

img = cv.cvtColor(
    cv.cvtColor(cv.imread("/home/scott/Pictures/sunflowers.jpg"), cv.COLOR_BGR2RGB),
    cv.COLOR_RGB2GRAY,
)

defo1 = latlon.LatlonImage(filename="path85_defo.npy", dem_rsc_file="dem.rsc")[:, :1000]
# defo1 = latlon.load_deformation_img('./data/igrams_subset')
# defo1u = skimage.img_as_ubyte(defo1 / np.nanmax(np.abs(defo1)))
defo1u = blob.utils.img_as_uint8(defo1)
img = defo1u
# img = 255 - defo1u

mser = cv.MSER_create()
mser.setMinArea(220)
vis = img.copy()
regions, bboxes = mser.detectRegions(img)
hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]

# cv.polylines(vis, hulls, 1, (128, 255, 0))
# cv.imshow('img', vis)
# cv.waitKey()
# cv.destroyAllWindows()

fig, axes = plotting.plot_image_shifted(defo1)
rr, bb = blob.utils.prune_regions(regions, bboxes)
for r in rr:
    blob.plot.plot_hull(regions=r, ax=fig.gca())

plt.show()
