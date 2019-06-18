# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
from apertools import plotting, latlon
from insar import blob

real_blob_locs = []
real_blob_locs.append((141, 288))
real_blob_locs.append((140, 608))
real_blob_locs.append((140, 628))
real_blob_locs.append((163, 618))
real_blob_locs.append((120, 605))
real_blob_locs.append((620, 925))
real_blob_locs.append((1060, 922))
real_blob_locs.append((1010, 798))
real_blob_locs.append((970, 778))
real_blob_locs.append((1015, 75))
real_blob_locs.append((972, 5))
real_blob_locs.append((990, 110))
real_blob_locs.append((934, 168))
real_blob_locs.append((1060, 15))
real_blob_locs.append((980, 55))
real_blob_locs.append((1225, 1315))
real_blob_locs.append((1225, 1315))
real_blob_locs.append((1163, 1295))

os.chdir('/data3/scott/pecos/path85stitch-linear/igrams')
blobs_real = np.load('blobs.npy')
blobs = np.load('blobs_0.5_0.2.npy')
image = latlon.load_deformation_img('.')

plotting.plot_image_shifted(image)

real_blob_locs = np.array(real_blob_locs)
np.save('real_blob_locs', real_blob_locs)
for y, x in real_blob_locs:
    plt.plot(x, y, 'gx', ms=10)

# btest = np.array([512, 558, 35, -2])
truths = np.zeros(blobs.shape[0], ).astype(bool)
for idx, b in enumerate(blobs):
    m1 = blob.indexes_within_circle(mask_shape=image.shape, blob=b)
    truths[idx] = any(m1[r, c] for r, c in real_blob_locs)

blobs_bad = blobs[~truths]
blobs_duped = blobs[truths]
_, ax = blob.plot_blobs(image=image, blobs=blobs_bad)
ax.set_title('blobs bad')
_, ax = blob.plot_blobs(image=image, blobs=blobs_duped)
ax.set_title('blobs duped')
