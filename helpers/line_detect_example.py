import numpy as np

import skimage
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
from skimage.feature import canny
# from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm

# Constructing test image
image = np.zeros((100, 100))
idx = np.arange(25, 75)
image[idx[::-1], idx] = 255
image[idx, idx] = 255

defo1 = np.load('path78_defo.npy')
image = skimage.exposure.rescale_intensity(np.nan_to_num(defo1, 0))
# defo1 = np.load('path85_defo.npy')
# image = skimage.exposure.rescale_intensity(defo1[:, :1000])

# edges = canny(image, sigma=3.0, low_threshold=.1, high_threshold=.6, mask=None, use_quantiles=True)
edges = canny(image, sigma=5.0, low_threshold=.3, high_threshold=.9, mask=None, use_quantiles=True)

# Classic straight-line Hough transform
# h, theta, d = hough_line(image)
h, theta, d = hough_line(edges)

# Generating figure 1
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(
    np.log(1 + h),
    extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
    cmap=cm.gray,
    aspect=1 / 1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

ax[2].imshow(image, cmap=cm.gray)
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
    ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

plt.tight_layout()

# Line finding using the Probabilistic Hough Transform
# image = data.camera()
# edges = canny(image, 2, 1, 25)
lines = probabilistic_hough_line(edges, threshold=5, line_length=100, line_gap=30)

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edges, cmap=cm.gray)
ax[1].set_title('Canny edges')

ax[2].imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()
