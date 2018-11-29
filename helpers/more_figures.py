import numpy as np
import matplotlib.pyplot as plt
from insar import latlon, plotting
import sys

# defo = np.load('deformation.npy')
# defo_img = latlon.LatlonImage(data=np.mean(defo[-3:], axis=0), dem_rsc_file='dem.rsc')
defo2 = np.load('/data3/scott/pecos/path85stitch-linear/igrams/deformation.npy')
defo_img2 = latlon.LatlonImage(
    data=np.mean(defo2[-3:], axis=0), dem_rsc_file='/data3/scott/pecos/path85stitch/igrams/dem.rsc')

# fig, ax = plotting.make_figure_noborder()
# nrows, ncols = defo_img2.shape
# plotting.set_aspect_image(fig, defo_img2, height=8)
# ax.imshow(defo_img2, cmap='seismic', vmax=8, vmin=-8, aspect='auto')
# fig.savefig('test.png', dpi=80, transparent=True)
vmin = -5
vmax = 3.2
nrows, ncols = defo_img2.shape
fig, ax = plt.subplots(1, 1)
plotting.set_aspect_image(fig, defo_img2, height=10)
plotting.plot_image_shifted(
    defo_img2,
    fig=fig,
    cmap='discrete_seismic7',
    label='Centimeters',
    vmin=vmin,
    vmax=vmax,
    perform_shift=True)
fig.savefig(
    'defo_img_large_figure_discrete_seismic.png', bbox_inches='tight', transparent=True, dpi=300)

sys.exit()

# FIRST: smaller image
nrows, ncols = defo_img.shape

fig, ax = plt.subplots(1, 1)
plotting.plot_image_shifted(
    defo_img, fig=fig, cmap='discrete_seismic7', label='Centimeters', perform_shift=False)
fig.savefig('defo_img_figure.png', bbox_inches='tight', transparent=True, dpi=300)

fig, ax = plt.subplots(1, 1)
plotting.set_aspect_image(fig, defo_img2, height=8)
plotting.plot_image_shifted(
    defo_img, fig=fig, cmap='seismic', label='Centimeters', perform_shift=False)
fig.savefig('defo_img_figure_seismic.png', bbox_inches='tight', transparent=True, dpi=300)

# NOTE: for noaxes image, remove bbox_inches = tight
fig, ax = plotting.make_figure_noborder()
plotting.set_aspect_image(fig, defo_img, height=8)
ax.imshow(defo_img, cmap='seismic', vmax=8, vmin=-8, aspect='auto')
fig.savefig('defo_img_noaxes_seismic.png', transparent=True, dpi=300)

fig, ax = plotting.make_figure_noborder()
plotting.set_aspect_image(fig, defo_img, height=8)
ax.imshow(defo_img, cmap='discrete_seismic7', vmax=5, vmin=-5, aspect='auto')
print('defo_img_noaxes_discrete_seismic.png')
fig.savefig('defo_img_noaxes_discrete_seismic.png', transparent=True, dpi=300)

# SECOND: large image
vmin = -5
vmax = 3.2
nrows, ncols = defo_img2.shape
fig, ax = plt.subplots(1, 1)
plotting.set_aspect_image(fig, defo_img2, height=10)
plotting.plot_image_shifted(
    defo_img2,
    fig=fig,
    cmap='discrete_seismic7',
    label='Centimeters',
    vmin=vmin,
    vmax=vmax,
    perform_shift=True)
fig.savefig(
    'defo_img_large_figure_discrete_seismic.png', bbox_inches='tight', transparent=True, dpi=300)

fig, ax = plotting.make_figure_noborder()
plotting.set_aspect_image(fig, defo_img2, height=24)
plotting.plot_image_shifted(
    defo_img2,
    fig=fig,
    cmap='discrete_seismic7',
    label='Centimeters',
    vmin=vmin,
    vmax=vmax,
    perform_shift=True,
    colorbar=False)
fig.savefig('defo_img_large_noaxes_discrete_seismic.png', transparent=True, dpi=300)
print('defo_img_large_noaxes_discrete_seismic.png')
