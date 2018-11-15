import numpy as np
import matplotlib.pyplot as plt
from insar import blob

from pick_out_blobs import *

os.chdir('/data3/scott/pecos/path85stitch-linear/igrams')
image = latlon.load_deformation_img('.')
blobs_real = np.load('blobs.npy')
blobs = np.load('blobs_0.5_0.2.npy')

blobs_real_with_vars_ptp = blob.append_stats(blobs_real, image=image)
blobs_with_vars_ptp = blob.append_stats(blobs_bad, image=image)
blobs_all = np.vstack([blobs_real_with_vars_ptp, blobs_with_vars_ptp])
clusters_3d = blob.cluster_blobs(np.abs(blobs_all))

fig, ax = blob.scatter_blobs_3d(np.abs(blobs_with_vars_ptp), color='r', label='bad')
blob.scatter_blobs_3d(np.abs(blobs_real_with_vars_ptp), color='g', label='real', ax=ax)
fig.tight_layout()
fig.legend()
fig, axes = blob.scatter_blobs(np.abs(blobs_with_vars_ptp), color='r', label='bad')
blob.scatter_blobs(np.abs(blobs_real_with_vars_ptp), color='g', label='real', axes=axes)

truth = np.zeros(len(clusters_3d))
truth[:len(blobs_real)] = 1
print('Missed blobs:')
print(blobs_all[np.where(truth - clusters_3d)[0]])
f, a = plotting.plot_image_shifted(image)
blob.plot_blobs(cur_axes=f.gca(), blobs=blobs_all[np.where(truth - clusters_3d)[0]])

plt.show()
