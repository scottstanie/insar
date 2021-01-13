# coding: utf-8
import sys
import matplotlib.pyplot as plt
import numpy as np
from insar import blob
from insar.blob import synthetic

if len(sys.argv) < 4:
    print("Usage: python %s nrows sigma amp" % sys.argv[0])
    sys.exit(1)

nrows = int(sys.argv[1])
sigma = int(sys.argv[2])
amp = int(sys.argv[3])

out = synthetic.make_gaussian((nrows, nrows), sigma, amp=amp, noise_sigma=1)

phase = np.angle(np.exp(1j * out))

plt.figure()
plt.imshow(phase)
plt.title("Wrapped signal for max phase of {amp}, sigma={sigma}".format(amp=amp, sigma=sigma))

nsigma = 8
sigma_list = blob.skblob.create_sigma_list(min_sigma=max(sigma - 40, 4),
                                           max_sigma=sigma + 20,
                                           num_sigma=nsigma)

image_cube_phase = blob.skblob.create_gl_cube(phase, sigma_list=sigma_list)
image_cube = blob.skblob.create_gl_cube(out, sigma_list=sigma_list)

fig1, axes1 = plt.subplots(nsigma // 2, 2)
maxv = np.max(np.abs(image_cube))
for idx, ax in enumerate(axes1.ravel()):
    axim = ax.imshow(image_cube[:, :, idx], vmin=-maxv, vmax=maxv)
    fig1.colorbar(axim, ax=ax)
    ax.set_title("Sigma=%d" % sigma_list[idx])
fig1.suptitle("Unwrapped LoG responses")

fig2, axes2 = plt.subplots(nsigma // 2, 2)
maxv = np.max(np.abs(image_cube_phase))
for idx, ax in enumerate(axes2.ravel()):
    axim = ax.imshow(-image_cube_phase[:, :, idx], vmin=-maxv, vmax=maxv)
    fig2.colorbar(axim, ax=ax)
    ax.set_title("Sigma=%d" % sigma_list[idx])

fig2.suptitle("wrapped LoG responses")

plt.show(block=True)
