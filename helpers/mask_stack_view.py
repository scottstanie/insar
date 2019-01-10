import sys
import numpy as np
import matplotlib.pyplot as plt
from insar import sario

try:
    path = sys.argv[1]
except IndexError:
    path = '.'

masks = sario.load_stack(directory=path, file_ext='*.mask.npy')
total = len(masks)
mask_sum = np.sum(masks.astype(int), axis=0)
plt.figure()
plt.imshow(mask_sum)
plt.colorbar()
plt.title("Masked Regions (higher=more mask, out of %s total)" % total)
plt.show()
