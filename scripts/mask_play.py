# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import insar.sario
import insar.utils
import create_masks
file1 = '/home/scott/uav-sar-data/one-strip-test/brazos-20170831-mlc/brazos_14937_17087_002_170831_L090HHHV_CX_01.1.mlc'
file3 = 'home/scott/uav-sar-data/one-strip-test/brazos-20170903-mlc/brazos_14937_17090_017_170903_L090HHHH_CX_01.1.mlc'
file2 = '/home/scott/uav-sar-data/one-strip-test/brazos-20170901-mlc/brazos_14937_17088_003_170901_L090HHHH_CX_01.1.mlc'
file3 = '/home/scott/uav-sar-data/one-strip-test/brazos-20170903-mlc/brazos_14937_17090_017_170903_L090HHHH_CX_01.1.mlc'
mlc1 = insar.sario.load_file(file1)
mlc2 = insar.sario.load_file(file2)
mlc3 = insar.sario.load_file(file3)

mask1 = create_masks.mask(file1)
mask2 = create_masks.mask(file2)
mask3 = create_masks.mask(file3)

mask1 = mask1[:2900, :]
mask2 = mask2[:2900, :]
mask3 = mask3[:2900, :]

mask12 = mask1 & (np.logical_not(mask2))
mask23 = mask2 & (np.logical_not(mask3))

plt.subplot(221)
plt.imshow(mask3)
plt.title('Mask 3')
plt.subplot(222)
plt.imshow(mask12)
plt.title('Mask 12')
plt.subplot(223)
plt.imshow(mask23)
plt.title('Mask 23')
plt.subplot(224)
plt.imshow(mask1)
plt.title('Mask 1')

plt.show(block=True)
