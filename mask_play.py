# coding: utf-8
get_ipython().run_line_magic('pylab', '')
import sar.io; import sar.utils
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
file1 = '/home/scott/uav-sar-data/one-strip-test/brazos-20170831-mlc/brazos_14937_17087_002_170831_L090HHHV_CX_01.1.mlc'
file3 = 'home/scott/uav-sar-data/one-strip-test/brazos-20170903-mlc/brazos_14937_17090_017_170903_L090HHHH_CX_01.1.mlc'
file2 = '/home/scott/uav-sar-data/one-strip-test/brazos-20170901-mlc/brazos_14937_17088_003_170901_L090HHHH_CX_01.1.mlc'
file3 = '/home/scott/uav-sar-data/one-strip-test/brazos-20170903-mlc/brazos_14937_17090_017_170903_L090HHHH_CX_01.1.mlc'
mlc1 = sar.io.load_file(file1)
mlc2 = sar.io.load_file(file2)
mlc3 = sar.io.load_file(file3)
import create_mask
get_ipython().run_line_magic('ls', '')
import create_masks
mask1 = create_masks.mask(file1)
figure()
plt.imshow(mask1)
mask2 = create_masks.mask(file2)
mask3 = create_masks.mask(file3)
plt.imshow(mask2)
plt.imshow(mask3)
mask1 = create_masks.mask(file1)
plt.imshow(mask1)
mask12 = mask1 & (not mask2)
mask12 = mask1 & (logical_not(mask2))
mask1.shape
mask2.shape
mask3.shape
mask1 = mask1[:2900, :]
mask2 = mask2[:2900, :]
mask3 = mask3[:2900, :]
mask12 = mask1 & (logical_not(mask2))
plt.imshow(mask12)
mask23 = mask2 & (logical_not(mask3))
plt.imshow(mask23)
subplot(221)
plt.imshow(mask3)
subplot(222)
plt.imshow(mask12)
subplot(223)
plt.imshow(mask23)
subplot(224)
plt.imshow(mask1)
