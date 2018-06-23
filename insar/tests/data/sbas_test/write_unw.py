# coding: utf-8
import numpy as np


def writefile(arr, filename):
    np.hstack((np.zeros((3, 1)), arr)).astype('float32').tofile(filename)


delta_phis = np.array([2, 14, 12, 14, 2]).reshape((-1, 1))
delta_phis = np.hstack((delta_phis, 2 * delta_phis))
# 3rd pixel down is the "reference": doesn't change over time
delta_phis = np.hstack((delta_phis, np.zeros((5, 1))))
print(delta_phis)
print(delta_phis.shape)
unwlist = [
    '20180420_20180422.unw',
    '20180420_20180428.unw',
    '20180422_20180428.unw',
    '20180422_20180502.unw',
    '20180428_20180502.unw',
]
for idx, name in enumerate(unwlist):
    d = delta_phis[idx, :].reshape((3, 1))
    print("Writing size ", d.shape)
    print(d)
    writefile(d, name)
