import numpy as np
import itertools
from insar import timeseries
from datetime import datetime


def writefile(arr, filename):
    np.hstack((np.zeros((3, 2)), arr)).astype('float32').tofile(filename)


# Make 4 dummy 4x4 arrays of geo
# bottom row will be 0s
geo1 = np.zeros((4, 4)).astype(float)
geo2 = 1 + geo1
geo3 = 2 + geo2
geo4 = 4 + geo3
truth_geos = np.stack([geo1, geo2, geo3, geo4], axis=0)

# Add masks: 1 and 3 are all good, 2 and 4 have dead cells
mask1, mask2, mask3, mask4 = np.zeros((4, 4, 4))

mask2[3, 3] = 1
mask4[3, 3] = 1
mask4[0, 0] = 1

geolist = [
    datetime(2018, 1, 1),
    datetime(2018, 1, 2),
    datetime(2018, 1, 3),
    datetime(2018, 1, 4),
]
timediffs = timeseries.find_time_diffs(geolist)

geos_masked_list = [
    np.ma.array(geo1, mask=mask1, fill_value=np.NaN),
    np.ma.array(geo2, mask=mask2, fill_value=np.NaN),
    np.ma.array(geo3, mask=mask3, fill_value=np.NaN),
    np.ma.array(geo4, mask=mask4, fill_value=np.NaN),
]
geos_masked = np.ma.stack(geos_masked_list, axis=0)

igram_list = []
intlist = []  # tuples of dates
for early_idx, late_idx in itertools.combinations(range(len(truth_geos)), 2):
    early, late = geos_masked[early_idx], geos_masked[late_idx]
    early_date, late_date = geolist[early_idx], geolist[late_idx]
    intlist.append((early_date, late_date))

    igram_list.append(late - early)

igram_stack = np.ma.stack(igram_list, axis=0)

columns_masked = timeseries.stack_to_cols(igram_stack)

columns_with_masks = np.ma.count_masked(columns_masked, axis=0)
B = timeseries.build_B_matrix(geolist, intlist)

varr = timeseries.invert_sbas(columns_masked, B)
phi_hat = timeseries.integrate_velocities(varr, timediffs)
phi_hat = timeseries.cols_to_stack(phi_hat, *geo1.shape)

# # This array is the time series we want to see
# delta_phis = np.array([2, 14, 12, 14, 2]).reshape((-1, 1))
# # Also double the same one for variety
# delta_phis = np.hstack((delta_phis, 2 * delta_phis))
# # 3rd pixel down is the "reference": doesn't change over time
# delta_phis = np.hstack((delta_phis, np.zeros((5, 1))))
# delta_phis = np.dstack((delta_phis, delta_phis))
# print(delta_phis)
# print(delta_phis.shape)
# unwlist = [
#     '20180420_20180422.unw',
#     '20180420_20180428.unw',
#     '20180422_20180428.unw',
#     '20180422_20180502.unw',
#     '20180428_20180502.unw',
# ]
# for idx, name in enumerate(unwlist):
#     d = delta_phis[idx, :].reshape((3, 2))
#     print("Writing size ", d.shape)
#     print(d)
#     writefile(d, name)
