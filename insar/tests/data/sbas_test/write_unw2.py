import numpy as np
import itertools
from insar import timeseries, mask
from apertools import sario
from datetime import datetime

REF_ROW_COL = (2, 2)

# 5 geos made so that the most igrams any date can make is
# n-1 = 4 igrams (but 10 total igrams among all)
geo_date_list = [
    datetime(2018, 1, 1),
    datetime(2018, 1, 2),
    datetime(2018, 1, 3),
    datetime(2018, 1, 4),
    datetime(2018, 1, 5),
    datetime(2018, 1, 6),
]

# Make 4 dummy 4x4 arrays of geo
# bottom row will be 0s
geo1 = np.zeros((4, 4)).astype(np.complex64)
geo2 = 1 + geo1
geo3 = 2 + geo2
geo4 = 4 + geo3
geo5 = 5 + geo4
geo6 = 6 + geo5
geo_tuple = [geo1, geo2, geo3, geo4, geo5, geo6]
# # Method 1: make masks manually
# # Add masks: 1 and 3 are all good, 2 and 4 have dead cells
# mask1, mask2, mask3, mask4, mask5 = np.zeros((5, 4, 4))
#
# mask2[3, 3] = 1
# mask4[3, 3] = 1
# mask4[0, 0] = 1

# Method 2: make some pixels 0, and run mask functions
geo_tuple[1][3, 3] = 0
geo_tuple[3][3, 3] = 0

geo_tuple[2][0, 0] = 0

truth_geos = np.stack(geo_tuple, axis=0)
print(truth_geos)

import ipdb
# ipdb.set_trace()
with open("geolist", "w") as f:
    for idx, geo in enumerate(truth_geos):
        fname = 'S1A_{}.geo'.format(geo_date_list[idx].strftime('%Y%m%d'))
        geo.tofile(fname)
        f.write("%s\n" % fname)

timediffs = timeseries.find_time_diffs(geo_date_list)

# geos_masked_list = [
#     np.ma.array(geo1, mask=mask1, fill_value=np.NaN),
#     np.ma.array(geo2, mask=mask2, fill_value=np.NaN),
#     np.ma.array(geo3, mask=mask3, fill_value=np.NaN),
#     np.ma.array(geo4, mask=mask4, fill_value=np.NaN),
#     np.ma.array(geo5, mask=mask5, fill_value=np.NaN),
# ]
# geos_masked = np.ma.stack(geos_masked_list, axis=0)

igram_list = []  # list of arrays
igram_date_list = []  # list of tuples of dates
igram_fname_list = []  # list of strings
for early_idx, late_idx in itertools.combinations(range(len(truth_geos)), 2):
    # Using masked data
    # early, late = geos_masked[early_idx], geos_masked[late_idx]
    # Using truth data to form igrams:
    early, late = truth_geos[early_idx], truth_geos[late_idx]

    early_date, late_date = geo_date_list[early_idx], geo_date_list[late_idx]
    igram_date_list.append((early_date, late_date))

    # igram = np.abs(late) - np.abs(early)
    igram_complex = late - early
    igram_complex[REF_ROW_COL] = 0
    igram = np.abs(igram_complex)
    fname = '{}_{}.int'.format(early_date.strftime('%Y%m%d'), late_date.strftime('%Y%m%d'))
    igram_complex.tofile(fname)

    # Note: using the 'height" as both amplitude and height
    new_unw = np.stack((igram, igram), axis=0)
    sario.save(fname.replace(".int", ".unw"), new_unw)

    igram_fname_list.append(fname)
    igram_list.append(igram)

with open("intlist", "w") as f:
    for ig in igram_fname_list:
        f.write("%s\n" % ig)

mask.save_int_masks(igram_fname_list, igram_date_list, geo_date_list, geo_path='.')

geo_masks = sario.load_stack(directory='.', file_ext='.geo.mask.npy')
geo_mask_columns = timeseries.stack_to_cols(geo_masks)

int_mask_file_names = [f + '.mask.npy' for f in igram_fname_list]
int_mask_stack = sario.load_stack(file_list=int_mask_file_names)
igram_stack = np.ma.stack(igram_list, axis=0)
igram_stack.mask = int_mask_stack

columns_masked = timeseries.stack_to_cols(igram_stack)

# columns_with_masks = np.ma.count_masked(columns_masked, axis=0)
B = timeseries.build_B_matrix(geo_date_list, igram_date_list)

varr = timeseries.invert_sbas(columns_masked, B, geo_mask_columns)
phi_hat = timeseries.integrate_velocities(varr, timediffs)
print("phi_hat velos", phi_hat.astype(int))
phi_hat = timeseries.cols_to_stack(phi_hat, *geo1.shape)
print("phi_hat", phi_hat.astype(int))
