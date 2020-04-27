import datetime
import rasterio as rio
import numpy as np
import h5py
from insar.stackavg import load_geolist_intlist
from insar import timeseries

if __name__ == "__main__":
    # '/data1/scott/pecos/path78-bbox2/subset_injection/igrams_looked'
    # '/data4/scott/path85/stitched/subset_injection/igrams_looked'
    with h5py.File("unw_stack.h5") as f:
        unw_stack = f["stack_flat_shifted"][:]

    with rio.open("velocities_201706_linear_max800_noprune_170_5.tif") as src:
        velos = src.read(2)

    r, c = np.unravel_index(np.argmin(velos), velos.shape)
    geolist, intlist, valid_idxs = load_geolist_intlist("geolist_ignore.txt",
                                                        800,
                                                        max_date=datetime.date(2018, 1, 1))
    unw_pixel = unw_stack[valid_idxs, r, c]
    A = timeseries.build_A_matrix(geolist, intlist)
    phi = np.insert(np.linalg.pinv(A) @ unw_pixel, 0, 0)

    unw_subset = unw_stack[valid_idxs]
    pA = np.linalg.pinv(A)
    # unw_subset.shape, pA.shape
    # ((1170, 120, 156), (51, 1170))
    # So want to multiply first dim by the last dim
    full_solution = np.einsum('a b c, d a -> d b c', unw_subset, pA)
    with h5py.File("stack_final.h5", "w") as f:
        f["stack"] = full_solution[-1]
