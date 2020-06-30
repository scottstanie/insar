import datetime
import rasterio as rio
import numpy as np
import h5py
from insar.stackavg import load_geolist_intlist, find_valid
import apertools.sario as sario
from insar import timeseries
from scipy.ndimage.filters import gaussian_filter


def _load_unw_stack(unw_file, unw_stack):
    if unw_stack is None:
        with h5py.File(unw_file) as f:
            unw_stack = f["stack_flat_shifted"][:]
    return unw_stack


def main(
    unw_file=None,
    unw_stack=None,
    max_date=None,
    max_temporal_baseline=800,
):
    unw_stack = _load_unw_stack(unw_file, unw_stack)
    # '/data1/scott/pecos/path78-bbox2/subset_injection/igrams_looked'
    # '/data4/scott/path85/stitched/subset_injection/igrams_looked'

    # with rio.open("velocities_201706_linear_max800_noprune_170_5.tif") as src:
    # velos = src.read(2)
    # r, c = np.unravel_index(np.argmin(velos), velos.shape)
    geo_date_list = sario.load_geolist_from_h5(unw_file)
    igram_date_list = sario.load_intlist_from_h5(unw_file)

    # geolist, intlist, valid_idxs = load_geolist_intlist("geolist_ignore.txt",
    #                                                     800,
    #                                                     max_date=datetime.date(2018, 1, 1))
    geolist, intlist, valid_idxs = find_valid(
        geo_date_list,
        igram_date_list,
        max_date=max_date,
        ignore_geo_file="geolist_ignore.txt",
        max_temporal_baseline=max_temporal_baseline,
    )
    A = timeseries.build_A_matrix(geolist, intlist)
    # r, c = 300, 300
    # unw_pixel = unw_stack[valid_idxs, r, c]
    # phi = np.insert(np.linalg.pinv(A) @ unw_pixel, 0, 0)

    unw_subset = unw_stack[valid_idxs]
    pA = np.linalg.pinv(A)
    # unw_subset.shape, pA.shape
    # ((1170, 120, 156), (51, 1170))
    # So want to multiply first dim by the last dim
    stack = np.einsum('a b c, d a -> d b c', unw_subset, pA)
    # import ipdb; ipdb.set_trace()
    with h5py.File("stack_final.h5", "w") as f:
        f["stack"] = stack
    return stack


def filter_aps(stack, space_sigma=5, time_sigma=3):
    """Performes temporal filter in time (axis 0), and spatial filter (axis 1, 2)

    """
    # First, to HP filt, subtract the LP filt from the original
    lp_time = gaussian_filter(stack, [time_sigma, 0, 0], mode='constant')
    hp_time = stack - lp_time

    # Then, low pass filt in space to get the estimate of atmo phase screen
    return gaussian_filter(hp_time, [0, space_sigma, space_sigma])


def solve_linear_offset(unw_file=None, unw_stack=None):
    unw_stack = _load_unw_stack(unw_file, unw_stack)
    eq_date = datetime.date(2020, 3, 26)
    # geo_date_list = sario.load_geolist_from_h5(unw_file)
    igram_date_list = sario.load_intlist_from_h5(unw_file)

    # TODO: valid idx stuff
    # Temporal matrix, for constant velo estimate
    T = np.array([(ifg[1] - ifg[0]).days for ifg in igram_date_list]).reshape((-1, 1))
    # "jump" matrix: indicator 1 if ifg contains eq
    J = np.array([(ifg[0] < eq_date and ifg[1] > eq_date) for ifg in igram_date_list])
    J = J.reshape((-1, 1)).astype(int)
    A = np.hstack((T, J))
    pA = np.linalg.pinv(A)
    # print(unw_stack.shape, pA.shape)
    # (465, 720, 720) (2, 465)
    # So want to multiply first dim by the last dim
    vj = np.einsum('a b c, d a -> d b c', unw_stack, pA)
    with h5py.File("stack_velos_jump.h5", "w") as f:
        f["velos/1"] = vj[0]
        f["jump/1"] = vj[1]
    return vj


# if __name__ == "__main__":
# main(unw_stack)
