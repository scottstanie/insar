import datetime
import rasterio as rio
import numpy as np
import h5py
from insar.stackavg import load_geolist_intlist, find_valid
import apertools.sario as sario
import apertools.utils as utils
import apertools.netcdf as netcdf
from insar import timeseries
from insar.timeseries import PHASE_TO_CM, cols_to_stack, stack_to_cols
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
    max_bandwidth=100,
    outfile="stack_final.h5",
):
    unw_stack = _load_unw_stack(unw_file, unw_stack)
    # '/data1/scott/pecos/path78-bbox2/subset_injection/igrams_looked'
    # '/data4/scott/path85/stitched/subset_injection/igrams_looked'

    # with rio.open("velocities_201706_linear_max800_noprune_170_5.tif") as src:
    # velos = src.read(2)
    # r, c = np.unravel_index(np.argmin(velos), velos.shape)
    # geolist_full = sario.load_geolist_from_h5(unw_file)
    # intlist_full = sario.load_intlist_from_h5(unw_file)
    geolist, intlist = sario.load_geolist_intlist(
        h5file=unw_file, geolist_ignore_file="geolist_ignore.txt"
    )

    # geolist, intlist, valid_idxs = load_geolist_intlist("geolist_ignore.txt",
    #                                                     800,
    #                                                     max_date=datetime.date(2018, 1, 1))
    # valid_sar_date, valid_ifg_dates, valid_ifg_idxs = utils.filter_geolist_intlist(
    geolist, intlist, valid_ifg_idxs = utils.filter_geolist_intlist(
        ifg_date_list=intlist,
        max_date=max_date,
        max_temporal_baseline=max_temporal_baseline,
        max_bandwidth=max_bandwidth,
    )
    # geolist, intlist, valid_idxs = find_valid(
    # geolist_full,
    # intlist_full,
    # max_date=max_date,
    # ignore_geo_file="geolist_ignore.txt",
    # max_temporal_baseline=max_temporal_baseline,
    # )
    A = timeseries.build_A_matrix(geolist, intlist)
    # r, c = 300, 300
    # unw_pixel = unw_stack[valid_idxs, r, c]
    # phi = np.insert(np.linalg.pinv(A) @ unw_pixel, 0, 0)

    unw_subset = unw_stack[valid_ifg_idxs]
    print(f"Selecting subset, shape = {unw_subset.shape}")
    pA = np.linalg.pinv(A)
    print(f"Forming pseudo-inverse of A")
    # unw_subset.shape, pA.shape
    # ((1170, 120, 156), (51, 1170))
    # So want to multiply first dim by the last dim
    # print("Running Einsum to solve")
    # stack = np.einsum("a b c, d a -> d b c", unw_subset, pA)
    # Quicker than einsum: reshape to cols, one multiply
    ni, r, c = unw_subset.shape  # igrams, rows, cols
    stack = cols_to_stack(pA @ stack_to_cols(unw_subset), *unw_subset.shape[1:])
    # equiv:
    # stack = (pA @ unw_subset.reshape((ni, -1))).reshape((nd, r, c))

    # Add a 0 image for the first date
    stack = np.concatenate((np.zeros((1, r, c)), stack), axis=0)
    stack *= PHASE_TO_CM
    # import ipdb; ipdb.set_trace()
    dset = "stack"
    with h5py.File(outfile, "w") as f:
        f[dset] = stack
    sario.save_geolist_to_h5(out_file=outfile, geo_date_list=geolist, dset_name=dset)
    dem_rsc = sario.load_dem_from_h5(unw_file)
    sario.save_dem_to_h5(outfile, dem_rsc)
    netcdf.hdf5_to_netcdf(
        outfile,
        dset_name=dset,
        stack_dim="date",
    )
    return stack


def filter_aps(stack, space_sigma=5, time_sigma=3):
    """Performes temporal filter in time (axis 0), and spatial filter (axis 1, 2)"""
    # First, to HP filt, subtract the LP filt from the original
    lp_time = gaussian_filter(stack, [time_sigma, 0, 0], mode="constant")
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

    # print(pA.shape, unw_stack.shape)
    # (2, 465) (465, 720, 720)
    # So want to multiply first dim by the last dim
    # vj = np.einsum("a b c, d a -> d b c", unw_stack, pA)
    ni, r, c = unw_stack.shape  # igrams, rows, cols
    vj = cols_to_stack(pA @ stack_to_cols(unw_stack), r, c)
    with h5py.File("stack_velos_jump.h5", "w") as f:
        f["velos/1"] = vj[0]
        f["jump/1"] = vj[1]
    return vj


def solve_multiple_bandwidths(unw_stack, bandwidths, rr, cc, max_date=None):
    ifg_dates_all = sario.load_intlist_from_h5("unw_stack.h5")
    phi_list = []
    sar_date_list = []
    for bw in bandwidths:
        geolist, intlist, valid_idx = utils.filter_geolist_intlist(
            ifg_date_list=ifg_dates_all,
            max_date=max_date,
            max_bandwidth=bw,
            ignore_file="geolist_ignore.txt",
        )
        A = timeseries.build_A_matrix(geolist, intlist)

        unw_pixel = unw_stack[valid_idx, rr, cc]
        # phi = pA @ unw_subset
        phi = np.insert(np.linalg.pinv(A) @ unw_pixel, 0, 0)
        phi_list.append(phi * PHASE_TO_CM)
        sar_date_list.append(geolist)
    return sar_date_list, phi_list


def plot_bw(sar_date_list, phi_list, bandwidths):
    import matplotlib.pyplot as plt

    bw_phi = sorted(zip(bandwidths, phi_list), key=lambda tup: tup[0], reverse=True)
    fig, axes = plt.subplots(1, 2)
    for idx, (bw, p) in enumerate(bw_phi):
        dates = sar_date_list[idx]
        ax = axes[0]
        ax.plot(dates, p, "-x", label=bw)

        ax = axes[1]
        ax.plot(dates, p - bw_phi[0][1], "-x", label=bw)
    ax.legend()
    ax.grid(True)
    

    # if __name__ == "__main__":
    # main(unw_stack)
