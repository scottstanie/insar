import os
from datetime import date
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import h5py

from insar import timeseries
from apertools import sario, gps, plotting, latlon
from apertools.colors import MATLAB_COLORS
from apertools.log import get_log

logger = get_log()

p2c = timeseries.PHASE_TO_CM
p2mm = timeseries.PHASE_TO_CM * 365 * 10

# TODO: MOVE THESE!!!
from insar import constants


station_name_list = [
    "NMHB",
    "TXAD",
    "TXBG",
    "TXBL",
    "TXCE",
    "TXFS",
    "TXKM",
    "TXL2",
    "TXMC",
    "TXMH",
    "TXOE",
    "TXOZ",
    "TXS3",
    "TXSO",
]


def set_rcparams():
    # https://matplotlib.org/tutorials/introductory/customizing.html#a-sample-matplotlibrc-file
    # https://matplotlib.org/3.1.1/tutorials/introductory/lifecycle.html
    style_dict = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "Helvetica",
        "font.size": 16,
        "font.weight": "bold",
    }
    mpl.rcParams.update(style_dict)


def plot_phase_vs_elevation(
    dem=None,
    unw_stack=None,
    outname="phase_vs_elevation_noraster.pdf",
    igram_idx=210,
    el_cutoff=1200,
    to_cm=True,
    rasterized=True,
):
    if unw_stack is None:
        with h5py.File("unw_stack_shiftedonly.h5", "r") as f:
            unw_stack = f["stack_flat_shifted"][:]
    if dem is None:
        dem = sario.load("elevation_looked.dem")

    # every = 5
    # X = np.repeat(dem[np.newaxis, ::every, 100:-100:every], 400, axis=0).reshape(-1).astype(float)
    # X += 30 * np.random.random(X.shape)
    # Y = unw_stack[:4000:10, ::every, 100:-100:every].reshape(-1)
    X = dem[:, 100:600].reshape(-1)
    Y = unw_stack[igram_idx, :, 100:600].reshape(-1)
    if el_cutoff:
        good_idxs = X < el_cutoff
        X, Y = X[good_idxs], Y[good_idxs]

    if to_cm:
        Y *= 0.44

    plt.style.use("default")
    # plt.style.use('ggplot')
    # plt.style.use('seaborn-paper')
    set_rcparams()
    # https://matplotlib.org/tutorials/introductory/customizing.html#a-sample-matplotlibrc-file
    # https://matplotlib.org/3.1.1/tutorials/introductory/lifecycle.html
    style_dict = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "Helvetica",
        "font.size": 16,
        "font.weight": "bold",
    }
    mpl.rcParams.update(style_dict)
    fig, ax = plt.subplots()
    # ax.scatter(X, Y, s=0.8)
    ax.plot(X, Y, "o", rasterized=rasterized, ms=0.8)
    # ax.set_xlim(700, 1200)
    ax.set_xlim(None, 1200)
    ax.set_xlabel("Elevation (m)")
    # ax.set_ylabel("Phase (rad)")
    ax.set_ylabel("[cm]")
    plt.show(block=False)
    fig.savefig("phase_vs_elevation_noraster.pdf", dpi=200, transparent=True)
    return fig, ax


def plot_phase_elevation_igrams(dem, unw_stack, n=10, start=0, el_cutoff=None):
    nn = np.ceil(np.sqrt(n)).astype(int)
    fig, axes = plt.subplots(nn, nn)
    for idx, ax in enumerate(axes.ravel()):
        X = dem[:, 100:600].reshape(-1)
        Y = unw_stack[start + idx, :, 100:600].reshape(-1)
        if el_cutoff:
            good_idxs = X < el_cutoff
            X, Y = X[good_idxs], Y[good_idxs]
        ax.plot(X, Y, "o", ms=0.8, rasterized=True)


def plot_l1_vs_stack(
    offset=True,
    alpha=300,
    h=3,
    w=4.5,
    yy=2018,
    unwfile="unw_stack_shiftedonly.h5",
    station="TXSO",
    days_smooth=1,
    save=False,
):
    # https://matplotlib.org/tutorials/introductory/customizing.html#a-sample-matplotlibrc-file
    # so obscure

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 16
    plt.rcParams["font.weight"] = "bold"
    years = mdates.YearLocator()  # every year
    # months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter("%Y")
    # yrs = (date(2015), date(2016), date(2017), date(2018))

    input_dset = "stack_flat_shifted"
    geolist, intlist, igram_idxs = load_geolist_intlist(
        h5file=unwfile,
        geolist_ignore_file="geolist_ignore.txt",
        max_temporal_baseline=800,
        max_date=date(yy, 1, 1),
    )
    Blin = np.sum(timeseries.prepB(geolist, intlist), axis=1, keepdims=1)

    timediffs = timeseries.find_time_diffs(geolist)
    unw_vals = get_stack_vals(
        unwfile,
        station_name=station,
        window=3,
        dset=input_dset,
        valid_indices=igram_idxs,
    )

    # Timeseries: unregularized, with all outliers (noisiest)
    B = timeseries.prepB(geolist, intlist, False, 0)
    print(f"{B.shape = }")
    # vv = np.linalg.lstsq(B, unw_vals, rcond=None)[0]
    vv = np.linalg.pinv(B) @ unw_vals
    unregged = timeseries.PHASE_TO_CM * timeseries.integrate_velocities(
        vv.reshape(-1, 1), timediffs
    )

    # Timeseries: regularized, but with all outliers
    Ba = timeseries.prepB(geolist, intlist, False, alpha)
    unw_vals_a = timeseries._augment_zeros(Ba, unw_vals)
    # vv_a = np.linalg.lstsq(Ba, unw_vals_a, rcond=None)[0]
    vv_a = np.linalg.pinv(Ba) @ unw_vals_a
    regged = timeseries.PHASE_TO_CM * timeseries.integrate_velocities(
        vv_a.reshape(-1, 1), timediffs
    )

    #
    # Plot timeseries with-outlier cases:
    ms = 4
    fig, ax = gps.plot_gps_los(
        station,
        end_date=date(yy, 1, 1),
        offset=offset,
        days_smooth=days_smooth,
        gps_color=MATLAB_COLORS[2],
        ms=ms,
    )

    ax.plot(geolist, unregged, "-x", lw=3, c=MATLAB_COLORS[3], label="Unregularized")
    ax.plot(geolist, regged, "-x", lw=3, c=MATLAB_COLORS[4], label="Regularized")
    ax.format_xdata = years_fmt
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    # ax.tick_params(axis="x", labelrotation=45)
    y0 = np.ceil(np.max(np.abs(unregged)))
    ax.set_ylim((-y0, y0))
    ax.set_yticks(np.arange(-y0, y0 + 2, step=2))
    _set_figsize(fig, h, w)
    fig.legend()
    if save:
        fig.savefig(
            f"compare_{yy}_timeseries.pdf",
            bbox_inches="tight",
            transparent=True,
            dpi=100,
        )

    # No outlier removal linear cases
    stack, l2, l1 = prunesolve(geolist, intlist, unw_vals, Blin, 1000, shrink=False)
    print(f"No outlier, linear: {stack}, {l1=}")
    print(f"Difference: {abs(stack - l1)}")

    # Plot linear with-outlier cases
    fig, ax = gps.plot_gps_los(
        station,
        end_date=date(yy, 1, 1),
        insar_mm_list=[l1, l2],
        offset=offset,
        labels=["L1 linear", "L2 linear"],
        gps_color=MATLAB_COLORS[2],
        insar_colors=MATLAB_COLORS[:2],
        days_smooth=days_smooth,
        ms=ms,
    )

    # ax.plot(geolist, regged, "-x", lw=3, label="reg")
    ax.plot(geolist, unregged, "-x", lw=3, c=MATLAB_COLORS[3], label="Unregularized")
    ax.plot(geolist, regged, "-x", lw=3, c=MATLAB_COLORS[4], label="Regularized")
    ax.format_xdata = years_fmt
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    # ax.tick_params(axis="x", labelrotation=45)
    # y0 = ceil(maximum(abs.(regged)))
    y0 = 6
    ax.set_ylim((-y0, y0))
    ax.set_yticks(np.arange(-y0, y0 + 2, step=2))
    _set_figsize(fig, h, w)
    fig.legend()
    if save:
        fig.savefig(
            f"compare_{yy}_linear.pdf", bbox_inches="tight", transparent=True, dpi=100
        )

    ###### Outlier remove cases ################
    # timeseries: regularized with outliers removed
    geo_clean, intlist_clean, unw_clean = remove_outliers(
        geolist, intlist, unw_vals, mean_sigma_cutoff=4
    )
    B2 = timeseries.prepB(geo_clean, intlist_clean, False, 0)
    td_clean = timeseries.find_time_diffs(geo_clean)
    unregged2 = timeseries.PHASE_TO_CM * timeseries.integrate_velocities(
        np.linalg.lstsq(B2, unw_clean, rcond=None)[0], td_clean
    )

    Ba2 = timeseries.prepB(geo_clean, intlist_clean, False, alpha)
    unw_vals_a2 = timeseries._augment_zeros(Ba2, unw_clean)
    regged2 = timeseries.PHASE_TO_CM * timeseries.integrate_velocities(
        np.linalg.lstsq(Ba2, unw_vals_a2, rcond=None)[0], td_clean
    )

    # linear solves with outlier removal
    stack2, l22, l12 = prunesolve(geolist, intlist, unw_vals, Blin, 4, shrink=False)

    # PLOT:
    fig, ax = gps.plot_gps_los(
        station,
        end_date=date(yy, 1, 1),
        insar_mm_list=[l12, l22],
        offset=offset,
        labels=["L1 linear", "L2 linear"],
        gps_color=MATLAB_COLORS[2],
        insar_colors=MATLAB_COLORS[:2],
        days_smooth=days_smooth,
        ms=ms,
    )
    # ax.plot(geo_clean, regged2, "-x", lw=3, label="reg")
    ax.plot(geo_clean, unregged2, "-x", c=MATLAB_COLORS[3], lw=3, label="Unregularized")
    ax.plot(geo_clean, regged2, "-x", lw=3, c=MATLAB_COLORS[4], label="Regularized")
    # y0 = ceil(maximum(abs.(regged2)))
    # y0 = 4
    ax.set_ylim((-y0, y0))
    ax.set_yticks(np.arange(-y0, y0 + 2, step=2))
    ax.format_xdata = years_fmt
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    _set_figsize(fig, h, w)
    fig.legend()
    if save:
        fig.savefig(
            f"compare_{yy}_removed.pdf", bbox_inches="tight", transparent=True, dpi=100
        )


# TODO: MOVE THESE!!!
def _set_figsize(fig, h=3, w=3.5):
    return (fig.set_figheight(h), fig.set_figwidth(w))


def load_geolist_intlist(
    igram_dir=None,
    geo_dir=None,
    h5file=None,
    geolist_ignore_file=None,
    igram_ext=".int",
    parse=True,
    min_date=None,
    max_date=None,
    max_temporal_baseline=None,
):
    import apertools.utils

    ifg_date_list = sario.load_intlist_from_h5(h5file, parse=parse)
    geo_date_list = sario.load_geolist_from_h5(h5file, parse=parse)

    if geolist_ignore_file is not None:
        ignore_filepath = os.path.join(igram_dir or ".", geolist_ignore_file)
        valid_geo_dates, valid_ifg_dates = sario.ignore_geo_dates(
            geo_date_list, ifg_date_list, ignore_file=ignore_filepath, parse=parse
        )

    valid_ifg_dates = apertools.utils.filter_min_max_date(
        valid_ifg_dates, min_date, max_date
    )
    if max_temporal_baseline is not None:
        ll = len(valid_ifg_dates)
        valid_ifg_dates = [
            ifg
            for ifg in valid_ifg_dates
            if abs((ifg[1] - ifg[0]).days) <= max_temporal_baseline
        ]
        logger.info(
            f"Ignoring {ll - len(valid_ifg_dates)} longer than {max_temporal_baseline}"
        )

    logger.info(f"Ignoring {len(ifg_date_list) - len(valid_ifg_dates)} igrams total")

    # Now just use the ones remaining to reform the geo dates
    valid_geo_dates = list(sorted(set(itertools.chain.from_iterable(valid_ifg_dates))))
    # valid_geo_idxs = np.searchsorted(geo_date_list, valid_geo_dates)
    valid_ifg_idxs = np.searchsorted(
        sario.intlist_to_filenames(ifg_date_list),
        sario.intlist_to_filenames(valid_ifg_dates),
    )
    # return valid_geo_idxs, valid_ifg_idxs
    return valid_geo_dates, valid_ifg_dates, valid_ifg_idxs


def get_stack_vals(
    unw_stack_file,
    row=None,
    col=None,
    valid_indices=None,
    station_name=None,
    window=5,
    dset=constants.STACK_FLAT_DSET,
    reference_station=None,
):
    dem_rsc = sario.load("dem.rsc")
    if station_name is not None:
        lon, lat = gps.station_lonlat(station_name)
        row, col = map(
            lambda x: int(x), latlon.nearest_pixel(dem_rsc, lon=lon, lat=lat)
        )

    unw_vals = _read_vals(
        unw_stack_file, row, col, valid_indices, window=window, dset=dset
    )
    return _subtract_reference(
        unw_vals,
        reference_station,
        unw_stack_file,
        window=window,
        dset=dset,
        valid_indices=valid_indices,
    )


def _read_vals(
    unw_stack_file,
    row,
    col,
    valid_indices,
    window=5,
    dset=constants.STACK_FLAT_DSET,
):
    # println("Loading $row, $col from $dset, avging window $window")
    halfwin = max(window // 2, 1)
    with h5py.File(unw_stack_file, "r") as hf:
        print(row, col, hf[dset].shape)
        unw_depth = hf[dset][
            :,
            row - halfwin : row + halfwin,
            col - halfwin : col + halfwin,
        ]
    unw_vals_all = np.mean(unw_depth, axis=(1, 2)).ravel()
    unw_vals = unw_vals_all[valid_indices]
    return unw_vals


def _subtract_reference(
    unw_vals,
    reference_station,
    unw_stack_file,
    window=1,
    dset=None,
    valid_indices=None,
):
    if reference_station is None:
        return unw_vals
    return unw_vals - get_stack_vals(
        unw_stack_file,
        valid_indices=valid_indices,
        window=window,
        dset=dset,
        reference_station=reference_station,
    )
