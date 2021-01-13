import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import h5py
from apertools import sario


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


def plot_phase_vs_elevation(dem=None,
                            unw_stack=None,
                            outname="phase_vs_elevation_noraster.pdf",
                            igram_idx=210,
                            el_cutoff=1200,
                            to_cm=True,
                            rasterized=True):
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
        Y *= .44

    plt.style.use('default')
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
    ax.plot(X, Y, 'o', rasterized=rasterized, ms=0.8)
    # ax.set_xlim(700, 1200)
    ax.set_xlim(None, 1200)
    ax.set_xlabel("Elevation (m)")
    # ax.set_ylabel("Phase (rad)")
    ax.set_ylabel("[cm]")
    plt.show(block=False)
    fig.savefig('phase_vs_elevation_noraster.pdf', dpi=200, transparent=True)
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
        ax.plot(X, Y, 'o', ms=0.8, rasterized=True)
