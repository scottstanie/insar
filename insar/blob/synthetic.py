# coding: utf-8
import scipy.ndimage as nd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def make_delta(N):
    delta = np.zeros((N, N))
    delta[N // 2, N // 2] = 1
    return delta


def make_gaussian(N, sigma):
    delta = make_delta(N)
    return nd.gaussian_filter(delta, sigma)


def make_log(N, sigma):
    delta = make_delta(N)
    log = -nd.gaussian_laplace(delta, sigma)


GAUSSIAN = make_gaussian
LOG = make_log


def plot_func(func, N=501, sigma=None):
    if sigma is None:
        sigma = N / 19
    f = func(N, sigma)
    xx = np.linspace(-sigma, sigma, N)
    X, Y = np.meshgrid(xx, xx)

    fig = plt.figure(frameon=False)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, f, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    plt.show()


# # ax.get_xaxis().set_visible(False)
# # ax.get_yaxis().set_visible(False)
# ax.w_zaxis.line.set_lw(0.)
# ax.set_zticks([])
# # ax.get_zaxis().set_visible(False)
#
# ax.set_xticks([])
# ax.set_yticks([])
# ax.grid(False)
# # fig.patch.set_visible(False)
#
# # fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_axis_off()
# # plt.show()
