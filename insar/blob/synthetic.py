# coding: utf-8
import scipy.ndimage as nd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def make_delta(N, row=None, col=None):
    delta = np.zeros((N, N))
    if row is None or col is None:
        row, col = N // 2, N // 2
    delta[row, col] = 1
    return delta


def make_gaussian(N, sigma, row=None, col=None, normalize=False):
    delta = make_delta(N, row, col)
    out = nd.gaussian_filter(delta, sigma) * sigma**2
    return out / np.max(out) if normalize else out


def make_log(N, sigma, row=None, col=None, normalize=False):
    delta = make_delta(N, row, col)
    out = nd.gaussian_laplace(delta, sigma) * sigma**2
    return out / np.max(out) if normalize else out


GAUSSIAN = make_gaussian
LOG = make_log


def plot_func(func=GAUSSIAN, N=501, sigma=None):
    if sigma is None:
        sigma = N / 19
    f = func(N, sigma)
    xx = np.linspace(-sigma, sigma, N)
    X, Y = np.meshgrid(xx, xx)

    fig = plt.figure(frameon=False)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, f, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    plt.show()


def make_stack(N=501, max_amp=3):
    """Makes composite of 3 blob sizes, with small negative inside big positive"""
    b1 = make_gaussian(N, 100, None, None)
    b2 = make_gaussian(N, 30, N // 3, N // 3)
    b3 = make_gaussian(N, 7, 4 * N // 7, 4 * N // 7)
    # 3 little ones that merge to one
    b4 = make_gaussian(N, 17, 6 * N // 7, 6 * N // 7)
    b4 += make_gaussian(N, 17, 33 + 6 * N // 7, 6 * N // 7)
    out = b1 - b2 - .7 * b3 + .68 * b4
    out *= max_amp / np.max(out)

    fig = plt.figure()
    plt.imshow(out)
    plt.colorbar()
    return out, fig


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
