from skimage import feature
from skimage.transform import probabilistic_hough_line
from skimage.morphology import skeletonize
from skimage.feature import (
    hessian_matrix,
    hessian_matrix_eigvals,
    structure_tensor_eigenvalues,
    structure_tensor,
)
import numpy as np
import matplotlib.pyplot as plt


def get_hessian_eigs(img, sigma):
    H_elems = hessian_matrix(img, sigma=sigma)
    eigs = hessian_matrix_eigvals(H_elems)
    return eigs


def get_structure_eigs(img, sigma):
    """https://en.wikipedia.org/wiki/Structure_tensor"""
    H_elems = structure_tensor(img, sigma=sigma)
    return structure_tensor_eigenvalues(H_elems)


def coh(st):
    return ((st[0] - st[1]) / np.sum(st, axis=0)) ** 2


def get_structure_coh(img, sigma):
    """https://en.wikipedia.org/wiki/Structure_tensor#Interpretation"""
    return coh(get_structure_eigs(img, sigma))


def plot_eigs(
    img,
    sigma,
    eig_pctile=99,
    func="hessian",
    figsize=(7, 3),
    imcmap="seismic_wide_y",
    outfile=None,
    **imshow_kwargs,
):
    if func == "hessian":
        c = get_hessian_eigs(img, sigma)[0]
    elif func == "structure":
        c = get_structure_coh(img, sigma)

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=figsize)
    axes = axes.ravel()

    ax = axes[0]
    axim = ax.imshow(img, vmax=5, vmin=-5, cmap=imcmap)
    fig.colorbar(axim, ax=ax)

    ax = axes[1]
    vm = np.nanpercentile(np.abs(c), eig_pctile)
    # axim = ax.imshow(c, vmin=-vm, vmax=vm, **imshow_kwargs)
    axim = ax.imshow(np.abs(c), vmin=0, vmax=vm, **imshow_kwargs)
    fig.colorbar(axim, ax=ax)

    ax = axes[-2]
    cutoff = np.nanpercentile(np.abs(c), 90)
    bin_img = c > cutoff
    skele = skeletonize(bin_img)
    axim = ax.imshow(skele)

    ax = axes[-1]
    plot_lines(bin_img, ax=ax)

    if outfile:
        import rasterio as rio

        with rio.open(
            outfile,
            "w",
            width=c.shape[1],
            height=c.shape[0],
            dtype="float32",
            driver="GTiff",
            count=1,
        ) as dst:
            dst.write(c, 1)
    return c, axes


def plot_structure(
    img,
    sigma,
    eig_pctile=99,
    figsize=(7, 3),
    imcmap="seismic_wide_y",
    **imshow_kwargs,
):
    eigs = get_structure_eigs(img, sigma)
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=figsize)
    axes = axes.ravel()

    axim = axes[0].imshow(img, vmax=5, vmin=-5, cmap=imcmap)
    fig.colorbar(axim, ax=axes[0])

    ax = axes[1]
    c = eigs[0]
    # c = feature.canny(eigs[0] / eigs[1], sigma)
    cc = eigs[0] / eigs[1]
    skele = skeletonize(cc > 10)
    ax.imshow(skele)
    # plot_lines(cc > 10)

    # vm = np.nanpercentile(np.abs(c), eig_pctile)
    # axim = ax.imshow(c, vmin=0, vmax=vm, **imshow_kwargs)
    # fig.colorbar(axim, ax=ax)
    # ax.set_title(r"$\lambda_1$")

    # c = eigs[1]
    c = eigs[0] / eigs[1]
    vm = np.nanpercentile(np.abs(c), eig_pctile)
    ax = axes[2]
    axim = ax.imshow(c, vmin=0, vmax=vm, **imshow_kwargs)
    fig.colorbar(axim, ax=ax)
    ax.set_title(r"$\lambda_1 / \lambda_2$")


def plot_lines(bin_img, thresh=10, line_length=25, line_gap=10, ax=None):
    skele = skeletonize(bin_img)
    lines = probabilistic_hough_line(
        skele, threshold=thresh, line_length=line_length, line_gap=line_gap
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(bin_img * 0)
    for line in lines:
        p0, p1 = line
        ax.plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax.set_xlim((0, bin_img.shape[1]))
    ax.set_ylim((bin_img.shape[0], 0))
    ax.set_title("Probabilistic Hough")
