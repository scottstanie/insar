from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import numpy as np
import matplotlib.pyplot as plt


def plot_hess_eig(
    img,
    sigma,
    eig_pctile=99,
    figsize=(7, 3),
    imcmap="seismic_wide_y",
    outfile=None,
    **imshow_kwargs,
):
    H_elems = hessian_matrix(img, sigma=sigma)
    eigs = hessian_matrix_eigvals(H_elems)[0]
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=figsize)
    axim = axes[0].imshow(img, vmax=5, vmin=-5, cmap=imcmap)
    fig.colorbar(axim, ax=axes[0])
    vm = np.nanpercentile(np.abs(eigs), eig_pctile)
    axim = axes[1].imshow(eigs, vmin=-vm, vmax=vm, **imshow_kwargs)
    fig.colorbar(axim, ax=axes[1])

    if outfile:
        import rasterio as rio

        with rio.open(
            outfile,
            "w",
            width=eigs.shape[1],
            height=eigs.shape[0],
            dtype="float32",
            driver="GTiff",
            count=1,
        ) as dst:
            dst.write(eigs, 1)
    return eigs, axes
