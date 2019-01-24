import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.qhull import ConvexHull
from insar.blob import utils as blob_utils
from insar import plotting


def plot_blobs(image=None,
               blobs=None,
               cur_fig=None,
               cur_axes=None,
               color='blue',
               blob_cmap=None,
               **kwargs):
    """Takes the blob results from find_blobs and overlays on image

    Can either make new figure of plot on top of existing axes.

    Returns:
        blobs
        cur_axes
    """
    if cur_fig:
        cur_axes = cur_fig.gca()
    if not cur_axes:
        cur_fig, ax_img = plotting.plot_image_shifted(
            image,
            fig=cur_fig,
            ax=cur_axes,
            **kwargs,
        )
        if not cur_axes:
            cur_axes = cur_fig.gca()
        # ax_img = cur_axes.imshow(image)
        # cur_fig.colorbar(ax_img)

    if blob_cmap:
        blob_cm = cm.get_cmap(blob_cmap, len(blobs))
    patches = []
    for idx, blob in enumerate(blobs):
        if blob_cmap:
            color_pct = idx / len(blobs)
            color = viridis(color_pct)
        c = plt.Circle((blob[1], blob[0]),
                       blob[2],
                       color=color,
                       fill=False,
                       linewidth=2,
                       clip_on=False)
        patches.append(c)
        cur_axes.add_patch(c)

    plt.draw()
    plt.show()
    return blobs, cur_axes


def plot_cropped_blob(image=None, blob=None, patch=None, crop_val=np.nan, sigma=0):
    """Plot a 3d view of heighs of a blob along with its circle imshow view

    Args:
        image (ndarray): image in which blobs are detected
        blob: (row, col, radius, ...)
        patch (ndarray): optional: the sub-image from `crop_blob`, which is
            the area of `image` cropped around `blob`
        crop_val (float or nan): value to make all pixels outside sigma radius
            default=np.nan. if None, leaves the edges of bbox untouched
        sigma (float): if provided, smooth by a gaussian filter of size `sigma`

    Returns:
        matplotlib.Axes
    """
    if patch is None:
        patch = blob_utils.crop_blob(image, blob, crop_val=crop_val, sigma=sigma)
    ax = plot_heights(patch)
    return ax


# TODO: export this? seems useful
def plot_heights(heights_grid):
    """Makes default X, Y meshgrid to plot a surface of heights"""
    rows, cols = heights_grid.shape
    xx = np.linspace(1, cols + 1, cols)
    yy = np.linspace(1, rows + 1, rows)
    X, Y = np.meshgrid(xx, yy)
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, heights_grid)
    ax2 = fig.add_subplot(1, 2, 2)
    axim = ax2.imshow(heights_grid)
    fig.colorbar(axim, ax=ax2)
    return ax


def plot_hist(H, row_edges, col_edges, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    axes_image = ax.imshow(H, extent=[col_edges[0], col_edges[-1], row_edges[-1], row_edges[0]])
    fig.colorbar(axes_image)
    return fig, ax


def scatter_blobs(blobs, image=None, axes=None, color='b', label=None):
    if axes is None:
        fig, axes = plt.subplots(1, 3)
    else:
        fig = axes[0].get_figure()

    if blobs.shape[1] < 6:
        blobs = blob_utils.append_stats(blobs, image)

    print('Taking abs value of blobs')
    blobs = np.abs(blobs)

    # Size vs amplitude
    sizes = blobs[:, 2]
    mags = blobs[:, 3]
    vars_ = blobs[:, 4]
    ptps = blobs[:, 5]

    axes[0].scatter(sizes, mags, c=color, label=label)
    axes[0].set_xlabel("Size")
    axes[0].set_ylabel("Magnitude")
    if label:
        axes[0].legend()

    axes[1].scatter(sizes, vars_, c=color, label=label)
    axes[1].set_xlabel("Size")
    axes[1].set_ylabel("variance")

    axes[2].scatter(sizes, ptps, c=color, label=label)
    axes[2].set_xlabel("Size")
    axes[2].set_ylabel("peak-to-peak")
    return fig, axes


def scatter_blobs_3d(blobs, image=None, ax=None, color='b', label=None, blob_img=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        fig = ax.get_figure()

    if blobs.shape[1] < 6:
        blobs = blob_utils.append_stats(blobs, image)

    if blob_img is not None:
        # Length of radii in km
        sizes = blob_img.pixel_to_km(blobs[:, 2])
    else:
        sizes = blobs[:, 2]
    mags = blobs[:, 3]
    vars_ = blobs[:, 4]
    ax.scatter(sizes, mags, vars_, c=color, label=label)
    ax.set_title("Size, mag, var of blobs")
    ax.set_xlabel('size')
    ax.set_ylabel('magniture')
    ax.set_zlabel('variance')
    return fig, ax


def plot_hull(regions=None, hull=None, ax=None, linecolor='k-'):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if hull is None:
        hull = ConvexHull(regions)
    for simplex in hull.simplices:
        ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], linecolor)


def plot_bbox(bbox, ax=None, linecolor='k-', cv_format=False):
    for c in blob_utils.bbox_to_coords(bbox, cv_format=cv_format):
        print(c)
        ax.plot(c[0], c[1], 'rx', markersize=6)


def plot_regions(regions, ax=None, linecolor='k-'):
    for shape in blob_utils.regions_to_shapes(regions):
        xx, yy = shape.convex_hull.exterior.xy
        ax.plot(xx, yy, linecolor)
