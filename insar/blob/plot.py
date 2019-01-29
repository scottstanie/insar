import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.qhull import ConvexHull
from insar.blob import utils as blob_utils
from insar import plotting


def on_pick(blobs, patches):
    def pick_event(event):
        """Store the index matching the clicked blob to delete"""
        ax = event.artist.axes
        for i, artist in enumerate(patches):
            if event.artist == artist:
                ax.picked_idx = i
                ax.picked_object = artist  # Also save circle Artist to remove

        print("Selected blob: %s" % str(blobs[ax.picked_idx]))

    return pick_event


def on_press(event):
    'on button press we will see if the mouse is over us and store some data'
    # print("on press event", event)
    # You can either double click or right click to unselect
    if event.button != 3 and not event.dblclick:
        return

    ax = event.inaxes
    if ax:
        print("Unselecting blob")
        ax.picked_idx = None
        ax.picked_object = None


def on_key(event):
    """
    Function to be bound to the key press event
    If the key pressed is delete and there is a picked object,
    remove that object from the canvas
    """
    if event.key == u'delete':
        ax = event.inaxes
        if ax is not None and ax.picked_object:
            cur_blob = ax.blobs[ax.picked_idx]
            print("Deleting blob %s" % str(cur_blob))
            ax.deleted_idxs.add(ax.picked_idx)
            ax.picked_idx = None

            ax.picked_object.remove()
            ax.picked_object = None
            ax.figure.canvas.draw()


def plot_blobs(image=None,
               blobs=None,
               fig=None,
               ax=None,
               color='blue',
               blob_cmap=None,
               plot_img=False,
               delete=False,
               **kwargs):
    """Takes the blob results from find_blobs and overlays on image

    Can either make new figure of plot on top of existing axes.

    Returns:
        blobs
        ax
    """
    if fig and not ax:
        ax = fig.gca()
    if plot_img or not ax:
        fig, ax_img = plotting.plot_image_shifted(image, fig=fig, ax=ax, **kwargs)
        # ax_img = ax.imshow(image)
        # fig.colorbar(ax_img)

    if not ax:
        ax = fig.gca()
    elif not fig:
        fig = ax.figure

    if blob_cmap:
        blob_cm = cm.get_cmap(blob_cmap, len(blobs))
    patches = []
    # Draw big blobs first to allow easier clicking on overlaps
    sorted_blobs = sorted(blobs, key=lambda b: b[2], reverse=True)
    for idx, blob in enumerate(sorted_blobs):
        if blob_cmap:
            color_pct = idx / len(blobs)
            color = blob_cm(color_pct)
        c = plt.Circle((blob[1], blob[0]),
                       blob[2],
                       color=color,
                       fill=False,
                       linewidth=2,
                       clip_on=False,
                       picker=True)
        ax.add_patch(c)
        patches.append(c)

    remaining_blobs = blobs
    plt.draw()
    if delete is False:
        return blobs, ax

    ax.blobs = sorted_blobs
    ax.picked_idx = None
    ax.picked_object = None
    ax.deleted_idxs = set()

    pick_handler = on_pick(sorted_blobs, patches)
    cid_pick = fig.canvas.mpl_connect('pick_event', pick_handler)
    cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

    if ax.deleted_idxs:
        print("Deleted %s blobs" % len(ax.deleted_idxs))
        all_idx = range(len(blobs))
        remaining = list(set(all_idx) - set(ax.deleted_idxs))
        remaining_blobs = np.array(sorted_blobs)[remaining]
    else:
        remaining_blobs = blobs

    fig.canvas.mpl_disconnect(cid_pick)
    fig.canvas.mpl_disconnect(cid_press)
    fig.canvas.mpl_disconnect(cid_key)

    return remaining_blobs, ax


def plot_cropped_blob(image=None, blob=None, patch=None, crop_val=None, sigma=0):
    """Plot a 3d view of heighs of a blob along with its circle imshow view

    Args:
        image (ndarray): image in which blobs are detected
        blob: (row, col, radius, ...)
        patch (ndarray): optional: the sub-image from `crop_blob`, which is
            the area of `image` cropped around `blob`
        crop_val (float or nan): value to make all pixels outside sigma radius
            e.g. np.nan. if None, leaves the edges of bbox untouched
        sigma (float): if provided, smooth by a gaussian filter of size `sigma`

    Returns:
        matplotlib.Axes
    """
    if patch is None:
        patch = blob_utils.crop_blob(image, blob, crop_val=crop_val, sigma=sigma)
    elif sigma > 0:
        patch = blob_utils.gaussian_filter_nan(patch, sigma=sigma)
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


if __name__ == '__main__':
    npz = np.load('patches/image_1.npz')
    image = npz['image']
    real_blobs = npz['real_blobs']
    plot_blobs(image=image, blobs=real_blobs)
