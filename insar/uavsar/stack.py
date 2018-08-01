from copy import copy
import numpy as np
import insar
from insar.plotting import make_shifted_cmap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Used for a red mask overlaid on other image
colors = [(1, 0, 0, c) for c in np.linspace(0, 1, 100)]
cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)


def align_uavsar_images(image_list):
    """Aligns stack of images to first

    Args:
        image_list (list[str]): list of names of files from different dates
            over same acquisition area
    """
    uav_files = [insar.parsers.Uavsar(f) for f in image_list]
    # Align all to first acquisition date
    sorted_by_date = sorted(uav_files, key=lambda x: x.date)
    # IF WE WANT ALL POSSIBLE PAIRS:
    # Grab each pair of (earlier date, later date)
    # sorted_pairs = list(itertools.combinations(sorted_by_date, 2))
    loaded_imgs = [insar.sario.load(u.filename) for u in sorted_by_date]
    loaded_imgs = insar.utils.crop_to_smallest(loaded_imgs)

    first_ann = sorted_by_date[0].ann_data
    first_img = loaded_imgs[0]
    # Align all subsequent images to first
    out_images = [first_img]
    for uavsar, img in zip(sorted_by_date[1:], loaded_imgs[1:]):
        shifted_late = insar.utils.align_image_pair((first_img, img), (first_ann, uavsar.ann_data))
        out_images.append(shifted_late)
    return out_images


def make_uavsar_time_diffs(image_list):
    aligned_images = align_uavsar_images(image_list)
    # Mask out the zeros so we don't divide by zero
    masked_images = [np.ma.masked_equal(im, 0) for im in aligned_images]
    return [insar.utils.db(im / masked_images[0]) for im in masked_images[1:]]


def plot_uavsar_time_diffs(image_list):
    ratio_list = make_uavsar_time_diffs(image_list)
    cmaps = [make_shifted_cmap(r, cmap_name='seismic') for r in ratio_list]
    fig, axes = plt.subplots(1, len(ratio_list))
    if len(ratio_list) == 1:
        # Just so we can index correctly for the 1 subplot
        axes = [axes]
    for idx, ratio_im in enumerate(ratio_list):
        axes_im = axes[idx].imshow(ratio_im, cmap=cmaps[idx])
        fig.colorbar(axes_im, ax=axes[idx])

    return ratio_list, fig, axes


def overlay(under_image, over_image, under_image_info=None, ax=None, alpha=0.5):
    """Plots two images, one under, one transparent over

    under_image will be cmap=gray, over_image with
    a shifted cmap (seismic default)
    Assumes images are aligned at top left corner
    """
    if not ax:
        fig, ax = plt.subplots(1, 1)
    if under_image_info:
        under_extent = insar.utils.latlon_grid_extent(**under_image_info)
        xlabel, ylabel = 'longitude', 'latitude'
        # Now get extent of under image, which mage be different due to crop
        over_image_info = copy(under_image_info)
        over_image_info['rows'] = over_image.shape[0]
        over_image_info['cols'] = over_image.shape[1]
        over_extent = insar.utils.latlon_grid_extent(**over_image_info)
    else:
        # No lat/lon provided: jsut use row, col, no extend arg
        xlabel, ylabel = 'col number', 'row number'
        under_extent = over_extent = None
    ax.imshow(under_image, cmap='gray', extent=under_extent)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cmap = make_shifted_cmap(over_image, cmap_name='seismic')
    ax.imshow(over_image, cmap=cmap, alpha=0.5, extent=over_extent)
    return ax
