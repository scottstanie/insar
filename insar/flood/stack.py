from copy import copy
import numpy as np
from insar import utils, sario, parsers, plotting
from insar.plotting import make_shifted_cmap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# Used for a red mask overlaid on other image
colors = [(1, 0, 0, c) for c in np.linspace(0, 1, 100)]
cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)


def align_uavsar_images(image_list):
    """Aligns stack of images to first

    Args:
        image_list (list[str]): list of names of files from different dates
            over same acquisition area
    """
    uav_files = [parsers.Uavsar(f) for f in image_list]
    # Align all to first acquisition date
    sorted_by_date = sorted(uav_files, key=lambda x: x.date)
    # IF WE WANT ALL POSSIBLE PAIRS:
    # Grab each pair of (earlier date, later date)
    # sorted_pairs = list(itertools.combinations(sorted_by_date, 2))
    loaded_imgs = [sario.load(u.filename) for u in sorted_by_date]
    loaded_imgs = utils.crop_to_smallest(loaded_imgs)

    first_ann = sorted_by_date[0].ann_data
    first_img = loaded_imgs[0]
    # Align all subsequent images to first
    out_images = [first_img]
    for uavsar, img in zip(sorted_by_date[1:], loaded_imgs[1:]):
        shifted_late = utils.align_image_pair((first_img, img), (first_ann, uavsar.ann_data))
        out_images.append(shifted_late)
    return out_images


def make_uavsar_time_diffs(image_list):
    aligned_images = align_uavsar_images(image_list)
    # Mask out the zeros so we don't divide by zero
    masked_images = [utils.mask_zeros(im) for im in aligned_images]
    return [utils.db(im / masked_images[0]) for im in masked_images[1:]]


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


def overlay(under_image, over_image, under_image_info=None, ax=None, alpha=0.5, title=''):
    """Plots two images, one under, one transparent over

    under_image will be cmap=gray, over_image with shifted cmap, default=seismic
    Assumes images are aligned at top left corner

    Returns:
        AxesSubplot: axes of the overlaid image
        under_extend (tuple): tuple used for extent keyword for under_image
            This is None if under_image_info give
    """
    if not ax:
        fig, ax = plt.subplots(1, 1)

    if under_image_info:
        under_extent = utils.latlon_grid_extent(**under_image_info)
        xlabel, ylabel = 'Longitude', 'Latitude'
        # Now get extent of under image, which mage be different due to crop
        over_image_info = copy(under_image_info)
        over_image_info['rows'] = over_image.shape[0]
        over_image_info['cols'] = over_image.shape[1]
        over_extent = utils.latlon_grid_extent(**over_image_info)
    else:
        # No lat/lon provided: jsut use row, col, no extend arg
        xlabel, ylabel = 'col number', 'row number'
        nrows, ncols = under_image.shape
        under_extent = (0, ncols, nrows, 0)
        nrows, ncols = over_image.shape
        over_extent = (0, ncols, nrows, 0)

    ax.imshow(under_image, cmap='gray', extent=under_extent)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cmap = make_shifted_cmap(over_image, cmap_name='seismic')
    ax.imshow(over_image, cmap=cmap, alpha=0.5, extent=over_extent)
    ax.set_title(title)
    return ax, under_extent, over_extent


def zoomed_box(under,
               over,
               zoom_bbox,
               ax=None,
               under_image_info=None,
               cmap='gray',
               zoom=500,
               loc='upper right',
               title=''):
    """
    Args:
        zoom_bbox (tuple): (left, bottom, width, height)

    Notes:
        locations: 1, 2, 3, 4 = 'upper right', 'upper left', 'lower left', 'lower right'
    """
    # TODO: how to recreate this with multiple imshow steps??
    # Ideally pass in the axins
    ax, under_extent, over_extent = overlay(
        under, over, under_image_info=under_image_info, ax=ax, title=title)
    # ax.imshow(under, cmap='gray')
    # ax.imshow(over, cmap=cmap, alpha=0.5)
    axins = zoomed_inset_axes(ax, zoom, loc=loc)

    # To zoom, we plot the whole thing again, then change axis limits
    print(under_extent)
    print(over_extent)
    plt.show(block=False)
    import pdb
    pdb.set_trace()

    axins.imshow(under, cmap='gray', extent=under_extent)
    axins.imshow(over, cmap=cmap, alpha=0.5, extent=over_extent)

    print(zoom_bbox)
    x0, y0, width, height = zoom_bbox
    axins.set_xlim(x0, x0 + width)
    axins.set_ylim(y0 + height, y0)

    # TODO: making this pick locations based on 'loc'?
    # Note: fc = facecolor, ec = edgecolor
    _patch, pp1, pp2 = mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")
    # Hack for inverted axes (in image), must flip manually
    # https://stackoverflow.com/a/48987903/4174466
    pp1.loc1, pp1.loc2 = 2, 3
    pp2.loc1, pp2.loc2 = 3, 2

    # plt.draw()
    # plt.show(block=True)
    return ax, axins, pp1, pp2


def demo_zoom():
    import glob
    hhfiles = glob.glob("/home/scott/Documents/Learning/research/uav-sar-data/grd/*14937*HHHH*.grd")
    ratio_list = make_uavsar_time_diffs(hhfiles)

    under = plotting.equalize_and_mask(sario.load(hhfiles[0]), fill_value=0.0)
    over = ratio_list[0]
    cmap = make_shifted_cmap(over, cmap_name='seismic')
    under_image_info = parsers.Uavsar(hhfiles[0]).ann_data

    bbox_ll = (-96.17, 30.05, 0.05, 0.07)

    zoomed_box(under, over, bbox_ll, under_image_info=under_image_info, cmap=cmap)
