"""plotting.py: functions for visualizing insar products
"""
from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from insar.log import get_log
from insar import utils, latlon

logger = get_log()


def shifted_color_map(cmap, start=0, midpoint=0.5, stop=1.0, num_levels=None):
    """Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Attribution: https://stackoverflow.com/a/20528097, Paul H

    Args:
      cmap (str or matplotlib.cmap): The matplotlib colormap to be altered.
          Can be matplitlib.cm.seismic or 'seismic'
      start (float): Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint (float): The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop (float): Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
      num_levels (int): form fewer discrete levels in the colormap

    Returns:
        matplotlib.cmap
    """
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap, num_levels)

    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

    # regular index to compute the colors
    # N = num_levels
    N = 256
    reg_index = np.linspace(start, stop, N + 1)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, N // 2, endpoint=False),
        np.linspace(midpoint, 1.0, N // 2 + 1, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap('shiftedcmap', cdict, N=num_levels or N)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def make_shifted_cmap(img=None, maxval=None, minval=None, cmap_name='seismic', num_levels=None):
    """Scales the colorbar so that 0 is always centered (white)"""
    if img is not None:
        maxval, minval = np.max(img), np.min(img)
    if maxval is None or minval is None:
        raise ValueError("Required args: img, or maxval and minval")
    midpoint = 1 - maxval / (abs(minval) + maxval)
    print(cmap_name)
    return shifted_color_map(cmap_name, midpoint=midpoint, num_levels=num_levels)


def discrete_seismic_colors():
    return list(
        np.array([
            (202, 0, 32),
            (244, 165, 130),
            (247, 247, 247),
            (146, 197, 222),
            (5, 113, 176),
        ]) / 256)


DISCRETE_SEISMIC = matplotlib.colors.LinearSegmentedColormap.from_list(
    'discrete_seismic', discrete_seismic_colors(), N=len(discrete_seismic_colors()))
plt.register_cmap(cmap=DISCRETE_SEISMIC)


def plot_image_shifted(img,
                       fig=None,
                       cmap='seismic',
                       img_data=None,
                       title='',
                       label='',
                       xlabel='',
                       ylabel='',
                       perform_shift=True):
    """Plot an image with a zero-shifted colorbar

    Args:
        img (ndarray): 2D numpy array to imshow
        fig (matplotlib.Figure): Figure to plot image onto
        ax (matplotlib.AxesSubplot): Axes to plot image onto
            mutually exclusive with fig option
        cmap (str): name of colormap to shift
        img_data (dict): rsc_data from load_dem_rsc containing lat/lon
            data about image, used to make axes into lat/lon instead of row/col
        title (str): Title for image
        label (str): label for colorbar
        perform_shift (bool): default True. If false, skip cmap shifting step
    """
    if img_data:
        extent = latlon.grid_extent(**img_data)
    else:
        nrows, ncols = img.shape
        extent = (0, ncols, nrows, 0)

    if not fig:
        fig = plt.figure()
    ax = fig.gca()
    shifted_cmap = make_shifted_cmap(img, cmap_name=cmap) if perform_shift else cmap
    axes_image = ax.imshow(img, cmap=shifted_cmap, extent=extent)  # Type: AxesImage
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cbar = fig.colorbar(axes_image)
    cbar.set_label(label)
    return fig, axes_image


def animate_stack(stack,
                  pause_time=200,
                  display=True,
                  titles=None,
                  label=None,
                  save_title=None,
                  cmap_name='seismic',
                  shifted='True',
                  vmin=None,
                  vmax=None,
                  **savekwargs):
    """Runs a matplotlib loop to show each image in a 3D stack

    Args:
        stack (ndarray): 3D np.ndarray, 1st index is image number
            i.e. the idx image is stack[idx, :, :]
        pause_time (float): Optional- time between images in milliseconds (default=200)
        display (bool): True if you want the plot GUI to pop up and run
            False would be if you jsut want to save the movie with as save_title
        titles (list[str]): Optional- Names of images corresponding to stack.
            Length must match stack's 1st dimension length
        label (str): Optional- Label for the colorbar
        save_title (str): Optional- if provided, will save the animation to a file
            extension must be a valid extension for a animation writer:
        cmap_name (str): Name of matplotlib colormap
        shifted (bool): default true: shift the colormap to be 0 centered
        vmin (float): min value passed to imshow
        vmax (float): max value passed to imshow
        savekwargs: extra keyword args passed to animation.save
            See https://matplotlib.org/api/_as_gen/matplotlib.animation.Animation.html
            and https://matplotlib.org/api/animation_api.html#writer-classes

    Returns:
        None
    """
    num_images = stack.shape[0]
    if titles:
        assert len(titles) == num_images, "len(titles) must equal stack.shape[0]"
    else:
        titles = ['' for _ in range(num_images)]  # blank titles, same length
    if np.iscomplexobj(stack):
        stack = np.abs(stack)

    # Use the same stack min and stack max (or vmin/vmax) for all colorbars/ color ranges
    minval = vmin or np.min(stack)
    maxval = vmax or np.max(stack)
    cmap = cmap_name if not shifted else make_shifted_cmap(
        minval=minval, maxval=maxval, cmap_name=cmap_name)

    fig, ax = plt.subplots()
    axes_image = plt.imshow(stack[0, :, :], vmin=minval, vmax=maxval, cmap=cmap)

    cbar = fig.colorbar(axes_image)
    cbar_ticks = np.linspace(minval, maxval, num=6, endpoint=True)
    cbar.set_ticks(cbar_ticks)
    if label:
        cbar.set_label(label)

    def update_im(idx):
        axes_image.set_data(stack[idx, :, :])
        fig.suptitle(titles[idx])
        return axes_image,

    stack_ani = animation.FuncAnimation(
        fig, update_im, frames=range(num_images), interval=pause_time, blit=False, repeat=True)

    if save_title:
        logger.info("Saving to %s", save_title)
        stack_ani.save(save_title, writer='imagemagick', **savekwargs)

    if display:
        plt.show()


def view_stack(
        stack,
        geolist=None,
        display_img=-1,
        label="Centimeters",
        cmap='seismic',
        title='',
        lat_lon=True,
        rsc_data=None,
        row_start=0,
        row_end=-1,
        col_start=0,
        col_end=-1,
):
    """Displays an image from a stack, allows you to click for timeseries

    Args:
        stack (ndarray): 3D np.ndarray, 1st index is image number
            i.e. the idx image is stack[idx, :, :]
        geolist (list[datetime]): Optional: times of acquisition for
            each stack layer. Used as xaxis if provided
        display_img (int, str): Optional- default = -1, the last image.
            Chooses which image in the stack you want as the display
            display_img = 'avg' will take the average across all images
        label (str): Optional- Label on colorbar/yaxis for plot
            Default = Centimeters
        cmap (str): Optional- colormap to display stack image (default='seismic')
        title (str): Optional- Title for plot
        lat_lon (bool): Optional- Use latitude and longitude in legend
            If False, displays row/col of pixel
        rsc_data (dict): Optional- if lat_lon=True, data to calc the lat/lon

    Returns:
        None

    Raises:
        ValueError: if display_img is not an int or the string 'mean'

    """
    stack = stack[:, row_start:row_end, col_start:col_end]
    # If we don't have dates, use indices as the x-axis
    if geolist is None:
        geolist = np.arange(stack.shape[0])

    if lat_lon and not rsc_data:
        raise ValueError("rsc_data is required for lat_lon=True")

    def get_timeseries(row, col):
        return stack[:, row, col]

    imagefig = plt.figure()

    if isinstance(display_img, int):
        img = stack[display_img, :, :]
    elif display_img == 'mean':
        img = np.mean(stack, axis=0)
    else:
        raise ValueError("display_img must be an int or 'mean'")

    title = title or "Deformation Time Series"  # Default title
    plot_image_shifted(
        img, fig=imagefig, title=title, cmap=DISCRETE_SEISMIC, label=label, perform_shift=False)

    timefig = plt.figure()

    plt.title(title)
    legend_entries = []

    def onclick(event):
        # Ignore right/middle click, clicks off image
        if event.button != 1 or not event.inaxes:
            return
        plt.figure(timefig.number)
        row, col = int(event.ydata), int(event.xdata)
        try:
            timeline = get_timeseries(row, col)
        except IndexError:  # Somehow clicked outside image, but in axis
            return

        if lat_lon:
            lat, lon = latlon.rowcol_to_latlon(row, col, rsc_data)
            legend_entries.append('Lat {:.3f}, Lon {:.3f}'.format(lat, lon))
        else:
            legend_entries.append('Row %s, Col %s' % (row, col))

        plt.plot(geolist, timeline, marker='o', linestyle='dashed', linewidth=1, markersize=4)
        plt.legend(legend_entries, loc='upper left')
        x_axis_str = "SAR image date" if geolist is not None else "Image number"
        plt.xlabel(x_axis_str)
        plt.ylabel(label)
        plt.show()

    imagefig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)


def equalize_and_mask(image, low=1e-6, high=2, fill_value=np.inf, db=True):
    """Clips an image to increase contrast"""
    # Mask the invalids, then mask zeros, then clip rest
    im = np.clip(utils.mask_zeros(np.ma.masked_invalid(image)), low, high)
    if fill_value:
        im.set_fill_value(fill_value)
    return utils.db(im) if db else im
