import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import skimage.feature
from insar.log import get_log

logger = get_log()


def shifted_color_map(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
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

    Returns:
        matplotlib.cmap
    """
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)

    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def make_shifted_cmap(img, cmap_name='seismic'):
    """Scales the colorbar so that 0 is always centered (white)"""
    midpoint = 1 - np.max(img) / (abs(np.min(img)) + np.max(img))
    return shifted_color_map(cmap_name, midpoint=midpoint)


def animate_stack(stack, pause_time=200, display=True, titles=None, save_title=None, **savekwargs):
    """Runs a matplotlib loop to show each image in a 3D stack

    Args:
        stack (ndarray): 3D np.ndarray, 1st index is image number
            i.e. the idx image is stack[idx, :, :]
        pause_time (float): Optional- time between images in milliseconds (default=200)
        display (bool): True if you want the plot GUI to pop up and run
            False would be if you jsut want to save the movie with as save_title
        titles (list[str]): Optional- Names of images corresponding to stack.
            Length must match stack's 1st dimension length
        save_title (str): Optional- if provided, will save the animation to a file
            extension must be a valid extension for a animation writer:
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

    # Use the same stack min and stack max for all colorbars/ color ranges
    minval, maxval = np.min(stack), np.max(stack)
    fig, ax = plt.subplots()
    axes_image = plt.imshow(stack[0, :, :], vmin=minval, vmax=maxval)  # Type: AxesImage

    cbar = fig.colorbar(axes_image)
    cbar_ticks = np.linspace(minval, maxval, num=6, endpoint=True)
    cbar.set_ticks(cbar_ticks)
    cbar.set_label("Centimeters")

    def update_im(idx):
        axes_image.set_data(stack[idx, :, :])
        fig.suptitle(titles[idx])
        return axes_image,

    stack_ani = animation.FuncAnimation(
        fig, update_im, frames=range(num_images), interval=pause_time, blit=False, repeat=True)

    if save_title:
        logger.info("Saving to %s", save_title)
        stack_ani.save(save_title, **savekwargs)

    if display:
        plt.show()


def rowcol_to_latlon(row, col, rsc_data=None):
    """ Takes the row, col of a pixel and finds its lat/lon

    Args:
        row (int): row number
        col (int): col number
        rsc_data (dict): data output from sario.load_dem_rsc

    Returns:
        tuple[float, float]: lat, lon for the pixel

    Example:
        >>> rsc_data = {"X_FIRST": 1.0, "Y_FIRST": 2.0, "X_STEP": 0.2, "Y_STEP": -0.1}
        >>> rowcol_to_latlon(7, 3, rsc_data)
        (1.4, 1.4)
    """
    start_lon = rsc_data["X_FIRST"]
    start_lat = rsc_data["Y_FIRST"]
    lon_step, lat_step = rsc_data["X_STEP"], rsc_data["Y_STEP"]
    lat = start_lat + (row - 1) * lat_step
    lon = start_lon + (col - 1) * lon_step
    return lat, lon


def view_stack(stack,
               geolist=None,
               display_img=-1,
               label="Centimeters",
               cmap='seismic',
               title='',
               lat_lon=True,
               rsc_data=None):
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

    shifted_cmap = make_shifted_cmap(img, cmap)
    axes_image = plt.imshow(img, cmap=shifted_cmap)  # Type: AxesImage
    title = title or "Deformation Time Series"  # Default title
    plt.title(title)

    cbar = imagefig.colorbar(axes_image)
    cbar.set_label(label)

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
            lat, lon = rowcol_to_latlon(row, col, rsc_data)
            legend_entries.append('Lat {:.3f}, Lon {:.3f}'.format(lat, lon))
        else:
            legend_entries.append('Row %s, Col %s' % (row, col))

        plt.plot(geolist, timeline, marker='o', linestyle='dashed', linewidth=1, markersize=4)
        plt.legend(legend_entries, loc='lower left')
        x_axis_str = "SAR image date" if geolist is not None else "Image number"
        plt.xlabel(x_axis_str)
        plt.ylabel(label)
        plt.show()

    imagefig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)


def find_blobs(image, blob_func='blob_log', **kwargs):
    """Use skimage to find blobs in image

    Args:
        image (ndarray): image containing blobs
        blob_func (str): which of the functions to use to find blobs
            Options: 'blob_log', 'blob_dog', 'blob_doh'

    Returns:
        ndarray: list of blobs: [(r, c, s)], r = row num of center,
        c is column, s is sigma (size of Gaussian that detected blob)

    Notes:
        kwargs can be passed to the blob_func. Examples extras are
        threshold (default=0.2, high=fewer blobs), min_sigma,
        max_sigma, num_sigma (except for blob_dog), overlap.
        See reference for full list

    Reference:
    [1] http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html
    """
    blob_func = getattr(skimage.feature, blob_func)
    return blob_func(image, **kwargs)


def plot_blobs(image, blobs=None, cur_axes=None, color='blue', **kwargs):
    """Takes the blob results from find_blobs and overlays on image

    Can either make new figure of plot on top of existing axes.
    """
    if not cur_axes:
        cur_fig = plt.figure()
        cur_axes = cur_fig.gca()
        cur_axes.imshow(image)

    if blobs is None:
        logger.info("Searching for blobs in image.")
        blobs = find_blobs(image, **kwargs)

    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, fill=False, linewidth=2, clip_on=False)
        cur_axes.add_patch(c)

    plt.show()
    return blobs


def get_blob_values(image, blobs):
    """Finds the image's value of each blob center"""
    coords = blobs[:, :2].astype(int)
    return image[coords[:, 0], coords[:, 1]]


def sort_blobs_by_val(image, blobs):
    blob_vals = get_blob_values(image, blobs)
    return sorted(zip(blobs, blob_vals), key=lambda tup: abs(tup[1]), reverse=True)
