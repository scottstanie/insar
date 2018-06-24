import matplotlib.pyplot as plt
import matplotlib.animation as animation


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

    Notes: may need this
        https://github.com/matplotlib/matplotlib/issues/7759/#issuecomment-271110279
    """
    num_images = stack.shape[0]
    if titles:
        assert len(titles) == num_images, "len(titles) must equal stack.shape[0]"
    else:
        titles = ['' for _ in range(num_images)]

    fig, ax = plt.subplots()
    image = plt.imshow(stack[0, :, :])  # Type: AxesImage
    fig.colorbar(image)

    def update_im(idx):
        image.set_data(stack[idx, :, :])
        fig.suptitle(titles[idx])
        return image,

    stack_ani = animation.FuncAnimation(
        fig, update_im, frames=range(num_images), interval=pause_time, blit=False, repeat=False)

    if display:
        plt.show()

    if save_title:
        stack_ani.save(save_title, **savekwargs)


def explore_stack(stack, geolist, image_num=-1, title=""):
    """Displays an image from a stack, allows you to click for timeseries

    Args:
        stack (ndarray): 3D np.ndarray, 1st index is image number
            i.e. the idx image is stack[idx, :, :]
        geolist (list[datetime]): times of acquisition for each stack layer
        image_num (int): Optional- default = -1, the last image. Choose which
            image in the stack you want as the display to click on
        title (str): Optional- Title for plot

    Returns:
        None

    Notes: may need this
        See https://matplotlib.org/users/event_handling.html for click handling
    """

    def get_timeseries(row, col):
        return stack[:, row, col]

    fig = plt.figure(1)
    image = plt.imshow(stack[image_num, :, :])  # Type: AxesImage
    fig.colorbar(image)

    if not title:
        title = "Time series for pixel"

    def onclick(event):
        plt.figure(2)
        timeline = get_timeseries(int(event.ydata), int(event.xdata))
        print(int(event.ydata), int(event.xdata))
        print(timeline[-5:])

        plt.plot(geolist, timeline)
        plt.title(title)
        plt.show()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)
