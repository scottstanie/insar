import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def display_stack(stack, pause_time=200, titles=None, save_title=None):
    """Runs a matplotlib loop to show each image in a 3D stack

    Args:
        stack (ndarray): 3D np.ndarray, 1st index is image number
            i.e. the idx image is stack[idx, :, :]
        pause_time (float): Optional- time between images in milliseconds (default=200)
        titles (list[str]): Optional- Names of images corresponding to stack.
            Length must match stack's 1st dimension length
        save_title (str): Optional- if provided, will save the animation to a .mp4 file

    Returns:
        None

    Notes: may need this
        https://github.com/matplotlib/matplotlib/issues/7759/#issuecomment-271110279
    """
    if titles:
        assert len(titles) == stack.shape[0], "len(titles) must equal stack.shape[0]"
    else:
        titles = itertools.repeat('')

    fig, ax = plt.subplots()
    image = plt.imshow(stack[0, :, :])  # Type: AxesImage
    fig.colorbar(image)

    def update_im(idx):
        image.set_data(stack[idx, :, :])
        fig.suptitle(titles[idx])
        return image,

    num_images = stack.shape[0]
    stack_ani = animation.FuncAnimation(
        fig, update_im, frames=range(num_images), interval=pause_time, blit=False)

    if save_title:
        stack_ani.save(save_title + '.mp4' if '.mp4' not in save_title else save_title)

    plt.show()
