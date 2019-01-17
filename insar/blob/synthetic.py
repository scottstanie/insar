# coding: utf-8
import scipy.ndimage as nd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from insar import blob
import ipdb


def generate_blobs(num_blobs,
                   imsize=(1000, 1000),
                   border_pad=100,
                   min_sigma=5,
                   max_sigma=80,
                   mean_sigma=10,
                   max_amp=5,
                   mean_amp=3,
                   min_amp=1,
                   amp_scale=3):
    # end columns are (x, y, sigma, Amplitude)
    # Uniformly spread blobs: first make size they lie within (pad, max-pad)
    rand_xy = np.random.rand(num_blobs, 2)
    rand_xy *= np.array([imsize[0] - 2 * border_pad, imsize[1] - 2 * border_pad])
    rand_xy += border_pad
    rand_xy = rand_xy.astype(int)

    sigmas = np.random.exponential(scale=mean_sigma, size=(num_blobs, ))
    sigmas += min_sigma
    sigmas = np.clip(sigmas, None, max_sigma)

    # Make larger sigma blobs have lower expected amplitude
    amp_means = 1 / sigmas
    amplitudes = np.random.exponential(scale=amp_means * amp_scale)
    amplitudes += min_amp
    amplitudes = np.clip(amplitudes, None, max_amp)
    signs = 2 * np.random.randint(0, 2, size=(num_blobs, )) - 1
    amplitudes *= signs

    # # Or just use exponential dist
    # amplitudes = np.random.exponential(scale=mean_amp)

    blobs = np.stack([rand_xy[:, 1], rand_xy[:, 0], sigmas, amplitudes], axis=1)

    out = np.zeros(imsize)
    for row, col, sigma, amp in blobs:
        # TODO: correct the N to be sizes
        out += make_gaussian(imsize[0], sigma, row=int(row), col=int(col), amp=amp)
    # convert blobs into (row, col, radius, amp) format
    return blobs * np.array([1, 1, np.sqrt(2), 1]), out




def calc_detection_stats(blobs_real, detected, min_iou=0.5):
    """Calculate how well find_blobs performed on synthetic blobs_real

    Uses a distance threshold and sigma margin to confirm same blobs
    Args:
        blobs_real (ndarray): actual blobs in image
        detected (ndarray): output from find_blobs
        min_iou (float): 0 to 1, minimum intersection/union between blobs
            to be called a detection

    Returns:
        true_detects (ndarray): blobs in detected which are real
        false_positives (ndarray): blobs in detected, not in real
        misses (ndarray): blobs in blobs_real, not detected
        precision (float): len(true_detection) / len(detections)
        recall (float): true_detections / len(blobs_real)

    Note: should be case that
        2*len(true_detects) + len(false_positives) + len(misses) == len(blobs_real) + len(detected)

    """

    true_detects = []
    false_positives = []
    matched_idx_set = set()
    for cur_clob in detected:
        # Find closest blob in 3d, check its overlap
        diffs_3d = (blobs_real - cur_clob)[:, :3]
        dist_3d = np.sqrt(np.sum(diffs_3d**2, axis=1))

        closest_idx = dist_3d.argmin()
        closest_dist = dist_3d[closest_idx]
        min_dist_real_blob = blobs_real[closest_idx, :]
        iou_frac = blob.skblob.intersection_over_union(cur_clob[:3], min_dist_real_blob[:3], using_sigma=False)
        is_overlapping = iou_frac > min_iou
        # ipdb.set_trace()
        # if np.any(matches):  # If we want to check all iou, maybe do this
        if is_overlapping:
            print('true (overlap, dist):', iou_frac, dist_3d[closest_idx])
            true_detects.append(cur_clob)
            # match_idxs = np.where(matches)[0]
            # matched_idx_set.update(match_idxs)
            matched_idx_set.add(closest_idx)
        else:
            print('false (overlap, dist):', iou_frac, dist_3d[closest_idx])
            false_positives.append(cur_clob)

    # Now find misses
    miss_idxs = set(np.arange(len(blobs_real))) - matched_idx_set
    misses = blobs_real[tuple(miss_idxs), :]

    precision = len(true_detects) / len(detected)
    recall = len(true_detects) / len(blobs_real)

    if 2 * len(true_detects) + len(false_positives) + len(misses) != len(blobs_real) + len(
        detected):
        print('CAUTION: weird num of true + fp + miss')
    return np.array(true_detects), np.array(false_positives), misses, precision, recall


def demo_ghost_blobs(min_iou=0.5):
    np.random.seed(1)
    finding_params = {
        'positive': True,
        'negative': True,
        'threshold': 0.35,
        'mag_threshold': None,
        'min_sigma': 3,
        'max_sigma': 60,
        'num_sigma': 50,
        'sigma_bins': 3,
    }
    real_blobs, out = generate_blobs(10, max_amp=15, amp_scale=25)
    detected, sigma_list = blob.find_blobs(out, **finding_params)

    true_d, fp, misses, precision, recall = calc_detection_stats(real_blobs, detected, min_iou=min_iou)
    print("Results:")
    print("precision: ", precision)
    print("recall: ", recall)

    _, cur_axes = blob.plot.plot_blobs(image=out, blobs=true_d, color='green')
    _, cur_axes = blob.plot.plot_blobs(image=out, blobs=fp, color='black', cur_axes=cur_axes)
    _, cur_axes = blob.plot.plot_blobs(image=out, blobs=misses, color='red', cur_axes=cur_axes)
    return out, sigma_list, real_blobs, detected, fp, misses


def make_delta(N, row=None, col=None):
    delta = np.zeros((N, N))
    if row is None or col is None:
        row, col = N // 2, N // 2
    delta[row, col] = 1
    return delta


def make_gaussian(N, sigma, row=None, col=None, normalize=False, amp=None):
    delta = make_delta(N, row, col)
    out = nd.gaussian_filter(delta, sigma) * sigma**2
    if normalize:
        return out / np.max(out) if normalize else out
    elif amp is not None:
        return amp * (out / np.max(out))
    else:
        return out


def make_log(N, sigma, row=None, col=None, normalize=False):
    delta = make_delta(N, row, col)
    out = nd.gaussian_laplace(delta, sigma) * sigma**2
    return out / np.max(out) if normalize else out


GAUSSIAN = make_gaussian
LOG = make_log


def plot_func(func=GAUSSIAN, N=501, sigma=None):
    if sigma is None:
        sigma = N / 19
    f = func(N, sigma)
    xx = np.linspace(-sigma, sigma, N)
    X, Y = np.meshgrid(xx, xx)

    fig = plt.figure(frameon=False)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, f, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    plt.show()


def make_stack(N=501, max_amp=3, cmap='jet'):
    """Makes composite of 3 blob sizes, with small negative inside big positive"""
    b1 = make_gaussian(N, 100, None, None)
    b2 = make_gaussian(N, 30, N // 3, N // 3)
    b3 = make_gaussian(N, 7, 4 * N // 7, 4 * N // 7)
    # little ones that merge to one
    # b4 = make_gaussian(N, 19, 6 * N // 7, 6 * N // 7)
    # b4 += make_gaussian(N, 19, 47 + 6 * N // 7, 6 * N // 7)
    b4 = make_gaussian(N, 19, 6 * N // 8, 6 * N // 8)
    b4 += make_gaussian(N, 19, 48 + 6 * N // 8, 6 * N // 8)
    b4 += make_gaussian(N, 19, 6 * N // 8, 48 + 6 * N // 8)
    b4 += make_gaussian(N, 19, 48 + 6 * N // 8, 48 + 6 * N // 8)
    out = b1 - b2 - .7 * b3 + .68 * b4
    out *= max_amp / np.max(out)

    fig = plt.figure()
    plt.imshow(out, cmap=cmap)
    plt.colorbar()
    return out, fig


def make_stack2(N=501, max_amp=3, cmap='jet'):
    """Simpler composite of 3 blob sizes, no superimposing"""
    b1 = make_gaussian(N, 60, N // 3, 2 * N // 3)
    b2 = make_gaussian(N, 20, N // 3, N // 3)

    b4 = make_gaussian(N, 29, 345, 345)
    b4 += make_gaussian(N, 29, 73 + 345, 345)
    b4 += make_gaussian(N, 29, 345, 73 + 345)
    b4 += make_gaussian(N, 29, 73 + 345, 73 + 345)
    out = b1 - .65 * b2 + .84 * b4
    out *= max_amp / np.max(out)

    fig = plt.figure()
    plt.imshow(out, cmap=cmap)
    plt.colorbar()
    return out, fig


# # ax.get_xaxis().set_visible(False)
# # ax.get_yaxis().set_visible(False)
# ax.w_zaxis.line.set_lw(0.)
# ax.set_zticks([])
# # ax.get_zaxis().set_visible(False)
#
# ax.set_xticks([])
# ax.set_yticks([])
# ax.grid(False)
# # fig.patch.set_visible(False)
#
# # fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_axis_off()
# # plt.show()


def igarss_fig():
    # out, fig = make_stack()
    out, fig = make_stack2()
    blobs, sigma_list = blob.find_blobs(out, min_sigma=3, max_sigma=100, num_sigma=40)
    image_cube = blob.skblob.create_gl_cube(out, sigma_list=sigma_list)

    _, cur_axes = blob.plot.plot_blobs(
        image=out,
        blobs=blob.find_edge_blobs(blobs, out.shape)[0],
        cur_axes=fig.gca(),
        color='blue')

    plt.imshow(image_cube[:, :, 30], cmap='jet', vmin=-1.4, vmax=1.3)
    plt.imshow(image_cube[:, :, 10], cmap='jet', vmin=-1.4, vmax=1.3)
    return out, blobs, sigma_list, image_cube, fig
