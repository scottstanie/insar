# coding: utf-8
import os
import sys
import itertools
from collections import defaultdict
import glob
import numpy as np
import scipy.ndimage as nd
from scipy.stats import multivariate_normal
from skimage import feature, transform
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

from insar import blob
import insar.blob.scores as scores
import insar.blob.utils as blob_utils
import ipdb
from insar.log import log_runtime


@log_runtime
def generate_blobs(num_blobs,
                   imsize=(1000, 1000),
                   border_pad=100,
                   min_sigma=6,
                   max_sigma=80,
                   mean_sigma=10,
                   max_amp=5,
                   mean_amp=3,
                   min_amp=1,
                   max_ecc=0.5,
                   amp_scale=3,
                   noise_sigma=1,
                   overlap=0.1,
                   sigma_bins=1):
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

    if max_ecc > 0:  # Noncircular allowed
        if max_ecc > 0.99:
            print("WARNING: max_ecc must be 0.99 or less")
            max_ecc = min(max_ecc, 0.99)

        mean_ecc = np.clip(max_ecc / 2, 0, 0.99)
        ecc_arr = np.random.exponential(scale=mean_ecc, size=(num_blobs, ))
        ecc_arr = np.clip(ecc_arr, 0, max_ecc)
        # print('ecc')
        # print(ecc_arr)
        # b_arr = sigmas
        # a_arr = ecc_arr * b_arr
        theta_arr = np.random.randint(0, 180, size=(num_blobs, ))
        # print(np.vstack((a_arr, b_arr)).T)

    if overlap is not None and overlap > 0:
        blobs = prune_nonzero_overlap(blobs, overlap)

    out = np.zeros(imsize)
    for idx, (row, col, sigma, amp) in enumerate(blobs):
        if max_ecc > 0:
            out += make_gaussian_ellipse(
                imsize,
                # a=a_arr[idx],
                # b=b_arr[idx],
                sigma=sigma,
                ecc=ecc_arr[idx],
                row=int(row),
                col=int(col),
                theta=theta_arr[idx],
                amp=amp,
            )
        else:
            out += make_gaussian(imsize, sigma, row=int(row), col=int(col), amp=amp)

    if noise_sigma > 0:
        out += make_noise(imsize, noise_sigma)

    # convert blobs into (row, col, radius, amp) format
    return blobs * np.array([1, 1, np.sqrt(2), 1]), out


@log_runtime
def calc_detection_stats(blobs_real, detected, min_iou=0.5, verbose=True):
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
        iou_frac = blob.skblob.intersection_over_union(
            cur_clob[:3], min_dist_real_blob[:3], using_sigma=False)
        is_overlapping = iou_frac > min_iou
        # ipdb.set_trace()
        # if np.any(matches):  # If we want to check all iou, maybe do this
        if is_overlapping:
            if verbose:
                print('true (overlap, dist):', iou_frac, dist_3d[closest_idx])
            true_detects.append(cur_clob)
            # match_idxs = np.where(matches)[0]
            # matched_idx_set.update(match_idxs)
            matched_idx_set.add(closest_idx)
        else:
            if verbose:
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


@log_runtime
def demo_ghost_blobs(num_blobs=10,
                     min_iou=0.5,
                     out=None,
                     real_blobs=None,
                     noise_sigma=.5,
                     max_ecc=0.4):
    np.random.seed(1)
    finding_params = {
        'positive': True,
        'negative': True,
        'threshold': 0.35,
        'mag_threshold': None,
        'min_sigma': 5,
        'max_sigma': 140,
        'num_sigma': 70,
        'sigma_bins': 3,
        'log_scale': True,
        # 'bowl_score': .7,
        'bowl_score': 5 / 8,
    }
    if out is None or real_blobs is None:
        print("Generating %s blobs" % num_blobs)
        overlap = 0.5
        real_blobs, out = generate_blobs(
            num_blobs,
            max_amp=15,
            amp_scale=25,
            noise_sigma=noise_sigma,
            max_ecc=max_ecc,
            overlap=overlap,
            sigma_bins=finding_params['sigma_bins'],
        )

    print("Finding blobs in synthetic images")
    detected, sigma_list = blob.find_blobs(out, **finding_params)

    true_d, fp, misses, precision, recall = calc_detection_stats(
        real_blobs, detected, min_iou=min_iou)
    print("Results:")
    print("precision: ", precision)
    print("recall: ", recall)

    plot_blob_results(out, true_d=true_d, fp=fp, misses=misses, delete=False)

    return out, sigma_list, real_blobs, detected, fp, misses


def plot_blob_results(image, true_d=None, fp=None, misses=None, delete=False):
    ax = None
    if true_d is not None:
        fixed, ax = blob.plot.plot_blobs(image=image, blobs=true_d, color='green', delete=delete)
    if fp is not None:
        fixed, ax = blob.plot.plot_blobs(image=image, blobs=fp, color='black', ax=ax, delete=delete)
    if misses is not None:
        fixed, ax = blob.plot.plot_blobs(
            image=image, blobs=misses, color='red', ax=ax, delete=delete)
    return fixed, ax


def make_delta(shape, row=None, col=None):
    delta = np.zeros(shape)
    rows, cols = shape
    if col is None:
        col = cols // 2
    if row is None:
        row = rows // 2
    delta[row, col] = 1
    return delta


def _normalize_gaussian(out, normalize=False, amp=None):
    if normalize:
        return out / np.max(out) if normalize else out
    elif amp is not None:
        return amp * (out / np.max(out))
    else:
        return out


def make_gaussian(
        shape,
        sigma,
        row=None,
        col=None,
        normalize=False,
        amp=None,
        noise_sigma=0,
):
    delta = make_delta(shape, row, col)
    out = nd.gaussian_filter(delta, sigma, mode='constant') * sigma**2
    normed = _normalize_gaussian(out, normalize=normalize, amp=amp)
    if noise_sigma > 0:
        normed += make_noise(shape, noise_sigma)
    return normed


def make_edge(imsize, row=None, col=None, jump=1, max_val=2, min_val=0, noise_sigma=0.2):
    """Create a discontinuity to check for edge detection/igoring"""
    rows, cols = imsize
    ramp = np.dot(np.ones((rows, 1)), np.linspace(1, 1 + max_val, cols).reshape((1, cols)))
    ramp = ramp - 1 + min_val
    if col is None:
        col = cols // 2
    if row is None:
        row = rows // 2
    ramp[row:, col:] = ramp[row:, col:] + jump
    if noise_sigma:
        ramp += make_noise(imsize, sigma=noise_sigma)
    return ramp


def _rotation_matrix(theta):
    """CCW rotation matrix by `theta` degrees"""
    theta_rad = np.deg2rad(theta)
    return np.array([[np.cos(theta_rad), np.sin(theta_rad)],
                     [-np.sin(theta_rad), np.cos(theta_rad)]])


def _calc_ab(sigma, ecc):
    a = np.sqrt(sigma**2 / (1 - ecc))
    b = a * (1 - ecc)
    return a, b


def _xy_grid(shape, xmin=None, xmax=None, ymin=None, ymax=None):
    if xmin is None or xmax is None:
        xmin, xmax = (1, shape[1])
    if ymin is None or ymax is None:
        ymin, ymax = (1, shape[0])
    xx = np.linspace(xmin, xmax, shape[1])
    yy = np.linspace(ymin, ymax, shape[0])
    return np.meshgrid(xx, yy)


def make_gaussian_ellipse(
        shape,
        a=None,
        b=None,
        sigma=None,
        ecc=None,
        row=None,
        col=None,
        theta=0,
        normalize=False,
        amp=None,
        noise_sigma=0,
):
    """Make an ellipse using multivariate gaussian

    Args:
        shape (tuple[int, int]): size of grid
        a: semi major axis length
        b: semi minor axis length
        sigma: std dev of gaussian, if it were circular
        ecc: from 0 to 1, alternative to (a, b) specification is (sigma, ecc)
            ecc = 1 - (b/a), and area = pi*sigma**2 = pi*a*b
        row: row of center
        col: col of center
        theta: degrees of rotation (CCW)
        normalize (bool): if true, set max value to 1
        amp (float): value of peak of gaussian
        noise_sigma (float): optional, adds gaussian noise to blob

    Returns:
        ndarray: grid with one multivariate gaussian heights

    """
    if row is None or col is None:
        nrows, ncols = shape
        row, col = nrows // 2, ncols // 2

    if sigma is not None and ecc is not None:
        a, b = _calc_ab(sigma, ecc)
    if a is None or b is None:
        raise ValueError("Need a,b or sigma,ecc")

    R = _rotation_matrix(theta)
    # To rotate, we do R @ P @ R.T to rotate eigenvalues of P = S L S^-1
    cov = np.dot(R, np.array([[b**2, 0], [0, a**2]]))
    cov = np.dot(cov, R.T)
    var = multivariate_normal(mean=[col, row], cov=cov)

    xx, yy = _xy_grid(shape)
    xy = np.vstack((xx.flatten(), yy.flatten())).T
    out = var.pdf(xy).reshape(shape)
    normed = _normalize_gaussian(out, normalize=normalize, amp=amp)
    if noise_sigma > 0:
        normed += make_noise(shape, noise_sigma)
    return normed


def make_log(shape, sigma, row=None, col=None, normalize=False):
    delta = make_delta(shape, row, col)
    out = nd.gaussian_laplace(delta, sigma) * sigma**2
    return out / np.max(out) if normalize else out


def make_noise(shape, sigma):
    """Generate (N, N) grid of noise terms with variance sigma**2"""
    return sigma * np.random.standard_normal(shape)


GAUSSIAN = make_gaussian
LOG = make_log


def plot_func(func=GAUSSIAN, shape=(501, 501), sigma=None):
    if sigma is None:
        sigma = shape[0] / 19
    f = func(shape, sigma)
    X, Y = _xy_grid(shape, xmin=sigma, xmax=sigma, ymin=sigma, ymax=sigma)

    fig = plt.figure(frameon=False)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, f, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    plt.show()


def auto_corr_ratio(image, sigma, mode='nearest', cval=0):
    A = feature.structure_tensor(image, sigma=sigma, mode=mode, cval=cval)
    lambda1, lambda2 = feature.structure_tensor_eigvals(*A)
    return blob_utils.get_center_value(lambda1 / lambda2)


def plot_auto_corr(image, sigma, mode='nearest', cval=0):
    A = feature.structure_tensor(image, sigma=sigma, mode=mode, cval=cval)
    lambda1, lambda2 = feature.structure_tensor_eigvals(*A)
    fig, axes = plt.subplots(2, 3)
    axim = axes.ravel()[0].imshow(image)
    axes.ravel()[0].set_title('orig. image')
    fig.colorbar(axim, ax=axes.ravel()[0])

    titles = ['Ix * Ix', 'Ix * Iy', 'Iy * Iy']

    for idx, ax in enumerate(axes.ravel()[1:4]):
        axim = ax.imshow(A[idx])
        ax.set_title(titles[idx])
        fig.colorbar(axim, ax=ax)

    ax4, ax5 = axes.ravel()[4:6]
    axim = ax4.imshow(lambda1)
    fig.colorbar(axim, ax=ax4)
    ax4.set_title('larger lambda')
    axim = ax5.imshow(lambda2)
    fig.colorbar(axim, ax=ax5)
    ax5.set_title('smaller lambda')

    return A, lambda1, lambda2, fig, axes


def make_valley(shape, rotate=0):
    rows, cols = shape
    out = np.dot(np.ones((rows, 1)), (np.linspace(-1, 1, cols)**2).reshape((1, cols)))
    if rotate > 0:
        out = transform.rotate(out, None, mode='edge')
    return out


def make_bowl(shape):
    rows, cols = shape
    xx, yy = _xy_grid(shape)
    z = xx**2 + yy**2
    return z / np.max(z)


def demo_shape_index():
    shape = (50, 50)
    sigma = 9
    bowl = make_bowl(shape)
    valley = make_valley(shape)
    gauss = make_gaussian(shape, sigma, amp=-1) + 1
    fig, axes = plt.subplots(2, 3)

    a1 = axes[0, 0].imshow(bowl, vmin=0, vmax=1)
    axes[0, 0].set_title('bowl')
    axes[0, 1].imshow(valley, vmin=0, vmax=1)
    axes[0, 1].set_title('valley')
    axes[0, 2].imshow(gauss, vmin=0, vmax=1)
    axes[0, 2].set_title('gauss')
    fig.colorbar(a1, ax=axes[0, 2])

    a2 = axes[1, 0].imshow(feature.shape_index(bowl, sigma=1, mode='nearest'), vmin=-1, vmax=1)
    axes[1, 0].set_title('shape index:bowl')
    axes[1, 1].imshow(feature.shape_index(valley, sigma=sigma, mode='nearest'), vmin=-1, vmax=1)
    axes[1, 1].set_title('shape index:valley')
    axes[1, 2].imshow(feature.shape_index(gauss, sigma=sigma, mode='nearest'), vmin=-1, vmax=1)
    axes[1, 2].set_title('shape index:gauss')
    fig.colorbar(a2, ax=axes[1, 2])

    return axes, bowl, valley, gauss


@log_runtime
def simulate_detections(num_sims,
                        result_file='blobsim.csv',
                        patch_dir=None,
                        num_blobs=None,
                        min_blobs=20,
                        max_blobs=60,
                        max_ecc=0.4,
                        noise_sigma=0.5,
                        min_amp=2,
                        max_amp=15,
                        amp_scale=25,
                        random_seed=None,
                        **kwargs):
    @log_runtime
    def run(run_idx):
        finding_params = {
            'positive': True,
            'negative': True,
            'threshold': 0.35,
            'mag_threshold': None,
            'min_sigma': 5,
            'max_sigma': 140,
            'num_sigma': 70,
            'sigma_bins': 3,
            'log_scale': True,
            # 'bowl_score': 0,
            # 'bowl_score': .7,
            'bowl_score': 5 / 8,
        }
        finding_params.update(kwargs)
        print("Using the following finding_params:")
        print(finding_params)

        if num_blobs is None:
            nblobs = np.random.randint(min_blobs, max_blobs)
        overlap = .1
        real_blobs, out = generate_blobs(
            nblobs,
            min_amp=min_amp,
            max_amp=max_amp,
            amp_scale=amp_scale,
            noise_sigma=noise_sigma,
            max_ecc=max_ecc,
            overlap=0.1,
        )

        # print("Finding blobs in synthetic images")
        detected, sigma_list = blob.find_blobs(out, **finding_params)

        true_d, fp, misses, precision, recall = calc_detection_stats(
            real_blobs, detected, verbose=False)
        print("Results for run %s:" % run_idx)
        print("num blobs real: ", len(real_blobs))
        print("precision: ", precision)
        print("recall: ", recall)
        if patch_dir is not None:
            td_fname = os.path.join(patch_dir, 'true_detections_%s' % run_idx)
            fp_fname = os.path.join(patch_dir, 'false_positives_%s' % run_idx)
            miss_fname = os.path.join(patch_dir, 'misses_%s' % run_idx)
            record_blob_patches(fp_fname, out, fp)
            record_blob_patches(miss_fname, out, misses)
            record_blob_patches(td_fname, out, true_d)
            img_fname = os.path.join(patch_dir, 'image_%s' % run_idx)
            np.savez(img_fname, image=out, real_blobs=real_blobs)
        return precision, recall, len(real_blobs)

    if random_seed is not None:
        np.random.seed(random_seed)

    total_precision, total_recall = 0, 0
    with open(os.path.join(patch_dir, result_file), 'w') as f:
        for run_idx in range(1, num_sims + 1):
            precision, recall, nblobs = run(run_idx)
            f.write("%.2f,%.2f,%d\n" % (precision, recall, nblobs))
            total_precision += precision
            total_recall += recall

    print("Total precision:", total_precision / num_sims)
    print("Total recall:", total_recall / num_sims)


def record_blob_patches(fname, image, blobs):
    print("Recording %s" % fname)
    patches = []
    for b in blobs:
        patch = blob_utils.crop_blob(image, b)
        patches.append(patch)
    np.savez(fname, *patches, blobs=blobs)


def load_run(run_idx, data_path='.'):
    """Load a simulation run to retrieve image, blobs, and patches
    Args:
        run_idx (int): run number
        data_path (str): where .npz data is saved

    Returns:
        dict: keys = 'td_patches', 'td_blobs', 'fp_patches', 'fp_blobs',
            'miss_patches', 'miss_blobs', 'image', 'real_blobs'
            values: associated numpy arrays

    """
    image_arrs = np.load(os.path.join(data_path, "image_%s.npz" % run_idx))
    td_arrs = np.load(os.path.join(data_path, 'true_detections_%s.npz' % run_idx))
    fp_arrs = np.load(os.path.join(data_path, 'false_positives_%s.npz' % run_idx))
    miss_arrs = np.load(os.path.join(data_path, 'misses_%s.npz' % run_idx))

    run_arrays = {
        'td_patches': [td_arrs[key] for key in td_arrs.keys() if key.startswith('arr')],
        'td_blobs': td_arrs['blobs'],
        'fp_patches': [fp_arrs[key] for key in fp_arrs.keys() if key.startswith('arr')],
        'fp_blobs': fp_arrs['blobs'],
        'miss_patches': [miss_arrs[key] for key in miss_arrs.keys() if key.startswith('arr')],
        'miss_blobs': miss_arrs['blobs'],
        'image': image_arrs['image'],
        'real_blobs': image_arrs['real_blobs'],
    }
    return run_arrays


def load_run_blob_type(data_path='.', patch_type='fp'):
    if patch_type in ('fp', 'false_positives'):
        fname_search = 'false_positive*.npz'
    elif patch_type in ('td', 'true_detections'):
        fname_search = 'true_detections*.npz'
    elif patch_type in ('miss', 'misses'):
        fname_search = 'misses*.npz'
    else:
        raise ValueError("patch_type must be 'td', 'fp', 'miss")

    file_list = glob.glob(os.path.join(data_path, fname_search))
    blobs = np.empty((0, 4))
    blob_patches = []
    for f in file_list:
        npz = np.load(f)
        new_blobs = npz["blobs"]
        new_patches = [npz[key] for key in npz.keys() if key.startswith('arr')]
        if new_blobs.size:
            blobs = np.concatenate((blobs, new_blobs))
            blob_patches.extend(new_patches)

    return blobs, blob_patches


def plot_run_summary(run_arrays=None,
                     image=None,
                     true_d=None,
                     fp=None,
                     misses=None,
                     data_path=None,
                     run_idx=None):
    """Takes `run_arrays` from `load_run` and plots the image with detections and misses"""
    if data_path is not None and run_idx is not None:
        run_arrays = load_run(run_idx, data_path=data_path)
    if image is None:
        image = run_arrays['image']
    if true_d is None:
        true_d = run_arrays['td_blobs']
    if fp is None:
        fp = run_arrays['fp_blobs']
    if misses is None:
        misses = run_arrays['miss_blobs']

    _, ax = plot_blob_results(image, true_d=true_d, fp=fp, misses=misses, delete=False)
    return ax


def correct_run(run_idx=None, data_path='.'):
    """Takes `run_arrays` from `load_run` and plots the image with detections and misses"""
    run_arrays = load_run(run_idx, data_path=data_path)
    image = run_arrays['image']
    true_d = run_arrays['td_blobs']
    fp = run_arrays['fp_blobs']
    misses = run_arrays['miss_blobs']

    len_blobs_real = len(misses) + len(true_d)
    len_detected = len(true_d) + len(fp)
    print("Old precision: %.3f" % (len(true_d) / len_detected))
    print("Old recall: %.3f" % (len(true_d) / len_blobs_real))

    print("FIRST SHOWING RESULTS: no delete handling")
    plot_blob_results(image, true_d=true_d, fp=fp, misses=misses, delete=False)

    _, ax = blob.plot.plot_blobs(image=image, blobs=true_d, color='green', delete=False)
    fp_fix, ax = blob.plot.plot_blobs(image=image, blobs=fp, color='black', delete=True)
    # plt.show(block=True)

    # The misses we just want to get rid of if they are bad
    misses_fix, ax = blob.plot.plot_blobs(image=image, blobs=misses, color='red', delete=True)
    plt.show(block=True)

    # Now add the bad false pisitives into the true_d array
    fp_wrongs = [b for b in fp if b not in fp_fix]
    if fp_wrongs:
        true_d_fix = np.concatenate((true_d, fp_wrongs), axis=0)
    else:
        true_d_fix = true_d

    td_fname = os.path.join(data_path, 'true_detections_%s_fix' % run_idx)
    fp_fname = os.path.join(data_path, 'false_positives_%s_fix' % run_idx)
    miss_fname = os.path.join(data_path, 'misses_%s_fix' % run_idx)
    record_blob_patches(fp_fname, image, fp_fix)
    record_blob_patches(miss_fname, image, misses_fix)
    record_blob_patches(td_fname, image, true_d_fix)

    len_blobs_real = len(misses_fix) + len(true_d_fix)
    len_detected = len(true_d_fix) + len(fp_fix)
    print("New precision: %.3f" % (len(true_d_fix) / len_detected))
    print("New recall: %.3f" % (len(true_d_fix) / len_blobs_real))


def plot_run_patches(run_arrs, keys=('td', 'fp', 'miss'), sigma=0):
    for key in keys:
        full_key = '%s_patches' % key
        patches = run_arrs[full_key]
        for patch in patches:
            if sigma > 0:
                patch = blob_utils.gaussian_filter_nan(patch, sigma=sigma)
            blob.plot.plot_cropped_blob(patch=patch, )
            plt.suptitle(full_key)


def simulation_results(outfile):
    df = pd.read_csv(outfile, header=None, names=['precision', 'recall', 'num_blobs'])
    df_grouped = df.groupby(['num_blobs']).describe()
    ax = df_grouped['precision'].plot(y='mean', use_index=True, label='mean precision')
    ax = df_grouped['recall'].plot(y='mean', use_index=True, ax=ax, label='mean recall')
    return df, ax


def split_pos_neg(blobs):
    return np.array([b for b in blobs if b[3] > 0]), np.array([b for b in blobs if b[3] < 0])


def prune_nonzero_overlap(blobs, overlap=0.1):
    pruned = np.empty((0, 4))
    for bs in split_pos_neg(blobs):
        out = blob.skblob.prune_overlap_blobs(bs, overlap, sigma_bins=1)
        if out.size:
            pruned = np.vstack((pruned, out))
    return pruned


def make_stack(shape=(501, 501), max_amp=3, cmap='jet'):
    """Makes composite of 3 blob sizes, with small negative inside big positive"""
    N = shape[0]
    b1 = make_gaussian(shape, 100, None, None)
    b2 = make_gaussian(shape, 30, N // 3, N // 3)
    b3 = make_gaussian(shape, 7, 4 * N // 7, 4 * N // 7)
    # little ones that merge to one
    # b4 = make_gaussian(shape, 19, 6 * N // 7, 6 * N // 7)
    # b4 += make_gaussian(shape, 19, 47 + 6 * N // 7, 6 * N // 7)
    b4 = make_gaussian(shape, 19, 6 * N // 8, 6 * N // 8)
    b4 += make_gaussian(shape, 19, 48 + 6 * N // 8, 6 * N // 8)
    b4 += make_gaussian(shape, 19, 6 * N // 8, 48 + 6 * N // 8)
    b4 += make_gaussian(shape, 19, 48 + 6 * N // 8, 48 + 6 * N // 8)
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

    _, ax = blob.plot.plot_blobs(
        image=out, blobs=blob.find_edge_blobs(blobs, out.shape)[0], ax=fig.gca(), color='blue')

    plt.imshow(image_cube[:, :, 30], cmap='jet', vmin=-1.4, vmax=1.3)
    plt.imshow(image_cube[:, :, 10], cmap='jet', vmin=-1.4, vmax=1.3)
    return out, blobs, sigma_list, image_cube, fig


if __name__ == '__main__':
    run_idx = int(sys.argv[1])
    try:
        patch_dir = sys.argv[2]
    except IndexError:
        patch_dir = 'patches'
    correct_run(run_idx, patch_dir)
