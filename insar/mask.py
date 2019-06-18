import os
import glob
import numpy as np
from apertools import sario, utils


def save_geo_masks(geo_dir=None, geo_filename=None, row_looks=1, col_looks=1):
    """Creates .mask files for geos where zeros occur

    Can either pass a directory to convert all geos, or one geo_filename
    """

    def _get_geo_mask(geo_arr):
        return np.ma.make_mask(geo_arr == 0, shrink=False)

    def _save_mask(geo_fname, row_looks, col_looks):
        mask_fname = geo_fname + '.mask.npy'
        g = sario.load(geo_fname, looks=(row_looks, col_looks))
        print('Saving %s' % mask_fname)
        np.save(mask_fname, _get_geo_mask(g))

    if geo_filename is not None:
        # Handle wither just geo, or mask name is passed
        geo_fname = geo_filename.replace('.mask.npy', '')
        _save_mask(geo_fname, row_looks, col_looks)
    elif geo_dir is not None:
        for geo_fname in sario.find_files(geo_dir, "*.geo"):
            _save_mask(geo_fname, row_looks, col_looks)
    else:
        raise ValueError("Need geo_dir or geo_filename")


def save_int_masks(igram_fnames,
                   igram_date_list,
                   geo_date_list,
                   geo_path="../",
                   row_looks=1,
                   col_looks=1,
                   verbose=False):
    """Creates igram masks by taking the logical-or of the two .geo files
    
    Assumes save_geo_masks already run
    """
    geomask_list = []
    for cur_date in geo_date_list:
        geoname = os.path.join(geo_path, '*{}*.geo'.format(cur_date.strftime('%Y%m%d')))
        # Now glob to grab whether it's S1A_ or S1B_
        geo_filename = glob.glob(geoname)[0]
        geo_mask_filename = geo_filename + '.mask.npy'
        if not os.path.exists(geo_mask_filename):
            save_geo_masks(geo_filename=geo_filename, row_looks=row_looks, col_looks=col_looks)
        geomask_list.append(sario.load(geo_mask_filename))

    for idx, (early, late) in enumerate(igram_date_list):
        early_idx = geo_date_list.index(early)
        late_idx = geo_date_list.index(late)
        early_mask = geomask_list[early_idx]
        late_mask = geomask_list[late_idx]

        igram_mask = np.logical_or(early_mask, late_mask)
        igram_mask_name = igram_fnames[idx] + '.mask.npy'
        if verbose:
            print("Saving %s" % igram_mask_name)
        np.save(igram_mask_name, igram_mask)


def masked_lstsq(A, b, geo_mask_columns=None):
    """Performs least squares on masked numpy arrays

    Handles the mask by deleting the row of A corresponding
    to a masked b element.

    Inputs same as numpy.linalg.lstsq, where b_masked can be
    a (n x k) matrix and each n-length column inverted.

    Inputs:
      A (np.ndarray) 2D array of Ax = b
      b (np.ndarray) an (m x k) array, the right hand side
      geo_mask_columns (np.ndarray): 2D array where each column is
        a boolean mask of geo dates that should be masked
    """
    b_masked = b.view(np.ma.MaskedArray)
    if b_masked.ndim == 1:
        b_masked = utils.force_column(b_masked)

    # First check if no masks exist (run normally if so)
    if b_masked.mask is np.ma.nomask or not b_masked.mask.any():
        return np.linalg.lstsq(A, b, rcond=None)[0]

    # Otherwise, run first in bulk on all b's with no masks and
    # only loop over ones with some mask
    out_final = np.ma.empty((A.shape[1], b_masked.shape[1]))

    good_col_idxs = ~(b_masked.mask).any(axis=0)
    bad_col_idxs = np.where((b_masked.mask).any(axis=0))[0]

    if np.any(good_col_idxs):
        out_final[:, good_col_idxs] = solve_good_columns(A, good_col_idxs, b_masked)

    out_final = solve_bad_columns(A, bad_col_idxs, b_masked, geo_mask_columns, out_final)
    return out_final


def solve_good_columns(A, good_col_idxs, b_masked):
    good_cols = b_masked[:, good_col_idxs]
    return np.linalg.lstsq(A, good_cols, rcond=None)[0]


def solve_bad_columns(A, bad_col_idxs, b_masked, geo_mask_columns, out_final):
    # Solve one partially masked column at a time
    for idx in bad_col_idxs:
        col_masked = b_masked[:, idx]

        missing = col_masked.mask
        if np.all(missing):
            # Here the entire column is masked, so skip with nans
            # Bracket around [idx] so that out_final[:,[idx]] is (n, 1) shape
            out_final[:, [idx]] = np.full((out_final.shape[0], 1), np.nan)
            continue

        A_deleted = A[~missing, :]
        col_deleted = col_masked[~missing]

        # TODO: failing if all except one of `missing` == True (only one pixel to solve)
        sol, residuals, rank, _ = np.linalg.lstsq(A_deleted, col_deleted, rcond=None)

        # If underdetermined, fill appropriate places with NaNs
        if not residuals:
            # If we aren't maksing specific geo dates, or doing a constant vel solution
            # TODO: Are there other cases we want to mask all?
            if geo_mask_columns is None or A.shape[1] == 1:
                sol[...] = np.NaN
            else:
                mask_col = geo_mask_columns[:, idx]
                masked_idxs = np.where(mask_col)[0]
                # In min velocity LS, mask on day j affects velocities (j-1, j)
                masked_idxs = np.clip(
                    np.concatenate((masked_idxs, masked_idxs - 1)),
                    0,
                    sol.shape[0] - 1,
                )
                sol[masked_idxs] = np.NaN
                # Also update the mask for these NaNs
                sol = np.ma.masked_invalid(sol)

        # Use tuple around out_final[:, [idx]] to make shape (N,1)
        out_final[:, [idx]] = utils.atleast_2d(sol)

    return out_final


def mask_int(image, dem_file=None, dem=None):
    """Masks image from the zeros of a dem"""
    if dem_file:
        dem = insar.sario.load(dem_file)

    mask = imresize((dem == 0).astype(float), image.shape)
    intmask = np.ma.array(image, mask=mask)
    return intmask.filled(0)
