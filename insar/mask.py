import os
import glob
import numpy as np
from insar import sario, utils


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
    """Assumes save_geo_masks already run"""
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


def masked_lstsq(A, b, geo_mask_columns=None, rcond=None, *args, **kwargs):
    """Performs least squares on masked numpy arrays

    Handles the mask by deleting the row of A corresponding
    to a masked b element.

    Inputs same as numpy.linalg.lstsq, where b_masked can be
    a (n x k) matrix and each n-length column inverted.
    """
    b_masked = b.view(np.ma.MaskedArray)
    if b_masked.ndim == 1:
        b_masked = utils.force_column(b_masked)

    # First check if no masks exist (run normally if so)
    if b_masked.mask is np.ma.nomask or not b_masked.mask.any():
        return np.linalg.lstsq(A, b, rcond=rcond, *args, **kwargs)[0]

    # Otherwise, run first in bulk on all b's with no masks
    # Only iterate over ones with some mask
    out_final = np.ma.empty((A.shape[1], b_masked.shape[1]))

    good_col_idxs = ~(b_masked.mask).any(axis=0)
    if np.any(good_col_idxs):
        good_cols = b_masked[:, good_col_idxs]
        good_sol = np.linalg.lstsq(A, good_cols, rcond=rcond, *args, **kwargs)[0]
        out_final[:, good_col_idxs] = good_sol

    bad_sol_list = []
    bad_col_idxs = np.where((b_masked.mask).any(axis=0))[0]
    for idx in bad_col_idxs:
        col_masked = b_masked[:, idx]

        missing = col_masked.mask
        if np.all(missing):
            # Here the entire column is masked, so skip with nans
            nan_solution = np.full(out_final.shape[0], np.nan)
            bad_sol_list.append(nan_solution)
            continue

        # Add squeeze for empty mask case, since it adds
        # a singleton dimension to beginning (?? why)
        A_deleted = np.squeeze(A[~missing])
        if A_deleted.ndim == 1:
            A_deleted = A_deleted[:, np.newaxis]
        elif A_deleted.ndim == 0:
            # Edge case: only 1 true "missing" entry
            A_deleted = np.atleast_2d(A_deleted)

        col_deleted = np.squeeze(col_masked[~missing])
        if col_deleted.ndim == 0:
            col_deleted = np.atleast_1d(col_deleted)

        # If all deleted, just use NaNs as the solution
        if A_deleted.size == 0:
            sol = np.full((out_final.shape[0], 1), np.NaN)
            residuals = []
        else:
            sol, residuals, rank, _ = np.linalg.lstsq(
                A_deleted, col_deleted, rcond=rcond, *args, **kwargs)

        # If underdetermined, fill appropriate places with NaNs
        if not residuals:
            if geo_mask_columns is None or A.shape[1] != geo_mask_columns.shape[0]:
                sol[...] = np.NaN
            else:
                mask_col = geo_mask_columns[:, idx]
                masked_idxs = np.where(mask_col)[0]
                # In min velocity LS, a 0 affects before and on index
                masked_idxs = np.unique(np.concatenate((masked_idxs, masked_idxs - 1)))
                sol[masked_idxs] = np.NaN
                sol = np.ma.masked_invalid(sol)

        bad_sol_list.append(sol)

    # Squeeze added because sometimes extra singleton dim added
    out = np.squeeze(np.ma.stack(bad_sol_list, axis=1))

    if out.ndim == 1:
        out = utils.force_column(out)
    try:
        out_final[:, bad_col_idxs] = out
    except ValueError:
        # kinda hacky :\
        # https://github.com/numpy/numpy/issues/5710
        # out sometimes ends up transposed in 1-dim cases, like the
        # constant velocity case
        out_final[:, bad_col_idxs] = out.T
    return out_final
