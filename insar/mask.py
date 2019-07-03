import os
import glob
import numpy as np
from apertools import sario, utils


def masked_lstsq(B, d, geo_mask_columns=None):
    """Performs least squares on masked numpy arrays

    Handles the mask by deleting the row of B corresponding
    to a masked d element.

    Inputs same as numpy.linalg.lstsq, where d_masked can be
    a (n x k) matrix and each n-length column inverted.

    Inputs:
      B (np.ndarray) 2D array of Bx = d
      d (np.ndarray) an (m x k) array, the right hand side
      geo_mask_columns (np.ndarray): 2D array where each column is
        a boolean mask of geo dates that should be masked
    """
    d_masked = d.view(np.ma.MaskedArray)
    if d_masked.ndim == 1:
        d_masked = utils.force_column(d_masked)

    # First check if no masks exist (run normally if so)
    if d_masked.mask is np.ma.nomask or not d_masked.mask.any():
        pB = np.linalg.pinv(B)
        return np.dot(pB, d)
        # return np.linalg.lstsq(B, d, rcond=None)[0]

    # Otherwise, run first in bulk on all d's with no masks and
    # only loop over ones with some mask
    out_final = np.ma.empty((B.shape[1], d_masked.shape[1]))

    good_col_idxs = ~(d_masked.mask).any(axis=0)
    bad_col_idxs = np.where((d_masked.mask).any(axis=0))[0]

    if np.any(good_col_idxs):
        out_final[:, good_col_idxs] = solve_good_columns(B, good_col_idxs, d_masked)

    out_final = solve_bad_columns(B, bad_col_idxs, d_masked, geo_mask_columns, out_final)
    return out_final


def solve_good_columns(B, good_col_idxs, d_masked):
    good_cols = d_masked[:, good_col_idxs]
    pB = np.linalg.pinv(B)
    return np.dot(pB, good_cols)
    # return np.linalg.lstsq(B, good_cols, rcond=None)[0]


def solve_bad_columns(B, bad_col_idxs, d_masked, geo_mask_columns, out_final):
    # Solve one partially masked column at a time
    for idx in bad_col_idxs:
        col_masked = d_masked[:, idx]

        missing = col_masked.mask
        if np.all(missing):
            # Here the entire column is masked, so skip with nans
            # Bracket around [idx] so that out_final[:,[idx]] is (n, 1) shape
            out_final[:, [idx]] = np.full((out_final.shape[0], 1), np.nan)
            continue

        B_deleted = B[~missing, :]
        col_deleted = col_masked[~missing]

        # TODO: failing if all except one of `missing` == True (only one pixel to solve)
        sol, residuals, rank, _ = np.linalg.lstsq(B_deleted, col_deleted, rcond=None)

        # If underdetermined, fill appropriate places with NaNs
        if not residuals:
            # If we aren't maksing specific geo dates, or doing a constant vel solution
            # TODO: Are there other cases we want to mask all?
            if geo_mask_columns is None or B.shape[1] == 1:
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
