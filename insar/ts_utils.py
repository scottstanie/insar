import numpy as np
from datetime import datetime, date
from matplotlib.dates import date2num
from concurrent.futures import Future, Executor
from threading import Lock
from itertools import chain, combinations
from apertools import sario


def build_A_matrix(sar_dates, ifg_date_pairs):
    """Takes the list of igram dates and builds the SBAS A matrix

    Args:
        sar_dates (list[date]): datetimes of the acquisitions
        ifg_date_pairs (list[tuple(date, date)])

    Returns:
        np.array 2D: the incident-like matrix from the SBAS paper: A*phi = dphi
            Each row corresponds to an igram, each column to a SAR date
            value will be -1 on the early (reference) igrams, +1 on later (secondary)
    """
    if isinstance(sar_dates[0], datetime) or isinstance(sar_dates[0], date):
        sar_dates = date2num(sar_dates)
    if isinstance(ifg_date_pairs[0][0], datetime) or isinstance(
        ifg_date_pairs[0][0], date
    ):
        ifg_date_pairs = date2num(ifg_date_pairs)
    # We take the first .geo to be time 0, leave out of matrix
    # Match on date (not time) to find indices
    M = len(ifg_date_pairs)  # Number of igrams, number of rows
    N = len(sar_dates) - 1
    A = np.zeros((M, N))
    for j in range(M):
        early_igram, late_igram = ifg_date_pairs[j]

        for idx in range(N):
            sd = sar_dates[idx + 1]
            if early_igram == sd:
                A[j, idx] = -1
            elif late_igram == sd:
                A[j, idx] = 1

    return A


def build_B_matrix(sar_dates, ifg_date_pairs, model=None):
    """Takes the list of igram dates and builds the SBAS B (velocity coeff) matrix

    Args:
        sar_dates (list[date]): dates of the SAR acquisitions
        ifg_date_pairs (list[tuple(date, date)])
        model (str): If "linear", creates the M x 1 matrix for linear velo model

    Returns:
        np.array: 2D array of the velocity coefficient matrix from the SBAS paper:
                Bv = dphi
            Each row corresponds to an igram, each column to a SAR date
            value will be t_k+1 - t_k for columns after the -1 in A,
            up to and including the +1 entry
    """
    if isinstance(sar_dates[0], datetime) or isinstance(sar_dates[0], date):
        sar_dates = date2num(sar_dates)
    if isinstance(ifg_date_pairs[0][0], datetime) or isinstance(
        ifg_date_pairs[0][0], date
    ):
        ifg_date_pairs = date2num(ifg_date_pairs)

    timediffs = np.diff(sar_dates)

    A = build_A_matrix(sar_dates, ifg_date_pairs)
    B = np.zeros_like(A)

    for j, row in enumerate(A):
        # if no -1 entry, start at index 0. Otherwise, add 1 to exclude the -1 index
        start_idx = 0
        for idx, item in enumerate(row):
            if item == -1:
                start_idx = idx + 1
            elif item == 1:
                end_idx = idx + 1

        # Now only fill in the time diffs in the range from the early igram index
        # to the later igram index
        B[j][start_idx:end_idx] = timediffs[start_idx:end_idx]

    if model == "linear":
        BB = B.sum(axis=1)
        return BB.reshape((-1, 1))
    else:
        return B


def A_polynomial(sar_dates, degree=1):
    """System matrix for a polynomial fit to data

    Args:
        sar_dates (iterable[date]): dates of the SAR acquisitions

    Returns:
        A, size=(len(sar_dates)) 2D Vandermonde array to solve for the polynomial coefficients
    """
    date_nums = date2num(sar_dates)
    return np.polynomial.polynomial.polyvander(date_nums, degree)


def stack_to_cols(stacked):
    """Takes a 3D array, makes vectors along the 3D axes into cols

    The reverse function of cols_to_stack

    Args:
        stacked (ndarray): 3D array, each [idx, :, :] is an array of interest

    Returns:
        ndarray: a 2D array where each of the stacked[:, i, j] is
            now a column

    Raises:
        ValueError: if input shape is not 3D

    Example:
        >>> a = np.arange(18).reshape((2, 3, 3))
        >>> cols = stack_to_cols(a)
        >>> print(cols)
        [[ 0  1  2  3  4  5  6  7  8]
         [ 9 10 11 12 13 14 15 16 17]]
    """
    if len(stacked.shape) != 3:
        raise ValueError("Must be a 3D ndarray")

    num_stacks = stacked.shape[0]
    return stacked.reshape((num_stacks, -1))


def cols_to_stack(columns, rows, cols):
    """Takes a 2D array of columns, reshapes to cols along 3rd axis

    The reverse function of stack_to_cols

    Args:
        stacked (ndarray): 2D array of columns of data
        rows (int): number of rows of original stack
        cols (int): number of rows of original stack

    Returns:
        ndarray: a 2D array where each output[idx, :, :] was column idx

    Raises:
        ValueError: if input shape is not 2D

    Example:
        >>> a = np.arange(18).reshape((2, 3, 3))
        >>> cols = stack_to_cols(a)
        >>> print(cols)
        [[ 0  1  2  3  4  5  6  7  8]
         [ 9 10 11 12 13 14 15 16 17]]
        >>> print(np.all(cols_to_stack(cols, 3, 3) == a))
        True
    """
    if len(columns.shape) != 2:
        raise ValueError("Must be a 2D ndarray")

    return columns.reshape((-1, rows, cols))


def integrate_velocities(velocity_array, timediffs):
    """Takes SBAS velocity output and finds phases

    Args:
        velocity_array (ndarray): output of run_sbas, velocities at
            each point in time
        timediffs (np.array): dtype=int, days between each SAR acquisitions
            length will be 1 less than num SAR acquisitions

    Returns:
        ndarray: integrated phase array

    """
    # multiply each column of vel array: each col is a separate solution
    phi_diffs = timediffs.reshape((-1, 1)) * velocity_array

    # Now the final phase results are the cumulative sum of delta phis
    phi_arr = np.ma.cumsum(phi_diffs, axis=0)
    # Add 0 as first entry of phase array to match slclist length on each col
    phi_arr = np.ma.vstack((np.zeros(phi_arr.shape[1]), phi_arr))

    return phi_arr


def _get_cumsum_mat(Tvec):
    T_mat = np.repeat(Tvec.reshape((-1, 1)), len(Tvec), axis=1).astype(float)
    return T_mat * np.tri(*T_mat.shape)  # tri makes a matrix of ones in the lower left


# Alternate way to integrate, using a matrix instead of "cumsum"
def integrate_velocities_mat(velocity_array, timediffs):
    T_mat = _get_cumsum_mat(timediffs)
    phi_arr = T_mat @ velocity_array

    # Add 0 as first entry of phase array to match geolist length on each col
    phi_arr = np.ma.vstack((np.zeros(phi_arr.shape[1]), phi_arr))

    return phi_arr


def prepB(slclist, ifglist, constant_velocity=False, alpha=0, difference=False):
    """TODO: transfer this to the "run_sbas"? this is from julia"""
    B = build_B_matrix(slclist, ifglist)
    # Adjustments to solution:
    # Force velocity constant across time
    if constant_velocity is True:
        B = np.expand_dims(np.sum(B, axis=1), axis=1)
    # Add regularization to the solution
    elif alpha > 0:
        # Augment only if regularization requested
        reg_matrix = (
            _create_diff_matrix(B.shape[1]) if difference else np.eye(B.shape[1])
        )
        B = np.vstack((B, alpha * reg_matrix))
    return B


def _create_diff_matrix(n, order=1):
    """Creates n x n matrix subtracting adjacent vector elements

    Example:
        >>> print(_create_diff_matrix(4, order=1))
        [[ 1 -1  0  0]
         [ 0  1 -1  0]
         [ 0  0  1 -1]]

        >>> print(_create_diff_matrix(4, order=2))
        [[ 1 -1  0  0]
         [-1  2 -1  0]
         [ 0 -1  2 -1]
         [ 0  0 -1  1]]

    """
    if order == 1:
        diff_matrix = -1 * np.diag(np.ones(n - 1), k=1).astype("int")
        np.fill_diagonal(diff_matrix, 1)
        diff_matrix = diff_matrix[:-1, :]
    elif order == 2:
        diff_matrix = -1 * np.diag(np.ones(n - 1), k=1).astype("int")
        diff_matrix = diff_matrix + -1 * np.diag(np.ones(n - 1), k=-1).astype("int")
        np.fill_diagonal(diff_matrix, 2)
        diff_matrix[-1, -1] = 1
        diff_matrix[0, 0] = 1

    return diff_matrix


def _augment_matrices(B, delta_phis, alpha, difference=False):
    reg_matrix = _create_diff_matrix(B.shape[1]) if difference else np.eye(B.shape[1])
    B = np.vstack((B, alpha * reg_matrix))
    # Now make num rows match
    zeros_shape = (B.shape[0] - delta_phis.shape[0], delta_phis.shape[1])
    delta_phis = np.vstack((delta_phis, np.zeros(zeros_shape)))
    return B, delta_phis


def _augment_zeros(B, delta_phis):
    try:
        r, c = delta_phis.shape
    except ValueError:
        delta_phis = delta_phis.reshape((-1, 1))
        r, c = delta_phis.shape
    zeros_shape = (B.shape[0] - r, c)
    delta_phis = np.vstack((delta_phis, np.zeros(zeros_shape)))
    return delta_phis


def ptp_by_date(da):
    import xarray as xr

    ydim, xdim = da.dims[-2:]
    return xr.apply_ufunc(
        np.ptp, da, input_core_dims=[[ydim, xdim]], kwargs={"axis": (-2, -1)}
    )


def ptp_by_date_pct(da, low=0.05, high=0.95):
    """Find the peak-to-peak amplitude per image, based on the `high`/`low` percentiles"""
    import xarray as xr

    ydim, xdim = da.dims[-2:]
    high_q = da.quantile(high, dim=(ydim, xdim))
    low_q = da.quantile(low, dim=(ydim, xdim))
    return high_q - low_q


def build_closure_matrix(ifg_date_pairs):
    """Takes the list of igram dates and builds the SBAS A matrix

    Args:
        ifg_date_pairs (list[tuple(date, date)])

    Returns:
        C (2D ndarray): Size = (K, M), where M = len(ifg_date_pairs),
            K = number of possible triplets from the ifgs.
            Each row has two 1s, one -1, corresponding to indexes within
            `ifg_date_pairs` so that `rows @ ifg_date_pairs` sums up the closure phase

    E.g. if `ifg_date_pairs` has [(0, 1), (0, 2),...,(1, 2),...], then
    row 1 of `C` will be
        [+1, -1, 0,...,0,+1,0,....]
    Since ifg(0, 1) + ifg(1,2) - ifg(0, 2) = closure(0,1,2)


    """
    # Get the unique SAR dates present in the interferogram list
    sar_dates = sorted(set(chain.from_iterable(ifg_date_pairs)))
    closure_list = combinations(sar_dates, 3)

    # Create an inverse map from tuple(date1, date1) -> index in ifg list
    ifg_to_idx = {tuple(ifg): idx for idx, ifg in enumerate(ifg_date_pairs)}

    M = len(ifg_date_pairs)  # Number of igrams, number of rows
    C_list = []
    for date1, date2, date3 in closure_list:

        ifg12 = (date1, date2)
        ifg23 = (date2, date3)
        ifg13 = (date1, date3)
        try:
            idx12 = ifg_to_idx[ifg12]
            idx23 = ifg_to_idx[ifg23]
            idx13 = ifg_to_idx[ifg13]
        except KeyError:
            continue

        row = np.zeros(M, dtype=np.int8)
        row[idx12] = 1
        row[idx23] = 1
        row[idx13] = -1
        C_list.append(row)

    return np.stack(C_list).astype(np.float32)


def get_mean_cor(defo_fname="deformation.h5", cor_fname="cor_stack.h5"):
    """Get the mean correlation images for the ifg subset used to make `defo_fname`
    as a DataArray
    """
    return get_cor_for_deformation(defo_fname, cor_fname).mean(axis=0)


def get_cor_for_deformation(
    defo_fname="deformation.h5", cor_fname="cor_stack.h5", cor_dset=sario.STACK_DSET
):
    """Get the stack of correlation images for the ifg subset used to make `defo_fname`
    as a DataArray
    """
    import xarray as xr

    cor_idxs = get_cor_indexes(defo_fname=defo_fname, cor_fname=cor_fname)
    with xr.open_dataset(cor_fname) as ds_cor:
        return ds_cor[cor_dset].sel(ifg_idx=cor_idxs)


def get_cor_indexes(defo_fname="deformation.h5", cor_fname="cor_stack.h5"):
    all_ifgs = [
        tuple(pair) for pair in sario.load_ifglist_from_h5(cor_fname, parse=False)
    ]
    defo_ifgs = [
        tuple(pair) for pair in sario.load_ifglist_from_h5(defo_fname, parse=False)
    ]
    return np.array([all_ifgs.index(ifg) for ifg in defo_ifgs])


def rewrap_to_2pi(phase):
    """Converts phase results to be centered from -pi to pi

    The result from multiplying by calculating, e.g., closure phase,
    will usually have many values centered around -2pi, 0, and 2pi.
    This function puts them all centered around 0.

    Args:
        phase (ndarray): array (or scalar) of phase values

    Returns:
        re-wrapped values within the interval -pi to pi
    """
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def wrap_phase(ph):
    ph = ph - np.round(ph / (2 * np.pi)) * 2 * np.pi
    return ph


def closure_phase(ifg_stack, ifg_date_pairs, rewrap=True):
    """Compute a stack of closure phase values for a stack of interferograms

    Args:
        ifg_stack (3D ndarray): stack of interferogram images
            Can be either complex, or the phase floats. Size = (m, r, c)
        ifg_date_pairs (list[tuple]): pairs of (reference, secondary) dates
        rewrap (bool): force the values to be within (-pi, pi)

    Returns:
        ndarray: stack of closure phase images, sized (nc, r, c)
            where nc is the number of rows produced by `build_closure_matrix`
    """
    C = build_closure_matrix(ifg_date_pairs)
    ifg_phase = np.angle(ifg_stack) if np.iscomplexobj(ifg_stack) else ifg_stack

    num_imgs, rows, cols = ifg_stack.shape
    closures = C @ ifg_phase.reshape((num_imgs, -1))
    closures = closures.reshape(-1, rows, cols)
    if rewrap:
        return rewrap_to_2pi(closures)
    else:
        return closures


def closure_integer_ambiguity(unw_stack, ifg_date_pairs, do_sum=False):
    """Compute the integer ambiguity from the closure phase of unwrapped ifgs

    Can be used to detect unwrapping errors, as a 2pi jump in one interferogram
    will lead to a non-zero integer closure phase

    Args:
        ifg_stack (3D ndarray): stack of interferogram images
            Can be either complex, or the phase floats. Size = (m, r, c)
        ifg_date_pairs (list[tuple]): pairs of (reference, secondary) dates
        rewrap (bool): force the values to be within (-pi, pi)

    Returns:
        ndarray: stack of closure phase images, sized (nc, r, c)
            where nc is the number of rows produced by `build_closure_matrix`
    """
    closures = closure_phase(unw_stack, ifg_date_pairs, rewrap=False)
    # Eq 9, Yunjun 2019
    closure_ambiguities = (closures - rewrap_to_2pi(closures)) / (2 * np.pi)
    if do_sum:
        return (closure_ambiguities != 0).sum(axis=0)
    else:
        return closure_ambiguities


# TODO
def _run_lowess(ts, xs, frac, it):
    import statsmodels.api as sm

    lowess = sm.nonparametric.lowess
    return lowess(ts, xs, frac=frac, it=it)[:, 1]


def lowess_stack(stack, xs, frac=0.5, it=1, workers=None):
    from concurrent.futures import ProcessPoolExecutor
    from itertools import repeat

    timeseries = stack.reshape(stack.shape[0], -1).T
    with ProcessPoolExecutor(max_workers=workers) as executor:
        outputs = executor.map(
            _run_lowess, timeseries, repeat(xs), repeat(frac), repeat(it)
        )
    return np.array(outputs).reshape(stack.shape)


# def plot_bootstrap(x, y, frac=.4, K=100, pct_bootstrap=0.75):
#     # from scipy import stats
#     bootstraps = smooth(x, y, K=K, pct_bootstrap=pct_bootstrap, frac=frac)
#     mean = np.nanmean(bootstraps, axis=1)
#     # stderr = stats.sem(bootstraps, axis=1, nan_policy='omit')
#     stderr = np.nanstd(bootstraps, axis=1, ddof=0) / np.sqrt(bootstraps.shape[1])

#     fig, ax = plt.subplots()
#     ax.fill_between(ts.date, mean-1.96*stderr, mean+1.96*stderr,alpha=0.95)
#     #ax.plot(ts.date, bootstraps,  color='tomato', alpha=0.25)
#     ax.plot(ts.date, mean, color='red')
#     ax.plot(ts.date, ts, '.-')
#     ax.set_title(f"{frac = :.2f}, {K = }, {pct_bootstrap = :.2f}")
#     return bootstraps, ax


def smooth(x, y, frac=0.3, K=1, pct_bootstrap=1.0):
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

    if K > 1:
        return np.stack(
            [
                smooth(x, y, K=1, frac=frac, pct_bootstrap=pct_bootstrap)
                for k in range(K)
            ]
        ).T
    nboot = int(pct_bootstrap * len(x))
    sample_idxs = np.random.choice(len(x), nboot, replace=True)
    y_s = y[sample_idxs]
    x_s = x[sample_idxs]
    return sm_lowess(y_s, x_s, frac=frac, it=1, xvals=x)


def plot_bootstrap_std(x, y, frac=0.4, K=100, pct_bootstrap=1.0, xplot=None):
    bootstraps = smooth(x, y, K=K, pct_bootstrap=pct_bootstrap, frac=frac)
    mean = np.nanmean(bootstraps, axis=1)
    # stderr = stats.sem(bootstraps, axis=1, nan_policy='omit')
    stderr = np.nanstd(bootstraps, axis=1, ddof=0)
    xplot = xplot if xplot is not None else x
    fig, ax = plt.subplots()
    ax.fill_between(xplot, mean - 1.96 * stderr, mean + 1.96 * stderr, alpha=0.25)
    # ax.plot(ts.date, bootstraps,  color='tomato', alpha=0.25)
    ax.plot(xplot, mean, color="red")
    ax.plot(xplot, ts, ".-")
    ax.set_title(f"{frac = :.2f}, {K = }, {pct_bootstrap = :.2f}")
    return bootstraps, ax


class DummyExecutor(Executor):
    def __init__(self, *args, **kwargs):
        self._shutdown = False
        self._shutdownLock = Lock()

    def submit(self, fn, *args, **kwargs):
        with self._shutdownLock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            f = Future()
            try:
                result = fn(*args, **kwargs)
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                f.set_exception(e)
            else:
                f.set_result(result)

            return f

    def shutdown(self, wait=True):
        with self._shutdownLock:
            self._shutdown = True


def solve_l1(
    unw_chunk,
    slclist,
    ifglist,
    phase_to_cm,
    use_B_matrix=False,
):
    import cvxpy as cp

    nifg, nrow, ncol = unw_chunk.shape
    nsar = len(slclist)
    bb = cp.Parameter(shape=nifg)
    if use_B_matrix:
        M = build_B_matrix(slclist, ifglist)
    else:
        M = build_A_matrix(slclist, ifglist)

    x = cp.Variable(shape=M.shape[1])
    objective = cp.Minimize(cp.norm1(M @ x - bb))
    # objective = cp.Minimize(cp.sum_squares(M @ x - bb))
    prob = cp.Problem(objective)

    xs = np.zeros((nsar, nrow * ncol), dtype=np.float32)
    unw_cols = unw_chunk.reshape(nifg, -1)
    for idx in range(nrow * ncol):
        bb.value = unw_cols[:, idx]
        prob.solve()
        xs[1:, idx] = x.value

    if use_B_matrix:
        # the xs are velocities
        timediffs = np.diff(slclist)
        phi_cols = integrate_velocities(xs[1:], timediffs)
        stack = phi_cols.reshape((nsar, nrow, ncol)).astype(np.float32)
    else:
        stack = xs.reshape((nsar, nrow, ncol))
    return stack * phase_to_cm
