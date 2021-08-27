import numpy as np
from matplotlib.dates import date2num
from itertools import chain, combinations


def build_A_matrix(sar_date_list, ifg_date_list):
    """Takes the list of igram dates and builds the SBAS A matrix

    Args:
        sar_date_list (list[date]): datetimes of the acquisitions
        ifg_date_list (list[tuple(date, date)])

    Returns:
        np.array 2D: the incident-like matrix from the SBAS paper: A*phi = dphi
            Each row corresponds to an igram, each column to a SAR date
            value will be -1 on the early (reference) igrams, +1 on later (secondary)
    """
    # We take the first .geo to be time 0, leave out of matrix
    # Match on date (not time) to find indices
    sar_date_list = sar_date_list[1:]
    M = len(ifg_date_list)  # Number of igrams, number of rows
    N = len(sar_date_list)
    A = np.zeros((M, N))
    for j in range(M):
        early_igram, late_igram = ifg_date_list[j]

        try:
            idx_early = sar_date_list.index(early_igram)
            A[j, idx_early] = -1
        except ValueError:  # The first SLC will not be in the matrix
            pass

        idx_late = sar_date_list.index(late_igram)
        A[j, idx_late] = 1

    return A


def build_B_matrix(sar_dates, ifg_date_list, model=None):
    """Takes the list of igram dates and builds the SBAS B (velocity coeff) matrix

    Args:
        sar_date_list (list[date]): dates of the SAR acquisitions
        ifg_date_list (list[tuple(date, date)])
        model (str): If "linear", creates the M x 1 matrix for linear velo model

    Returns:
        np.array: 2D array of the velocity coefficient matrix from the SBAS paper:
                Bv = dphi
            Each row corresponds to an igram, each column to a SAR date
            value will be t_k+1 - t_k for columns after the -1 in A,
            up to and including the +1 entry
    """
    try:
        sar_dates = sar_dates.date
    except AttributeError:
        pass
    timediffs = np.array([difference.days for difference in np.diff(sar_dates)])

    A = build_A_matrix(sar_dates, ifg_date_list)
    B = np.zeros_like(A)

    for j, row in enumerate(A):
        # if no -1 entry, start at index 0. Otherwise, add 1 to exclude the -1 index
        start_idx = list(row).index(-1) + 1 if (-1 in row) else 0
        # End index is inclusive of the +1
        end_idx = np.where(row == 1)[0][0] + 1  # +1 will always exist in row

        # Now only fill in the time diffs in the range from the early igram index
        # to the later igram index
        B[j][start_idx:end_idx] = timediffs[start_idx:end_idx]

    if model == "linear":
        return B.sum(axis=1, keepdims=True)
    else:
        return B


def A_polynomial(sar_dates, degree=1):
    """System matrix for a polynomial fit to data

    Args:
        sar_dates (iterable[date]): dates of the SAR acquisitions

    Returns:
        A, size=(len(sar_date_list)) 2D Vandermonde array to solve for the polynomial coefficients
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

    return xr.apply_ufunc(
        np.ptp, da, input_core_dims=[["lat", "lon"]], kwargs={"axis": (-2, -1)}
    )


def ptp_by_date_pct(da, low=0.05, high=0.95):
    """Find the peak-to-peak amplitude per image, based on the `high`/`low` percentiles"""
    import xarray as xr

    high_q = da.quantile(high, dim=("lat", "lon"))
    low_q = da.quantile(low, dim=("lat", "lon"))
    return high_q - low_q


def build_closure_matrix(ifg_date_list, closure_list=None):
    """Takes the list of igram dates and builds the SBAS A matrix

    Args:
        sar_date_list (list[date]): datetimes of the acquisitions
        ifg_date_list (list[tuple(date, date)])

    Returns:
        np.array 2D: the incident-like matrix from the SBAS paper: A*phi = dphi
            Each row corresponds to an igram, each column to a SAR date
            value will be -1 on the early (reference) igrams, +1 on later (secondary)
    """

    if closure_list is None:
        sar_date_list = list(sorted(set(chain.from_iterable(ifg_date_list))))
        closure_list = list(combinations(sar_date_list, 3))

    ifg_to_idx = {ifg: idx for idx, ifg in enumerate(ifg_date_list)}

    # N = len(sar_date_list)
    M = len(ifg_date_list)  # Number of igrams, number of rows
    K = len(closure_list)
    row = np.zeros(M)
    C_list = []
    for j in range(K):
        trip = closure_list[j]

        ifg1 = (trip[0], trip[1])
        ifg2 = (trip[1], trip[2])
        ifg3 = (trip[0], trip[2])
        try:
            idx1 = ifg_to_idx[ifg1]
            idx2 = ifg_to_idx[ifg2]
            idx3 = ifg_to_idx[ifg3]
        except KeyError:
            continue

        row[:] = 0
        row[idx1] = 1
        row[idx2] = 1
        row[idx3] = -1
        C_list.append(row)

    return np.stack(C_list)


def get_design_matrix4triplet(date12_list):
    """Generate the design matrix of ifgram triangle for unwrap error correction using phase closure
    Parameters: date12_list : list of string in YYYYMMDD_YYYYMMDD format
    Returns:    C : 2D np.array in size of (num_tri, num_ifgram) consisting 0, 1, -1
                    for 3 SAR acquisition in t1, t2 and t3 in time order,
                    ifg1 for (t1, t2) with 1
                    ifg2 for (t1, t3) with -1
                    ifg3 for (t2, t3) with 1
    Examples:   obj = ifgramStack('./inputs/ifgramStack.h5')
                date12_list = obj.get_date12_list(dropIfgram=True)
                C = ifgramStack.get_design_matrix4triplet(date12_list)
    """
    # Date info
    date12_list = list(date12_list)

    # calculate triangle_idx
    triangle_idx = []
    for ifgram1 in date12_list:
        # ifgram1 (date1, date2)
        date1, date2 = ifgram1.split("_")

        # ifgram2 candidates (date1, date3)
        date3_list = []
        for ifgram2 in date12_list:
            if date1 == ifgram2.split("_")[0] and ifgram2 != ifgram1:
                date3_list.append(ifgram2.split("_")[1])

        # ifgram2/3
        if len(date3_list) > 0:
            for date3 in date3_list:
                ifgram3 = "{}_{}".format(date2, date3)
                if ifgram3 in date12_list:
                    ifgram1 = "{}_{}".format(date1, date2)
                    ifgram2 = "{}_{}".format(date1, date3)
                    ifgram3 = "{}_{}".format(date2, date3)
                    triangle_idx.append(
                        [
                            date12_list.index(ifgram1),
                            date12_list.index(ifgram2),
                            date12_list.index(ifgram3),
                        ]
                    )

    if len(triangle_idx) == 0:
        print(
            "\nWARNING: No triangles found from input date12_list:\n{}!\n".format(
                date12_list
            )
        )
        return None

    triangle_idx = np.array(triangle_idx, np.int16)
    triangle_idx = np.unique(triangle_idx, axis=0)

    # triangle_idx to C
    num_triangle = triangle_idx.shape[0]
    C = np.zeros((num_triangle, len(date12_list)), np.float32)
    for i in range(num_triangle):
        C[i, triangle_idx[i, 0]] = 1
        C[i, triangle_idx[i, 1]] = -1
        C[i, triangle_idx[i, 2]] = 1
    return C
