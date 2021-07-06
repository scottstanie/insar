
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
        model (str): If 'linear', creates the M x 1 matrix for linear velo model

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
