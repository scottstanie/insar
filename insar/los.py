"""Functions to deal with Line of sight vector computation
"""
import os
import glob
import numpy as np
import subprocess
# from scipy import interpolate
import sardem.loading

from insar import utils, latlon, timeseries

from insar.log import get_log
logger = get_log()


def record_xyz_los_vector(lon, lat, db_path=".", outfile="./los_vectors.txt", clear=False):
    """Given one (lon, lat) point, find the LOS from Sat to ground

    Function will run through all possible .db files until non-zero vector is computed

    Records answer in outfile, can be read by utils.read_los_output

    Returns:
        db_files [list]: names of files used to find the xyz vectors
        May be multiple if first failed and produced 0s
    """
    if clear:
        open(outfile, 'w').close()
    print("Recording xyz los vectors for %s, %s" % (lat, lon))

    exec_path = os.path.expanduser("~/sentinel/look_angle/losvec/losvec_yj")
    stationname = "'{} {}'".format(lat, lon)  # Record where vec is from

    cur_dir = os.getcwd()  # To return to after
    db_path = os.path.realpath(db_path)
    # Make db_files an iterator so we can run through them
    db_files = glob.iglob(os.path.join(db_path, "*.db*"))
    try:
        db_file = next(db_files)
    except StopIteration:
        raise ValueError("Bad db_path, no .db files found: {}".format(db_path))

    os.chdir(db_path)
    db_files_used = []
    while True:
        db_files_used.append(db_file)
        # print("Changing to directory {}".format(db_path))
        # usage: losvel_yj file_db lat lon stationname outfilename
        cmd = "{} {} {} {} {} {}".format(exec_path, db_file, lat, lon, stationname, outfile)
        # print("Running command:")
        # print(cmd)
        print("Checking db file: %s" % db_file)
        subprocess.check_output(cmd, shell=True)
        _, xyz_list = utils.read_los_output(outfile)
        # if not all((any(vector) for vector in xyz_list)):  # Some corner produced 0s
        if not any(xyz_list[-1]):  # corner produced only 0s
            try:
                db_file = next(db_files)
            except StopIteration:
                print('Ran out of db files!')
                break
        else:
            break

    # print("Returning to {}".format(cur_dir))

    os.chdir(cur_dir)
    return db_files_used


def los_to_enu(los_file=None, lat_lons=None, xyz_los_vecs=None):
    """Converts Line of sight vectors from xyz to ENU

    Can read in the LOS vec file, or take a list `xyz_los_vecs`
    Args:
        los_file (str): file to the recorded LOS vector at lat,lon points
        lat_lons (list[tuple[float]]): list of (lat, lon) coordinares for LOS vecs
        xyz_los_vecs (list[tuple[float]]): list of xyz LOS vectors

    Notes:
        Second two args are the result of read_los_output, mutually
        exclusive with los_file

    Returns:
        ndarray: ENU 3-vectors
    """
    if los_file:
        lat_lons, xyz_los_vecs = utils.read_los_output(los_file)
    return np.array(latlon.convert_xyz_latlon_to_enu(lat_lons, xyz_los_vecs))


def corner_los_vectors(rsc_data, db_path, los_output_file):
    grid_corners = latlon.latlon_grid_corners(**rsc_data)
    # clear the output file:
    open(los_output_file, 'w').close()
    db_files_used = []
    for p in grid_corners:
        db_files = record_xyz_los_vector(*p, db_path=db_path, outfile=los_output_file)
        db_files_used.append(db_files)

    return db_files_used, utils.read_los_output(los_output_file)


def check_corner_differences(asc_dem_rsc, db_path_asc, db_path_desc, asc_los_file, desc_los_file):
    """Finds value range for ENU coefficients of the LOS vectors in dem.rsc area

    Used to see if east, north, and up components vary too much for a single value
    to be used to solve for east + vertical part from LOS components
    """
    grid_corners = latlon.latlon_grid_corners(**asc_dem_rsc)
    # clear the output file:
    open(desc_los_file, 'w').close()
    open(asc_los_file, 'w').close()
    for p in grid_corners:
        record_xyz_los_vector(*p, db_path=db_path_desc, outfile=desc_los_file)
        record_xyz_los_vector(*p, db_path=db_path_asc, outfile=asc_los_file)

    enu_asc = los_to_enu(asc_los_file)
    enu_desc = los_to_enu(desc_los_file)

    # Find range of data for E, N and U
    ranges_asc = np.ptp(enu_asc, axis=0)  # ptp = 'peak to peak' aka range
    ranges_desc = np.ptp(enu_desc, axis=0)
    return np.max(np.stack((ranges_asc, ranges_desc)))


def find_east_up_coeffs(asc_path, desc_path):
    """Find the coefficients for east and up components for LOS deformation

    Args:
        asc_path (str): path to the directory with the ascending sentinel
            timeseries inversion (contains line-of-sight deformation.npy, dem.rsc,
            and has .db files one directory higher)
        desc_path (str): same as asc_path but for descending orbit solution

    Returns:
        ndarray: east_up_coeffs contains 4 numbers for solving east-up deformation:
            [east_asc,  up_asc;
             east_desc, up_desc]
        Used as the "A" matrix for solving Ax = b, where x is [east_def; up_def]
    """
    asc_path = os.path.realpath(asc_path)
    desc_path = os.path.realpath(desc_path)
    asc_dem_rsc = sardem.loading.load_dem_rsc(os.path.join(asc_path, 'dem.rsc'), lower=True)

    midpoint = latlon.latlon_grid_midpoint(**asc_dem_rsc)
    # The path to each orbit's .db files: assumed one directory higher
    db_path_asc = os.path.dirname(asc_path)
    db_path_desc = os.path.dirname(desc_path)

    asc_los_file = os.path.realpath(os.path.join(asc_path, 'los_vectors.txt'))
    desc_los_file = os.path.realpath(os.path.join(desc_path, 'los_vectors.txt'))

    max_corner_difference = check_corner_differences(asc_dem_rsc, db_path_asc, db_path_desc,
                                                     asc_los_file, desc_los_file)
    logger.info(
        "Max difference in ENU LOS vectors for area corners: {:2f}".format(max_corner_difference))
    if max_corner_difference > 0.05:
        logger.warning("Area is not small, actual LOS vector differs over area.")
    logger.info("Using midpoint of area for line of sight vectors")

    print("Finding LOS vector for midpoint", midpoint)
    record_xyz_los_vector(*midpoint, db_path=db_path_asc, outfile=asc_los_file, clear=True)
    record_xyz_los_vector(*midpoint, db_path=db_path_desc, outfile=desc_los_file, clear=True)

    enu_asc = los_to_enu(asc_los_file)
    enu_desc = los_to_enu(desc_los_file)

    # Get only East and Up out of ENU
    eu_asc = enu_asc[:, ::2]
    eu_desc = enu_desc[:, ::2]
    # -1 multiplied since vectors are from sat to ground, so vert is negative
    east_up_coeffs = -1 * np.vstack((eu_asc, eu_desc))

    return east_up_coeffs


def find_vertical_def(asc_path, desc_path):
    """Calculates vertical deformation for all points in the LOS files

    Args:
        asc_path (str): path to the directory with the ascending sentinel
            timeseries inversion (contains line-of-sight deformation.npy, dem.rsc,
            and has .db files one directory higher)
        desc_path (str): same as asc_path but for descending orbit solution
    Returns:
        tuple[ndarray, ndarray]: def_east, def_vertical, the two matrices of
            deformation separated by verticl and eastward motion
    """
    asc_path = os.path.realpath(asc_path)
    desc_path = os.path.realpath(desc_path)

    east_up_coeffs = find_east_up_coeffs(asc_path, desc_path)
    print("East-up asc and desc:")
    print(east_up_coeffs)

    asc_geolist, asc_deform = timeseries.load_deformation(asc_path)
    desc_geolist, desc_deform = timeseries.load_deformation(desc_path)

    assert asc_deform.shape == desc_deform.shape, 'Asc and desc def images not same size'
    nlayers, nrows, ncols = asc_deform.shape
    # Stack and solve for the East and Up deformation
    d_asc_desc = np.vstack([asc_deform.reshape(-1), desc_deform.reshape(-1)])
    dd = np.linalg.solve(east_up_coeffs, d_asc_desc)
    def_east = dd[0, :].reshape((nlayers, nrows, ncols))
    def_vertical = dd[1, :].reshape((nlayers, nrows, ncols))
    return def_east, def_vertical


# def interpolate_coeffs(rsc_data, nrows, ncols, east_up):
#     # This will be if we want to solve the exact coefficients
#     # Make grid to interpolate one
#     grid_corners = latlon.latlon_grid_corners(**rsc_data)
#     xx, yy = latlon.latlon_grid(sparse=True, **rsc_data)
#     interpolated_east_up = np.empty((2, nrows, ncols))
#     for idx in (0, 1):
#         component = east_up[:, idx]
#         interpolated_east_up[idx] = interpolate.griddata(
#             points=grid_corners, values=component, xi=(xx, yy))
#     interpolated_east_up = interpolated_east_up.reshape((2, nrows * ncols))
