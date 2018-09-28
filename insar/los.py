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
    grid_corners = latlon.grid_corners(**rsc_data)
    # clear the output file:
    open(los_output_file, 'w').close()
    db_files_used = []
    for p in grid_corners:
        db_files = record_xyz_los_vector(*p, db_path=db_path, outfile=los_output_file)
        db_files_used.append(db_files)

    return db_files_used, utils.read_los_output(los_output_file)


def check_corner_differences(rsc_data, db_path, los_file):
    """Finds value range for ENU coefficients of the LOS vectors in dem.rsc area

    Used to see if east, north, and up components vary too much for a single value
    to be used to solve for east + vertical part from LOS components
    """
    grid_corners = latlon.grid_corners(**rsc_data)
    # clear the output file:
    open(los_file, 'w').close()
    for p in grid_corners:
        record_xyz_los_vector(*p, db_path=db_path, outfile=los_file)

    enu_coeffs = los_to_enu(los_file)

    # Find range of data for E, N and U
    enu_ranges = np.ptp(enu_coeffs, axis=0)  # ptp = 'peak to peak' aka range
    return np.max(enu_ranges), enu_coeffs


def find_east_up_coeffs(geo_path):
    """Find the coefficients for east and up components for LOS deformation

    Args:
        geo_path (str): path to the directory with the sentinel
            timeseries inversion (contains line-of-sight deformation.npy, dem.rsc,
            and has .db files one directory higher)

    Returns:
        ndarray: east_up_coeffs, a 1x2 array [[east_def, up_def]]
        Combined with another path, used for solving east-up deformation.:
            [east_asc,  up_asc;
             east_desc, up_desc]
        Used as the "A" matrix for solving Ax = b, where x is [east_def; up_def]
    """
    # TODO: make something to adjust 'params' file in case we moved it
    geo_path = os.path.realpath(geo_path)
    # Are we doing this in the .geo folder, or the igram folder?
    # rsc_data = sardem.loading.load_dem_rsc(os.path.join(geo_path, 'dem.rsc'), lower=True)
    rsc_data = sardem.loading.load_dem_rsc(os.path.join(geo_path, 'elevation.dem.rsc'), lower=True)

    midpoint = latlon.grid_midpoint(**rsc_data)
    # The path to each orbit's .db files assumed in same directory as elevation.dem.rsc

    los_file = os.path.realpath(os.path.join(geo_path, 'los_vectors.txt'))
    db_path = os.path.join(geo_path, 'extra_files') if os.path.exists(
        os.path.join(geo_path, 'extra_files')) else geo_path

    max_corner_difference, enu_coeffs = check_corner_differences(rsc_data, db_path, los_file)
    logger.info(
        "Max difference in ENU LOS vectors for area corners: {:2f}".format(max_corner_difference))
    if max_corner_difference > 0.05:
        logger.warning("Area is not small, actual LOS vector differs over area.")
        logger.info('Corner ENU coeffs:')
        logger.info(enu_coeffs)
    logger.info("Using midpoint of area for line of sight vectors")

    print("Finding LOS vector for midpoint", midpoint)
    record_xyz_los_vector(*midpoint, db_path=db_path, outfile=los_file, clear=True)

    enu_coeffs = los_to_enu(los_file)

    # Get only East and Up out of ENU
    east_up_coeffs = enu_coeffs[:, ::2]
    # -1 multiplied since vectors are from sat to ground, so vert is negative
    return -1 * east_up_coeffs


def find_vertical_def(asc_path, desc_path):
    """Calculates vertical deformation for all points in the LOS files

    Args:
        asc_path (str): path to the directory with the ascending sentinel files
            Should contain elevation.dem.rsc, .db files, and igram folder
        desc_path (str): same as asc_path but for descending orbit
    Returns:
        tuple[ndarray, ndarray]: def_east, def_vertical, the two matrices of
            deformation separated by verticl and eastward motion
    """
    asc_path = os.path.realpath(asc_path)
    desc_path = os.path.realpath(desc_path)

    eu_asc = find_east_up_coeffs(asc_path)
    eu_desc = find_east_up_coeffs(desc_path)

    east_up_coeffs = np.vstack((eu_asc, eu_desc))
    print("East-up asc and desc:")
    print(east_up_coeffs)

    asc_igram_path = os.path.join(asc_path, 'igrams')
    desc_igram_path = os.path.join(desc_path, 'igrams')

    asc_geolist, asc_deform = timeseries.load_deformation(asc_igram_path)
    desc_geolist, desc_deform = timeseries.load_deformation(desc_igram_path)

    print(asc_igram_path, asc_deform.shape)
    print(desc_igram_path, desc_deform.shape)
    assert asc_deform.shape == desc_deform.shape, 'Asc and desc def images not same size'
    nlayers, nrows, ncols = asc_deform.shape
    # Stack and solve for the East and Up deformation
    d_asc_desc = np.vstack([asc_deform.reshape(-1), desc_deform.reshape(-1)])
    dd = np.linalg.solve(east_up_coeffs, d_asc_desc)
    def_east = dd[0, :].reshape((nlayers, nrows, ncols))
    def_vertical = dd[1, :].reshape((nlayers, nrows, ncols))
    return def_east, def_vertical


def merge_geolists(geolist1, geolist2):
    """Task asc and desc geolists, makes one merged

    Gives the overlap indices of the merged list for each smaller

    """
    merged_geolist = np.concatenate((geolist1, geolist2))
    merged_geolist.sort()

    _, indices1, _ = np.intersect1d(merged_geolist, geolist1, return_indices=True)
    _, indices2, _ = np.intersect1d(merged_geolist, geolist2, return_indices=True)
    return merged_geolist, indices1, indices2


# def interpolate_coeffs(rsc_data, nrows, ncols, east_up):
#     # This will be if we want to solve the exact coefficients
#     # Make grid to interpolate one
#     grid_corners = latlon.grid_corners(**rsc_data)
#     xx, yy = latlon.grid(sparse=True, **rsc_data)
#     interpolated_east_up = np.empty((2, nrows, ncols))
#     for idx in (0, 1):
#         component = east_up[:, idx]
#         interpolated_east_up[idx] = interpolate.griddata(
#             points=grid_corners, values=component, xi=(xx, yy))
#     interpolated_east_up = interpolated_east_up.reshape((2, nrows * ncols))
