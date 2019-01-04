import numpy as np
import math
from pyinsar.processing.deformation.elastic_halfspace import okada
from pyinsar.processing.geography.coordinates import compute_x_and_y_coordinates_maps


def setup_area():
    master_width = 1000
    master_height = 1000

    master_x_min = -20000  # m
    master_x_max = 20000  # m
    master_y_min = -20000  # m
    master_y_max = 20000  # m

    master_extent = (master_x_min / 1000., master_x_max / 1000., master_y_min / 1000.,
                     master_y_max / 1000.)  # km

    xx, yy = compute_x_and_y_coordinates_maps(master_x_min, master_x_max, master_y_min,
                                              master_y_max, master_width, master_height)
    return xx, yy


def make_fault():
    xx, yy = setup_area()

    fault_centroid_x = 0.
    fault_centroid_y = 0.
    fault_top_depth = 5000.  # m

    fault_strike = 200 * np.pi / 180.  # rad
    fault_dip = 10 * np.pi / 180.  # rad
    fault_length = 15000.  # m
    fault_width = .2 * 6000.  # m
    fault_rake = -90 * np.pi / 180.  # rad
    fault_slip = 1.  # m
    fault_open = 0.  # m

    poisson_ratio = 0.25

    fault_centroid_depth = fault_top_depth + math.sin(fault_dip) * fault_width / 2.
    return okada.compute_okada_displacement(
        fault_centroid_x, fault_centroid_y, fault_centroid_depth, fault_strike, fault_dip,
        fault_length, fault_width, fault_rake, fault_slip, fault_open, poisson_ratio, xx, yy)
