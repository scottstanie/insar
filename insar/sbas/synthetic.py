import argparse
import itertools
from collections import OrderedDict
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from insar import timeseries, mask, sario
import sardem.loading


class IgramMaker:
    def __init__(self, num_days=5, shape=(4, 4), constant_vel=False):
        # 5 geos made so that the most igrams any date can make is
        # n-1 = 4 igrams (but 10 total igrams among all)
        self.num_days = num_days
        self.shape = shape
        self.constant_vel = constant_vel

    def save_dem_rsc(self):
        rsc_dict = OrderedDict([('width', 2), ('file_length', 3),
                                ('x_first', -155.676388889),
                                ('y_first', 19.5755555567),
                                ('x_step', 0.000138888888),
                                ('y_step', -0.000138888888),
                                ('x_unit', 'degrees'), ('y_unit', 'degrees'),
                                ('z_offset', 0), ('z_scale', 1),
                                ('projection', 'LL')])
        with open("dem.rsc", "w") as f:
            f.write(sardem.loading.format_dem_rsc(rsc_dict))

    def make_geo_dates(self):
        geo_date_list = []
        for idx in range(self.num_days):
            geo_date_list.append(datetime(2018, 1, 1 + idx))
        return geo_date_list

    def make_geos(self):
        # Make 4 dummy 4x4 arrays of geo
        # bottom row will be 0s
        geo_base = np.ones(self.shape).astype(np.complex64)
        geo_list = [geo_base]
        for idx in range(1, self.num_days):
            next_geo = geo_list[idx - 1].copy() + idx
            geo_list.append(next_geo)
        return geo_list

    def mask_geos(self, geo_list, geo_date_list):
        # # Method 1: make masks manually
        # # Add masks: 1 and 3 are all good, 2 and 4 have dead cells
        # mask1, mask2, mask3, mask4, mask5 = np.zeros((5, 4, 4))
        #
        # mask2[3, 3] = 1
        # mask4[3, 3] = 1
        # mask4[0, 0] = 1

        # Method 2: make some pixels 0, and run mask functions
        geo_list[1][3, 3] = 0
        geo_list[3][3, 3] = 0

        geo_list[2][0, 0] = 0

        truth_geos = np.stack(geo_list, axis=0)

        for idx, geo in enumerate(truth_geos):
            fname = 'S1A_{}.geo'.format(geo_date_list[idx].strftime('%Y%m%d'))
            geo.tofile(fname)

        # geos_masked_list = [
        #     np.ma.array(geo1, mask=mask1, fill_value=np.NaN),
        #     np.ma.array(geo2, mask=mask2, fill_value=np.NaN),
        #     np.ma.array(geo3, mask=mask3, fill_value=np.NaN),
        #     np.ma.array(geo4, mask=mask4, fill_value=np.NaN),
        #     np.ma.array(geo5, mask=mask5, fill_value=np.NaN),
        # ]
        # geos_masked = np.ma.stack(geos_masked_list, axis=0)
        return truth_geos

    def make_igrams(self, geo_date_list, truth_geos):
        igram_list = []  # list of arrays
        igram_date_list = []  # list of tuples of dates
        igram_fname_list = []  # list of strings
        for early_idx, late_idx in itertools.combinations(
                range(len(truth_geos)), 2):
            # Using masked data
            # early, late = geos_masked[early_idx], geos_masked[late_idx]
            # Using truth data to form igrams:
            early, late = truth_geos[early_idx], truth_geos[late_idx]

            early_date, late_date = geo_date_list[early_idx], geo_date_list[
                late_idx]
            igram_date_list.append((early_date, late_date))

            igram = np.abs(late) - np.abs(early)
            fname = '{}_{}.int'.format(early_date.strftime('%Y%m%d'),
                                       late_date.strftime('%Y%m%d'))
            igram.tofile(fname)

            igram_fname_list.append(fname)
            igram_list.append(igram)

        return igram_list, igram_fname_list, igram_date_list

    def mask_igrams(self, igram_fname_list, igram_date_list, geo_date_list):
        mask.save_int_masks(igram_fname_list,
                            igram_date_list,
                            geo_date_list,
                            geo_path='.')

    def run_inversion(self):
        self.save_dem_rsc()
        geo_date_list = self.make_geo_dates()
        timediffs = timeseries.find_time_diffs(geo_date_list)

        geo_list = self.make_geos()
        truth_geos = self.mask_geos(geo_list, geo_date_list)

        igram_list, igram_fname_list, igram_date_list = self.make_igrams(
            geo_date_list,
            truth_geos,
        )
        self.mask_igrams(igram_fname_list, igram_date_list, geo_date_list)

        geo_masks = sario.load_stack(directory='.', file_ext='.geo.mask.npy')
        geo_mask_columns = timeseries.stack_to_cols(geo_masks)

        int_mask_file_names = [f + '.mask.npy' for f in igram_fname_list]
        int_mask_stack = sario.load_stack(file_list=int_mask_file_names)
        igram_stack = np.ma.stack(igram_list, axis=0)
        igram_stack.mask = int_mask_stack

        columns_masked = timeseries.stack_to_cols(igram_stack)

        # columns_with_masks = np.ma.count_masked(columns_masked, axis=0)
        B = timeseries.build_B_matrix(geo_date_list, igram_date_list)

        varr = timeseries.invert_sbas(columns_masked,
                                      B,
                                      geo_mask_columns,
                                      constant_vel=self.constant_vel)
        phi_hat = timeseries.integrate_velocities(varr, timediffs)
        phi_hat = timeseries.cols_to_stack(phi_hat, *geo_list[0].shape)
        return geo_date_list, phi_hat, truth_geos


def plot_results(geo_date_list, phi_hat, truth_geos, max_row=2, max_col=2):
    fig, axes = plt.subplots(max_row, max_col)
    for ridx in range(max_row):
        for cidx in range(max_col):
            print("Row, col", ridx, cidx)
            print('truth:', truth_geos[:, ridx, cidx])
            print('est:', phi_hat[:, ridx, cidx])
            axes[ridx, cidx].plot(geo_date_list,
                                  truth_geos[:, ridx, cidx],
                                  marker='o',
                                  linestyle='dashed',
                                  linewidth=1,
                                  markersize=4)
            axes[ridx, cidx].plot(geo_date_list,
                                  phi_hat[:, ridx, cidx],
                                  marker='o',
                                  linestyle='dashed',
                                  linewidth=1,
                                  markersize=4)

    plt.show(block=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constant-vel", action="store_true", default=False)
    parser.add_argument("--plot-rows", default=4)
    parser.add_argument("--plot-cols", default=4)
    args = parser.parse_args()

    igram_maker = IgramMaker(constant_vel=args.constant_vel)
    geo_date_list, phi_hat, truth_geos = igram_maker.run_inversion()

    # print(phi_hat.astype(int))
    normed_truth = np.abs(np.ma.masked_equal(truth_geos, 0)) - 1
    # print(normed_truth.astype(int))
    plot_results(
        geo_date_list,
        phi_hat,
        normed_truth,
        max_row=args.plot_rows,
        max_col=args.plot_cols,
    )
