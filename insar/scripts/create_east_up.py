"""Save the deformation outputs to a series of TIF files

python ~/repos/insar/helpers/create_east_up.py --directory /data1/scott/pecos/path78-bbox2/igrams_looked --path_num 78 run
"""
import fire
import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import tqdm
import xarray as xr
from apertools import deramp, gps, utils
from apertools.log import get_log, log_runtime

logger = get_log()


cmap = "seismic_wide_y_r"

BAD_GPS_STATIONS = [
    # http://geodesy.unr.edu/NGLStationPages/stations/RG08.sta
    # equipment change caused a big jump
    # "RG08",
    # http://geodesy.unr.edu/NGLStationPages/stations/MDO1.sta
    # Many jumps and gaps
    "MD01",
    # Stops at year 2016.5
    "TXPC",
]


# TODO: save attrs from the deformation.nc?
@dataclass
class LOS:
    directory: str
    path_num: int
    defo_filename: str = "deformation.nc"
    dset_name: str = "defo_lowess"
    los_dset: str = "los_enu"  # contained in the defo_filename
    los_map_filename: str = "los_enu.tif"

    out_directory: str = None

    # Masking
    mask_missing_threshold: int = 8
    cor_thresh: float = 0.11
    max_abs_val_masking: float = 2

    # deramping
    deramp_order = 1

    # Figure saving
    figure_directory: str = "figures"
    vm: float = 7  # Color limits (vmin, vmax)
    figheight = 6
    figwidth = 8

    def run(self):
        self.figsize = (self.figwidth, self.figheight)
        if self.out_directory is None:
            self.out_directory = Path(self.directory) / f"los_out_{self.path_num}/"
            utils.mkdir_p(self.out_directory)
        logger.info("Saving output to %s", self.out_directory)

        with xr.open_dataset(Path(self.directory) / self.defo_filename) as ds:
            self.ds = ds
        logger.info("Running GPS comparison")
        self.compare_gps()

    def _set_full_paths(self):
        self._set_abs_path(self.defo_filename)
        self._set_abs_path(self.los_map_filename)
        self._set_abs_path(self.figure_directory)

    def _set_abs_path(self, filename):
        """Allows a fully qualified path through; otherwise, appends to the directory"""
        if os.path.abspath(filename) != filename:
            return Path(self.directory) / filename
        return filename

    def record(self, filename):
        utils.record_params_as_yaml(filename, **asdict(self))

    def create_cor_mask(self, figname, **figkwargs):
        import proplot as pplt
        fig, axes = pplt.subplots(ncols=3, figsize=(self.figsize))
        ax = axes[0]
        cor = self.ds["cor_mean"]
        cor.plot.imshow(cmap="gray", ax=ax)

        ax = axes[1]
        self.cor_mask = cor < self.cor_thresh
        ax.imshow(self.cor_mask.astype(int))

        ax = axes[2]
        test_img = self.ds[self.dset_name][-1].copy()
        test_img.data[self.cor_mask.data] = np.nan
        test_img.plot.imshow(cmap=cmap, vmin=-self.vm, vmax=self.vm, ax=ax)

        self._save_figure(fig, figname, **figkwargs)

    def get_mask(self, datapath: str, mask_filename: str):
        # with h5py.File(self.data78 + "masks.h5") as hf:
        with h5py.File(datapath + self.mask_filename) as hf:
            msum = hf["slc_sum"][()]

        self.mask_missing = msum > self.mask_missing_threshold
        return self.mask_missing

    def remove_ramp(
        self,
        ds: xr.Dataset,
        dset_name: str = "defo_lowess",
        # out_filename: str = "defo_lowess_ramp_removed.nc",
        deramp_order: int = 1,
    ):
        ds_deramped = deramp.remove_ramp_xr(
            ds,
            dset_name=dset_name,
            deramp_order=deramp_order,
            # outfile=data78 + defo_fname.replace(".nc", "_finalderamped.nc"),
            mask=self.mask_missing,
            # overwrite=True,
            mask_val=0,
            max_abs_val=self.max_abs_val_masking,
        )
        return ds_deramped
        # tmp = ds_deramped
        # ds78_raw = ds78
        # ds78 = tmp

    def _save_figure(self, fig, fname, figkwargs={}):
        utils.mkdir_p(self.figure_directory)
        fig.savefig(fname, **figkwargs)

    def compare_gps(self):
        igc = gps.InsarGPSCompare(
            #     insar_filename=data78 + defo_fname,
            insar_ds=self.ds,
            dset=self.dset_name,
            los_dset=self.los_dset,
            los_map_file=self.los_map_filename,
        )
        df, df_diff = igc.run()
        df.to_csv(self.out_directory / "insar_gps.csv")
        df_diff.to_csv(self.out_directory / "insar_gps_diffs.csv")


# @log_runtime
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--path-num", "-p", required=True, help="InSAR path of the deformation."
#     )
#     parser.add_argument(
#         "--directory", "-d", required=True, help="Directory containing the deformation."
#     )
#     args = parser.parse_args()
   

if __name__ == "__main__":
    fire.Fire(LOS)
    # main()
