#!/usr/bin/env python
"""Save the deformation outputs to a series of TIF files

python ~/repos/insar/helpers/create_east_up.py --directory /data1/scott/pecos/path78-bbox2/igrams_looked --path_num 78 run
"""
import os
import re
import glob
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import fire
import h5py
import numpy as np
import pandas as pd
import tqdm
import xarray as xr
from apertools import deramp, los, gps, gps_plots, utils, subset
from apertools.log import get_log, log_runtime

logger = get_log()

# TODO: ignore this bad stations
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

# TODO: load a previous YAML file to redo without all the CLI args


# TODO: save attrs from the deformation.nc?
@dataclass
class LOS:
    directory: str
    path_num: int
    defo_filename: str = "deformation.nc"
    dset_name: str = "defo_lowess"
    shifted_dset_name: str = "defo_lowess_shifted"
    los_dset: str = "los_enu"  # contained in `defo_filename`
    los_map_filename: str = "los_enu.tif"
    cor_mean_filename: str = "cor_mean.tif"

    out_directory: str = None

    # GPS comparison
    gps_window_size: int = 5
    ref_station: str = None
    gps_max_nan_pct: float = 0.5

    # Masking
    mask_filename: str = "masks.h5"
    mask_missing_threshold: int = 8
    cor_thresh: float = 0.11
    max_abs_val_masking: float = 2

    # deramping
    do_final_deramp: bool = True
    deramp_order: int = 2

    # Output options:
    outfile_template: str = "cumulative_los_path{path_num}_{dt}.tif"
    freq: str = "6M"
    crs: str = "EPSG:4326"

    # Figure saving
    figure_directory: str = "figures"
    vm: float = 7  # Color limits (vmin, vmax)
    figheight: float = 6
    figwidth: float = 8
    defo_cmap: str = "seismic_wide_y_r"

    def run(self):
        self.figsize = (self.figwidth, self.figheight)

        if self.out_directory is None:
            p = Path(self.directory) / f"los_out_{self.path_num}/"
            self.out_directory = p.resolve()
            utils.mkdir_p(self.out_directory)
        else:
            self.out_directory = Path(self.out_directory).resolve()
        self._set_full_paths()
        logger.info("Saving output to %s", self.out_directory)

        self.ds = xr.open_dataset(Path(self.directory) / self.defo_filename)
        self.mask_missing = self.get_mask()
        if self.do_final_deramp:
            logger.info("Deramping deformation")
            ds_deramped = self.remove_ramp(self.ds, self.dset_name, self.deramp_order)
            # Swap places of old and new, so future checks use the deramped
            tmp = ds_deramped
            self.ds_raw = self.ds
            self.ds = tmp
        else:
            ds_deramped = self.ds

        logger.info("Running GPS comparison")
        self.compare_gps(self.dset_name)
        if self.ref_station:
            logger.info("Finding correction from GPS station %s", self.ref_station)
            self.reference_to_gps()
            self.compare_gps(self.shifted_dset_name)
        else:
            logger.info("No GPS reference station specified")
            self.ds[self.shifted_dset_name] = self.ds[self.dset_name]

        # Apply the mask
        self.set_mask()
        # Save output tifs
        self.save_output()

        # Also save a quicklook plot of the mask and final deformation
        logger.info("Saving quicklook plots")
        self.plot_cor_mask(self.figure_directory / "cor_mask.pdf")
        self.plot_img(idx=-1)

        # record all aspects of the run
        self.record(self.out_directory / "run_params.yaml")
        self.ds.close()

    def _set_full_paths(self):
        self.defo_filename = self._set_abs_path(self.defo_filename)
        self.los_map_filename = self._set_abs_path(self.los_map_filename)
        self.mask_filename = self._set_abs_path(self.mask_filename)
        self.outfile_template = self._set_abs_path(
            self.out_directory / self.outfile_template
        )
        self.figure_directory = self._set_abs_path(
            self.out_directory / self.figure_directory
        )
        utils.mkdir_p(self.figure_directory)

    def _set_abs_path(self, filename):
        """Allows a fully qualified path through; otherwise, appends to the directory"""
        if os.path.abspath(filename) != filename:
            return Path(self.directory) / filename
        return filename

    def record(self, filename):
        self_dict = asdict(self)
        for k, v in self_dict.items():
            if isinstance(v, Path):
                self_dict[k] = str(v)
        utils.record_params_as_yaml(filename, **self_dict)

    def plot_img(self, idx=-1, date=None, dset_name="defo_lowess_shifted"):
        import proplot as pplt

        if date is not None:
            da = self.ds[dset_name]
        else:
            da = self.ds[dset_name][idx]
        fig, ax = pplt.subplots(figsize=self.figsize)
        da.plot.imshow(cmap=self.defo_cmap, vmin=-self.vm, vmax=self.vm, ax=ax)
        self._save_figure(fig, f"{self.figure_directory}/deformation.pdf")

    def get_cor_mask(self):
        cor = self.ds["cor_mean"]
        return cor < self.cor_thresh

    def plot_cor_mask(self, figname, **figkwargs):
        import proplot as pplt

        fig, axes = pplt.subplots(
            ncols=3, figsize=(2 * self.figwidth, 0.7 * self.figheight)
        )

        ax = axes[0]
        cor = self.ds["cor_mean"]
        cor.plot.imshow(cmap="gray", ax=ax)
        cor_mask = self.get_cor_mask()

        ax = axes[1]
        cor_mask.astype(int).plot.imshow(ax=ax)

        ax = axes[2]
        test_img = self.ds[self.dset_name][-1].copy()
        test_img.data[cor_mask.data] = np.nan
        test_img.plot.imshow(cmap=self.defo_cmap, vmin=-self.vm, vmax=self.vm, ax=ax)

        self._save_figure(fig, figname, **figkwargs)

    def get_mask(self):
        # with h5py.File(self.data78 + "masks.h5") as hf:
        with h5py.File(self.mask_filename) as hf:
            msum = hf["slc_sum"][()]

        mask_missing = msum > self.mask_missing_threshold
        return mask_missing

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
            outfile=str(self.defo_filename).replace(
                ".nc", f"_finalderamped_order{deramp_order}.nc"
            ),
            mask=self.mask_missing,
            # overwrite=True,
            mask_val=0,
            max_abs_val=self.max_abs_val_masking,
        )
        return ds_deramped

    def _save_figure(self, fig, fname, figkwargs={}):
        utils.mkdir_p(self.figure_directory)
        fig.savefig(fname, **figkwargs)

    def compare_gps(self, dset_name):
        igc = gps.InsarGPSCompare(
            insar_ds=self.ds,
            dset=dset_name,
            los_dset=self.los_dset,
            los_map_file=self.los_map_filename,
            window_size=self.gps_window_size,
            max_nan_pct=self.gps_max_nan_pct,
        )
        df, df_diff = igc.run()
        df.to_csv(self.out_directory / "insar_gps.csv")
        df_diff.to_csv(self.out_directory / "insar_gps_diffs.csv")
        gps_plots.plot_all_stations(
            df,
            df_diff,
            ncols=int(np.sqrt(len(df_diff))),
            ylim=(-2, 2),
            figsize=self.figsize,
            outname=self.figure_directory / "insar_gps_plots.pdf",
        )
        self.insar_gps_df = df
        self.insar_gps_df_diff = df_diff

    def reference_to_gps(self):
        if not self.ref_station:
            logger.info("No reference station specified. Skipping shift")
            # self.shifted_dset_name = self.dset_name
            correction = 0
        else:
            ref_cols = [f"{self.ref_station}_gps", f"{self.ref_station}_insar"]
            lon, lat = gps.station_lonlat(self.ref_station)

            # Get the insar in the reference spot
            ref_win_ts = utils.window_stack_xr(
                self.ds.defo_noisy, lon=lon, lat=lat, window_size=self.gps_window_size
            )
            ref_win_ts_sm = ref_win_ts.rolling(
                {"date": 50}, min_periods=1, center=True
            ).mean()

            # Get the gps in the reference spot
            ref_gps = self.insar_gps_df[ref_cols].dropna(subset=[ref_cols[1]])[
                ref_cols[0]
            ]
            ref_gps_sm = ref_gps.rolling(50, min_periods=1, center=True).mean()
            correction = (ref_gps_sm - ref_win_ts_sm).to_xarray()

        if np.any(np.isnan(correction.values)):
            logger.error("NaN in correction")
            logger.error("Skipping the correction")
            correction = 0
        self.ds[self.shifted_dset_name] = self.ds[self.dset_name] + correction

    def set_mask(self):
        """Get the missing data, correlation mask and apply to the shifted dataset"""
        mask_missing = self.get_mask()
        # mask_missing = np.logical_or(self.ds[self.dset_name].values == 0, mask_missing)

        cor_mask = self.get_cor_mask()
        mask = np.logical_or(mask_missing, cor_mask.data)

        self.ds[self.shifted_dset_name].values[:, mask] = 0

    def save_output(self):
        from apertools import sario

        ds_interp = utils.interpolate_xr(
            self.ds, dset_name=self.shifted_dset_name, freq=self.freq, col="date"
        )

        for layer in ds_interp[self.shifted_dset_name][1:]:
            dt_str = pd.to_datetime(layer.date.item()).strftime("%Y%m%d")
            outfile = str(self.outfile_template).format(
                path_num=self.path_num, dt=dt_str
            )
            logger.info(f"Saving {outfile}")
            sario.save_xr_tif(layer, crs=self.crs, outname=outfile)
            # Set the units to cm
            sario.set_unit(outfile, unit="cm")

        # Copy the LOS file for the east/up decomposition
        sario.save_xr_tif(
            self.ds[self.los_dset],
            crs=self.crs,
            outfile=self.out_directory / self.los_map_filename,
        )
        # And save the mean correlation for reference
        sario.save_xr_tif(
            self.ds[self.cor_dset],
            crs=self.crs,
            outfile=self.out_directory / self.cor_mean_filename,
        )


@dataclass
class Decomp:
    asc_directory: str
    desc_directory: str
    asc_path_num: int
    desc_path_num: int
    out_directory: str
    infile_glob: str = "cumulative_los_*.tif"
    outfile_template: str = (
        "cumulative_east_up_paths_{asc_path_num}_{desc_path_num}_{dt}.tif"
    )
    los_map_filename: str = "los_enu.tif"

    def run(self):
        # TODO: factor out the repeated directory stuff here
        if self.out_directory is None:
            p = Path(self.directory) / f"los_out_{self.path_num}/"
            self.out_directory = p.resolve()
        else:
            self.out_directory = Path(self.out_directory).resolve()
        utils.mkdir_p(self.out_directory)
        self._set_full_paths()

        asc_filenames = sorted(glob.glob(str(self.asc_directory / self.infile_glob)))
        desc_filenames = sorted(glob.glob(str(self.desc_directory / self.infile_glob)))

        outfiles = []
        for af, df in zip(asc_filenames, desc_filenames):
            date_a = _get_date(af)
            date_d = _get_date(df)
            assert date_a == date_d
            logger.info("Solving {} and {}".format(af, df))

            outfile = str(self.outfile_template).format(
                asc_path_num=self.asc_path_num,
                desc_path_num=self.desc_path_num,
                dt=date_a,
            )
            outfile = Path(self.out_directory) / outfile
            logger.info("Saving to {}".format(outfile))

            east, up = los.solve_east_up(
                asc_img_fname=af,
                desc_img_fname=df,
                asc_enu_fname=self.asc_los_map_filename,
                desc_enu_fname=self.desc_los_map_filename,
                outfile=outfile,
            )
            outfiles.append(outfile)

        record(self, self.out_directory / "run_params.yaml")
        return outfiles

    def _set_full_paths(self):
        self.asc_directory = Path(self.asc_directory).resolve()
        self.desc_directory = Path(self.desc_directory).resolve()
        self.asc_los_map_filename = self.asc_directory / self.los_map_filename
        self.desc_los_map_filename = self.desc_directory / self.los_map_filename
        self.outfile_template = self.out_directory / self.outfile_template

    def _set_abs_path(self, filename):
        """Allows a fully qualified path through; otherwise, appends to the directory"""
        if os.path.abspath(filename) != filename:
            return (Path(self.directory) / filename).resolve()
        return filename


@dataclass
class Merger:
    in_dir1: str
    in_dir2: str
    infile_glob: str = "cumulative_east_up_paths_*.tif"
    east_template: str = "merged_east_{date}.tif"
    up_template: str = "merged_vertical_{date}.tif"
    out_directory: str = "merged_east_up"

    def run(self):
        utils.mkdir_p(self.out_directory)
        outfiles1 = glob.glob(str(Path(self.in_dir1) / self.infile_glob))
        outfiles2 = glob.glob(str(Path(self.in_dir2) / self.infile_glob))

        merged_imgs = []
        merged_outfiles = []

        out_templates = [self.east_template, self.up_template]
        bands = [1, 2]
        for f1, f2 in zip(outfiles1, outfiles2):
            for t, band in zip(out_templates, bands):
                cur_date = re.search(r"\d{8}", f1).group()
                assert cur_date == re.search(r"\d{8}", f2).group()
                outfile = Path(self.out_directory) / t.format(date=cur_date)
                print(f"creating {outfile}")
                m = subset.create_merged_files(
                    f1, f2, band1=band, band2=band, outfile=outfile
                )
                merged_imgs.append(m)
                merged_outfiles.append(outfile)

        record(self, Path(self.out_directory) / "run_params.yaml")
        # return merged_imgs, merged_outfiles


@dataclass
class Pipeline:
    """Wrapper to call either `los`, `decomp`, or `merge` commands"""

    def __init__(self):
        self.los = LOS
        self.decomp = Decomp
        self.merge = Merger

    # def run(self, *args, **kwargs):
    #     self.los.run()
    #     self.merger.run()
    #     return "Pipeline complete"


def _get_date(filename):
    m = re.search(r"\d{8}", filename)
    try:
        return m.group()
    except AttributeError:
        raise ValueError(f"Could not find date in filename {filename}")


def record(obj, filename):
    self_dict = asdict(obj)
    for k, v in self_dict.items():
        if isinstance(v, Path):
            self_dict[k] = str(v)
    utils.record_params_as_yaml(filename, **self_dict)


@log_runtime
def main():
    # fire.Fire(LOS)
    fire.Fire(Pipeline)


if __name__ == "__main__":
    main()
