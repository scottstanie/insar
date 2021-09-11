#!/usr/bin/env python
import argparse
from glob import glob
import sys
import os
import subprocess
from apertools import isce_helpers

# import datetime
import h5py
import numpy as np

# import matplotlib.pyplot as plt
# import rasterio as rio
# import rioxarray
# import xarray as xr
from tqdm import trange, tqdm

# from apertools import sario, plotting
# from apertools import latlon, gps, los, utils

looks = (15, 9)
row_looks, col_looks = looks

"""
for f in `ls configs_15_9`; do sed -i 's|stackprocess/SLC|stackprocess/merged/SLC|' configs_15_9/$f ; done
for f in `ls configs_15_9`; do sed -i 's|143A/stackprocess|143A_subset|' configs_15_9/$f ; done

slcxml = glob.glob("merged/**/**/*.slc.xml")
sedcmd = '''sed -i "s|<value>{slcname}|<value>{abspath}|" {xmlf}'''
for f in slcxml:
    xmlf = f
    slcname = os.path.split(f)[1].strip('.xml')
    abspath = os.path.abspath(f).strip('.xml')
    subprocess.run(sedcmd.format(slcname=slcname,abspath=abspath, xmlf=xmlf), shell=True)

"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir",
        "-p",
        default=".",
        help="Path to ISCE stripmap project top level (contains Igrams/)",
    )
    parser.add_argument(
        "--looks",
        nargs=2,
        type=int,
        default=(15, 9),
        metavar=("az_looks", "range_looks"),
        help="Number of looks (row, col)/(az, range) for overlooked ifgs",
    )
    parser.add_argument(
        "--max-temporal", default=1000, type=int, help="max temporal baseline to create"
    )
    parser.add_argument("--ref-lat", help="InSAR stable reference latitude", type=float)
    parser.add_argument("--ref-lon", help="InSAR stable reference longitude", type=float)
    args = parser.parse_args()

    os.chdir(args.project_dir)
    create_multilooked(args.looks, args.max_temporal)
    make_stacks(args.ref_lat, args.ref_lon, looks)


def create_multilooked(looks, max_temp=None):
    """Create overlooked versions of the Igrams by redo the ISCE config files"""
    from apertools import isce_helpers
    import logging

    logging.getLogger().setLevel(logging.INFO)

    row_looks, col_looks = looks
    config_dir = f"configs_{row_looks}_{col_looks}"
    # run_file, config_dir = isce_helpers.multilook_configs(
    #     looks=(row_looks, col_looks),
    #     crossmul_only=True,
    #     max_temp=max_temp,
    # )
    # subprocess.run(f"bash {run_file}", shell=True)
    isce_helpers.multilook_geom(looks=looks)
    ifg_file_list = create_ifg_cor(config_dir, looks)
    unwrap_file = "ifgs_to_unwrap.txt"
    with open(unwrap_file, "w") as f:
        for fname in ifg_file_list:
            f.write(f"{fname}\n")
    # Then unwrap
    snaphu_file = "/home/scott/repos/insar/insar/scripts/run_snaphu.py"
    cmd = f"{snaphu_file}  --file-list {unwrap_file} --float-cor --max-procs 20 --create-isce"
    subprocess.check_call(cmd, shell=True)


def create_ifg_cor(config_dir, looks):
    from apertools import isce_helpers

    config_dir = config_dir.rstrip("/")
    configs = glob(config_dir + "/*")
    igram_dir = f"Igrams_{row_looks}_{col_looks}"

    ifg_file_list = []
    for c in tqdm(configs):
        d12 = c.replace(f"{config_dir}/config_igram_", "")
        fint = os.path.join(igram_dir, d12, d12 + ".int")
        ifg_file_list.append(fint)
        if os.path.exists(fint.replace(".int", ".cor")):
            continue
        # ref, sec = _get_ref_sec(c)
        # _crossmul(ref, sec, fint, looks)
        # Create the int from crossmul first:
        subprocess.run(f"stripmapWrapper.py -c {c}", shell=True)
        # isce_helpers.generateIgram()
        # Then make an unfiltered correlation from the int/amp files
        isce_helpers.create_cor_from_int_amp(fint)

    return ifg_file_list


def make_stacks(ref_lat, ref_lon, looks):
    from insar import prepare

    row_looks, col_looks = looks
    prepare.prepare_isce(
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        geom_dir=f"geom_reference_{row_looks}_{col_looks}",
        # # for filtered ifgs:
        # search_term=f"Igrams_{row_looks}_{col_looks}/**/filt*.unw",
        # cor_search_term=f"Igrams_{row_looks}_{col_looks}/**/filt*.cor",
        # for unfiltered ifgs:
        search_term=f"Igrams_{row_looks}_{col_looks}/**/2*.unw",
        cor_search_term=f"Igrams_{row_looks}_{col_looks}/**/2*.cor",
        row_looks=row_looks,
        col_looks=col_looks,
        unw_filename=f"unw_stack_{row_looks}_{col_looks}.h5",
        cor_filename=f"cor_stack_{row_looks}_{col_looks}.h5",
        mask_filename=f"masks_{row_looks}_{col_looks}.h5",
    )


def transfer_phasemask(looks):
    from apertools import sario, deramp
    from insar import prepare
    import shutil

    row_looks, col_looks = looks
    shutil.copy("unw_stack.h5", "unw_stack_2stage.h5")

    unw_stack_looked = (f"unw_stack_{row_looks}_{col_looks}.h5",)
    ifglist = sario.load_ifglist_from_h5("unw_stack_2stage.h5")
    ifglist_low = sario.load_ifglist_from_h5(unw_stack_looked)
    ifg_full_idxs = [ifglist.index(ii) for ii in ifglist_low]
    ifg_files = sario.ifglist_to_filenames(ifglist)

    ifg_files = []
    for ifg in sario.ifglist_to_filenames(ifglist):
        datestr = ifg.replace(".int", "")
        # ifg_files.append(os.path.join("Igrams", datestr, "filt_" + ifg))
        ifg_files.append(os.path.join("Igrams", datestr, ifg))

    unw_files_low = []
    for ifg in sario.ifglist_to_filenames(ifglist_low):
        datestr = ifg.replace(".int", "")
        f_int = os.path.join(f"Igrams_{row_looks}_{col_looks}/", datestr, ifg)
        unw_files_low.append(f_int.replace(".int", ".unw"))

    with h5py.File("unw_stack.h5") as hf:
        ref_row, ref_col = hf["stack_flat_shifted"].attrs["reference"][()]

    with h5py.File("unw_stack_2stage.h5", "a") as hf_out:
        # dset_low = hf_low['stack_flat_shifted']
        dset_out = hf_out["stack_flat_shifted"]
        win = 5
        for idx in trange(len(unw_files_low)):
            idx_full = ifg_full_idxs[idx]
            # unw_low = dset_low[idx]
            ifg_high = sario.load(ifg_files[idx_full], use_gdal=True, band=1)

            # ifg_high = sario.load(ifg_files[idx], use_gdal=True, band=1)
            unw_low = sario.load(unw_files_low[idx], use_gdal=True, band=2)
            unw_high = prepare.apply_phasemask(unw_low, ifg_high)
            # unw_high_snaphu = dset_out[idx]

            deramped_phase = deramp.remove_ramp(
                unw_high,
                deramp_order=1,
            )

            # Now center it on the shift window
            deramped_phase = _shift(deramped_phase, ref_row, ref_col, win)
            # overwrite with new
            dset_out[idx] = deramped_phase


def _shift(deramped_phase, ref_row, ref_col, win):
    patch = deramped_phase[
        ref_row - win : ref_row + win + 1, ref_col - win : ref_col + win + 1
    ]
    return deramped_phase - np.nanmean(patch)


if __name__ == "__main__":
    main()
