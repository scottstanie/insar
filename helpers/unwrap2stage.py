#!/usr/bin/env python
import argparse
from glob import glob
import os
import subprocess
import h5py
import numpy as np

from tqdm import trange, tqdm

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
    parser.add_argument("--ref-lat", help="stable reference latitude", type=float)
    parser.add_argument("--ref-lon", help="stable reference longitude", type=float)
    parser.add_argument(
        "--ref-station",
        help="Name of GPS station to use as stable reference location",
    )
    parser.add_argument(
        "--no-mask", action="store_false", help="Skip masking from water/radar shadow"
    )
    args = parser.parse_args()

    geom_dir = f"geom_reference_{row_looks}_{col_looks}"

    os.chdir(args.project_dir)
    do_mask = not args.no_mask
    create_multilooked(
        args.looks, geom_dir, max_temp=args.max_temporal, do_mask=do_mask
    )
    make_stacks(args.ref_lat, args.ref_lon, looks, geom_dir)
    transfer_phasemask(
        looks,
        ref_lat=args.ref_lat,
        ref_lon=args.ref_lon,
        ref_station=args.ref_station,
        geom_dir=geom_dir,
    )


def create_multilooked(
    looks,
    geom_dir,
    max_temp=None,
    do_mask=True,
):
    """Create overlooked versions of the Igrams by redo the ISCE config files"""
    from apertools import isce_helpers
    import logging

    logging.getLogger().setLevel(logging.INFO)

    row_looks, col_looks = looks
    config_dir = f"configs_{row_looks}_{col_looks}"
    run_file, config_dir = isce_helpers.multilook_configs(
        looks=(row_looks, col_looks),
        crossmul_only=True,
        max_temp=max_temp,
    )
    # subprocess.run(f"bash {run_file}", shell=True)
    isce_helpers.multilook_geom(looks=looks, overwrite=False)

    ifg_file_list = create_ifg_cor(
        config_dir, geom_dir=geom_dir, looks=looks, do_mask=do_mask
    )
    unwrap_file = "ifgs_to_unwrap.txt"
    with open(unwrap_file, "w") as f:
        for fname in ifg_file_list:
            f.write(f"{fname}\n")
    # Then unwrap
    snaphu_file = "/home/scott/repos/insar/insar/scripts/run_snaphu.py"
    cmd = f"{snaphu_file}  --file-list {unwrap_file} --float-cor --max-procs 20 --create-isce"
    subprocess.check_call(cmd, shell=True)


def create_ifg_cor(
    config_dir,
    looks=None,
    geom_dir=None,
    do_mask=True,
):
    from apertools import isce_helpers, sario

    row_looks, col_looks = looks

    config_dir = config_dir.rstrip("/")
    configs = glob(config_dir + "/*")
    igram_dir = f"Igrams_{row_looks}_{col_looks}"
    if do_mask:
        mask = sario.get_combined_mask(geom_dir)
    else:
        mask = None

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
        isce_helpers.create_cor_from_int_amp(fint, mask=mask)

    return ifg_file_list


def make_stacks(ref_lat, ref_lon, looks, geom_dir):
    from insar import prepare

    row_looks, col_looks = looks
    prepare.prepare_isce(
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        geom_dir=geom_dir,
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


def transfer_phasemask(
    looks,
    full_stack="unw_stack.h5",
    looked_stack="unw_stack_2stage.h5",
    ref_lat=None,
    ref_lon=None,
    ref_station=None,
    geom_dir="geom_reference",
    coordinates="rdr",
    mask=None,
    window=(5, 5),
    overwrite=False
):
    from apertools import sario, deramp, latlon
    from insar import prepare
    import shutil

    row_looks, col_looks = looks
    print(f"Copying {full_stack} to {looked_stack}")
    if not os.path.exists(looked_stack):
        shutil.copy(full_stack, looked_stack)
    else:
        if overwrite:
            print(f"{looked_stack} exists, overwriting.")
        else:
            print(f"{looked_stack} exists, exiting.")
            return

    unw_stack_looked = f"unw_stack_{row_looks}_{col_looks}.h5"
    ifglist = sario.load_ifglist_from_h5(looked_stack)
    ifglist_low = sario.load_ifglist_from_h5(unw_stack_looked)
    ifg_full_idxs = [ifglist.index(ii) for ii in ifglist_low]

    ifg_files = []
    for ifg in sario.ifglist_to_filenames(ifglist):
        # ifg_files.append(os.path.join("Igrams", datestr, "filt_" + ifg))
        ifg_files.append(os.path.join("Igrams", ifg.rstrip(".int"), ifg))

    unw_files_low = []
    for ifg in sario.ifglist_to_filenames(ifglist_low, ext=".unw"):
        unw_files_low.append(
            os.path.join(f"Igrams_{row_looks}_{col_looks}/", ifg.rstrip(".unw"), ifg)
        )

    if ref_station is None and ref_lat is None:
        with h5py.File(full_stack) as hf:
            ref_row, ref_col = hf["stack_flat_shifted"].attrs["reference"][()]
        ref_lat, ref_lon = latlon.rowcol_to_latlon_rdr(
            ref_row, ref_col, geom_dir=geom_dir
        )
    else:
        ref_row, ref_col, ref_lat, ref_lon, ref_station = prepare.get_reference(
            None,
            None,
            ref_lat,
            ref_lon,
            ref_station,
            coordinates,
            unw_stack_looked,
            geom_dir,
        )
    if mask is None:
        shape = sario.load(ifg_files[0], use_gdal=True, band=1).shape
        mask = np.zeros(shape, dtype=bool)

    win_rows, win_cols = np.array(window) // 2
    with h5py.File(looked_stack, "a") as hf_out:
        # dset_low = hf_low['stack_flat_shifted']
        dset_out = hf_out["stack_flat_shifted"]
        dset_out.attrs["reference"] = [ref_row, ref_col]
        dset_out.attrs["reference_latlon"] = [ref_lat, ref_lon]
        dset_out.attrs["reference_station"] = ref_station or ""
        print(list(dset_out.attrs.items()))

        for idx in trange(len(unw_files_low)):
            idx_full = ifg_full_idxs[idx]
            # unw_low = dset_low[idx]
            ifg_high = sario.load(ifg_files[idx_full], use_gdal=True, band=1)

            # ifg_high = sario.load(ifg_files[idx], use_gdal=True, band=1)
            unw_low = sario.load(unw_files_low[idx], use_gdal=True, band=2)
            unw_high = prepare.apply_phasemask(unw_low, ifg_high)
            # unw_high_snaphu = dset_out[idx]

            deramped_phase = deramp.remove_ramp(unw_high, deramp_order=1, mask=mask)
            deramped_phase[mask] = np.nan

            # Now center it on the shift window
            deramped_phase = _shift(
                deramped_phase, ref_row, ref_col, win_rows, win_cols
            )
            # overwrite with new
            dset_out[idx] = deramped_phase


def _shift(deramped_phase, ref_row, ref_col, win_rows, win_cols):
    patch = deramped_phase[
        ref_row - win_rows : ref_row + win_rows + 1,
        ref_col - win_cols : ref_col + win_cols + 1,
    ]
    return deramped_phase - np.nanmean(patch)


if __name__ == "__main__":
    main()
