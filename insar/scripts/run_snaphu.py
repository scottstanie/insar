#!/usr/bin/env python
import os
import shutil
from glob import glob
import argparse
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# TODO: pass in the path: currently only runs on the current directory
# Wrapper to run in parallel:
PHASE_UNWRAP_DIR = os.path.expanduser("~/phase_unwrap/bin")


from apertools.log import get_log, log_runtime
logger = get_log()

def unwrap(
    intfile,
    outfile,
    width,
    ext_int=".int",
    ext_cor=".cor",
    float_cor=False,
    skip_cor=False,
    do_tile=True,
):
    corname = intfile.replace(ext_int, ext_cor) if ext_cor else None
    if skip_cor:
        corname = None
    conncomp_name = intfile.replace(ext_int, ".conncomp")
    cmd = _snaphu_cmd(
        intfile, width, corname, outfile, conncomp_name, float_cor=float_cor, do_tile=do_tile
    )
    print(cmd)
    subprocess.check_call(cmd, shell=True)
    if os.path.exists(intfile + ".rsc"):
        shutil.copy(intfile + ".rsc", outfile + ".rsc")
    set_unw_zeros(outfile, intfile)


def _snaphu_cmd(intfile, width, corname, outname, conncomp_name, float_cor=False, do_tile=True):
    conf_name = outname + ".snaphu_conf"
    # Need to specify the conncomp file format in a config file
    conf_string = f"""STATCOSTMODE SMOOTH
INFILE {intfile}
LINELENGTH {width}
OUTFILE {outname}
CONNCOMPFILE {conncomp_name} # TODO: snaphu has a bug for tiling conncomps
"""
    if float_cor:
        # Need to specify the input file format in a config file
        # the rest of the options are overwritten by command line options
        # conf_string += "INFILEFORMAT     COMPLEX_DATA\n"
        # conf_string += "CORRFILEFORMAT   ALT_LINE_DATA"
        conf_string += "CORRFILEFORMAT   FLOAT_DATA\n"
    if corname:
        # cmd += f" -c {corname}"
        conf_string += f"CORRFILE	{corname}\n"
    
    # Calculate the tiles sizes/number of procs to use, separate for width/height
    nprocs = 1
    if do_tile:
        if width > 1000:
            conf_string += "NTILECOL 3\nCOLOVRLP 400\n"
            nprocs *= 3
            # cmd += " -S --tile 3 3 400 400 --nproc 9"
        elif width > 500:
            conf_string += "NTILECOL 2\nCOLOVRLP 400\n"
            nprocs *= 2
            # cmd += " -S --tile 2 2 400 400 --nproc 4"

        height = os.path.getsize(intfile) / width / 8
        if height > 1000:
            conf_string += "NTILEROW 3\nROWOVRLP 400\n"
            nprocs *= 3
        elif height > 500:
            conf_string += "NTILEROW 2\nROWOVRLP 400\n"
            nprocs *= 2
    if nprocs > 1:
        conf_string += f"NPROC {nprocs}\n"

    with open(conf_name, "w") as f:
        f.write(conf_string)

    cmd = f"{PHASE_UNWRAP_DIR}/snaphu -f {conf_name} "
    return cmd


def set_unw_zeros(unw_filename, ifg_filename):
    """Set areas that are 0 in the ifg to be 0 in the unw"""
    tmp_file = unw_filename.replace(".unw", "_tmp.unw")
    cmd = (
        f"gdal_calc.py --quiet --outfile={tmp_file} --type=Float32 --format=ROI_PAC "
        f'--allBands=A -A {unw_filename} -B {ifg_filename} --calc "A * (B!=0)"'
    )
    print(f"Setting zeros for {unw_filename}")
    print(cmd)
    subprocess.check_call(cmd, shell=True)
    subprocess.check_call(f"mv {tmp_file} {unw_filename}", shell=True)
    subprocess.check_call(f"rm -f {tmp_file}.rsc", shell=True)


@log_runtime
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", "-p", default=".", help="Path to directory of .unw files"
    )
    parser.add_argument(
        "--ext",
        help="extension of interferograms to search in `--path` (default=%(default)s)",
        default=".int",
    )
    parser.add_argument(
        "--file-list",
        "-f",
        help="Alternative to --path/--ext. Give filename containing names of files to unwrap",
    )
    parser.add_argument(
        "--cols", "-c", type=int, help="Optional: Specify number of cols in the file"
    )
    parser.add_argument("--ext-unw", default=".unw")
    parser.add_argument(
        "--ext-cor",
        default=".cor",
        help="extension for correlation files. (default=%(default)s)",
    )
    parser.add_argument("--float-cor", action="store_true")
    parser.add_argument(
        "--skip-cor", action="store_true", help="Skip the use of correlation files."
    )
    parser.add_argument(
        "--no-tile", help="Don't use tiling mode no matter how big the image is.", action="store_true"
    )
    parser.add_argument("--create-isce-headers", action="store_true")
    parser.add_argument(
        "--overwrite", help="Overwrite existing unwrapped file", action="store_true"
    )
    parser.add_argument("--max-jobs", type=int, default=20)

    args = parser.parse_args()
    if args.file_list:
        with open(args.file_list) as f:
            filenames = [line for line in f.read().splitlines() if line]
    else:
        filenames = glob(os.path.join(args.path, "*" + args.ext))

    if len(filenames) == 0:
        print("No files found. Exiting.")
        return

    if not args.cols:
        import rasterio as rio

        with rio.open(filenames[0]) as src:
            width = src.shape[-1]
    else:
        width = args.cols

    if not args.ext:
        ext = os.path.splitext(filenames[0])[1]
    else:
        ext = args.ext
        assert args.ext == os.path.splitext(filenames[0])[1]

    all_out_files = [inf.replace(ext, args.ext_unw) for inf in filenames]
    in_files, out_files = [], []
    for inf, outf in zip(filenames, all_out_files):
        if os.path.exists(outf) and not args.overwrite:
            # print(outf, "exists, skipping.")
            continue
        in_files.append(inf)
        out_files.append(outf)
    print(f"{len(out_files)} left to unwrap")

    with ThreadPoolExecutor(max_workers=args.max_jobs) as exc:
        futures = [
            exc.submit(
                unwrap,
                inf,
                outf,
                width,
                ext,
                args.ext_cor,
                args.float_cor,
                args.skip_cor,
                not args.no_tile,
            )
            for inf, outf in zip(in_files, out_files)
        ]
        for idx, fut in enumerate(tqdm(as_completed(futures)), start=1):
            fut.result()
            tqdm.write("Done with {} / {}".format(idx, len(futures)))

    if not args.create_isce_headers:
        return

    from apertools import isce_helpers, utils

    for f in tqdm(filenames):
        f = f.replace(args.ext, args.ext_unw)

        dirname, fname = os.path.split(f)
        with utils.chdir_then_revert(dirname):
            isce_helpers.create_unw_image(fname)
            # isce_helpers.create_int_image(fname)


if __name__ == "__main__":
    main()
