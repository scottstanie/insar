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


def unwrap(
    intfile,
    outfile,
    width,
    ext_int=".int",
    ext_cor=".cor",
    float_cor=False,
    skip_cor=False,
):
    corname = intfile.replace(ext_int, ext_cor) if ext_cor else None
    if skip_cor:
        corname = None
    conncomp_name = intfile.replace(ext_int, ".conncomp")
    cmd = _snaphu_cmd(
        intfile, width, corname, outfile, conncomp_name, float_cor=float_cor
    )
    print(cmd)
    subprocess.check_call(cmd, shell=True)
    if os.path.exists(intfile + ".rsc"):
        shutil.copy(intfile + ".rsc", outfile + ".rsc")


def _snaphu_cmd(intfile, width, corname, outname, conncomp_name, float_cor=False):
    cmd = f"{PHASE_UNWRAP_DIR}/snaphu -s "
    # Need to specify the conncomp file format in a config file
    dummy_conf = f"CONNCOMPFILE    {conncomp_name}\n"
    if float_cor:
        # Need to specify the input file format in a config file
        # the rest of the options are overwritten by command line options
        # dummy_conf += "INFILEFORMAT     COMPLEX_DATA\n"
        # dummy_conf += "CORRFILEFORMAT   ALT_LINE_DATA"
        dummy_conf += "CORRFILEFORMAT   FLOAT_DATA\n"
    conf_name = os.path.join(os.path.split(intfile)[0], "snaphu_tmp.conf")
    with open(conf_name, "w") as f:
        f.write(dummy_conf)
    cmd += f" -f {conf_name}"
    cmd += f" {intfile} {width} -o {outname}"
    if corname:
        cmd += f" -c {corname}"
    if width > 1000:
        cmd += " -S --tile 3 3 400 400 --nproc 9"
    elif width > 500:
        cmd += " -S --tile 2 2 400 400 --nproc 4"

    return cmd


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
            )
            for inf, outf in zip(in_files, out_files)
        ]
        for idx, fut in enumerate(tqdm(as_completed(futures)), start=1):
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
