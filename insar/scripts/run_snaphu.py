#!/usr/bin/env python
from concurrent.futures.thread import ThreadPoolExecutor
import os
from glob import glob
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# TODO: pass in the path: currently only runs on the current directory
# Wrapper to run in parallel:
PHASE_UNWRAP_DIR = os.path.expanduser("~/phase_unwrap/bin")


def unwrap(
    intfile,
    width,
    ext_int=".int",
    ext_cor=".cor",
    ext_unw=".unw",
    float_cor=False,
    overwrite=False,
):
    outname = intfile.replace(ext_int, ext_unw)
    if os.path.exists(outname) and not overwrite:
        print(outname, "exists, skipping.")

    corname = intfile.replace(ext_int, ext_cor)
    conncomp_name = intfile.replace(ext_int, ".conncomp")
    cmd = _snaphu_cmd(intfile, width, corname, outname, conncomp_name, float_cor=float_cor)
    print(cmd)
    subprocess.check_call(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", "-o", default="filename.out", help="Output filename"
    )
    parser.add_argument(
        "--path", "-p", default=".", help="Path to directory of .unw files"
    )
    parser.add_argument(
        "--ext",
        help="extension of interferograms to search in `--path` (default=%(default)s)",
        default=".int",
    )
    parser.add_argument(
        "--file-list", "-f", help="Path to a file containing names of files to unwrap"
    )
    parser.add_argument(
        "--cols", "-c", type=int, help="Optional: Specify number of cols in the file"
    )
    parser.add_argument("--ext-cor", default=".cor")
    parser.add_argument("--ext-unw", default=".unw")
    parser.add_argument("--float-cor", action="store_true")
    parser.add_argument(
        "--overwrite", help="Overwrite existing unwrapped file", action="store_true"
    )
    parser.add_argument("--max-procs", type=int, default=10)

    args = parser.parse_args()
    if args.file_list:
        with open(args.file_list) as f:
            filenames = [line for line in f.read().splitlines() if line]
    else:
        filenames = glob(os.path.join(args.path, args.ext))
    if not args.cols:
        import rasterio as rio

        with rio.open(filenames[1]) as src:
            width = src.shape[-1]

    if not args.ext:
        ext = os.path.splitext(filenames[0])[1]
    else:
        ext = args.ext

    with ThreadPoolExecutor(max_workers=args.max_procs) as exc:
        futures = [
            exc.submit(
                unwrap,
                f,
                width,
                ext,
                args.ext_cor,
                args.ext_unw,
                args.float_cor,
                args.overwrite,
            )
            for f in filenames
        ]
        for idx, fut in enumerate(as_completed(futures)):
            print("Done with", idx, "/", len(futures))


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
    cmd += f" {intfile} {width} -c {corname} -o {outname}"
    if width > 1000:
        cmd += " -S --tile 3 3 400 400 --nproc 9"
    elif width > 500:
        cmd += " -S --tile 2 2 400 400 --nproc 4"

    return cmd


if __name__ == "__main__":
    main()
