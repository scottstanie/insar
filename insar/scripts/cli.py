"""
Main command line entry point to manage all other sub commands
"""
from os.path import abspath, join, split
import glob
import json
import click
import insar.scripts.process


# Main entry point:
@click.group()
@click.option("--verbose", is_flag=True)
@click.option(
    "--path",
    type=click.Path(exists=False, file_okay=False, writable=True),
    default=".",
    help="Path of interest for command. "
    "Will search for files path or change directory, "
    "depending on command.",
)
@click.pass_context
def cli(ctx, verbose, path):
    """Command line tools for processing insar."""
    # Store these to be passed to all sub commands
    ctx.obj = {}
    ctx.obj["verbose"] = verbose
    ctx.obj["path"] = path


# COMMAND: PROCESS
def parse_steps(ctx, param, value):
    import insar.scripts.process

    """Allows ranges of steps, from https://stackoverflow.com/a/4726287"""
    if value is None:
        return []

    step_nums = set()
    try:
        for part in value.split(","):
            x = part.split("-")
            step_nums.update(range(int(x[0]), int(x[-1]) + 1))
    except (ValueError, AttributeError):
        raise click.BadParameter(
            "Must be comma separated integers and/or dash separated range."
        )
    max_step = len(insar.scripts.process.STEPS)
    if any((num < 1 or num > max_step) for num in step_nums):
        raise click.BadParameter("must be ints between 1 and {}".format(max_step))

    return sorted(step_nums)


@cli.command()
@click.option(
    "--start",
    type=click.IntRange(min=1, max=len(insar.scripts.process.STEPS)),
    help="Choose which step to start on, then run all after. Steps: {}".format(
        insar.scripts.process.STEP_LIST
    ),
    default=1,
)
@click.option(
    "--step",
    callback=parse_steps,
    help="Run a one or a range of steps and exit. "
    "Examples:\n--step 4,5,7\n--step 3-6\n--step 1,9-10",
    required=False,
)
@click.argument("left_lon", type=float, required=False)
@click.argument("top_lat", type=float, required=False)
@click.argument("dlon", type=float, required=False)
@click.argument("dlat", type=float, required=False)
@click.option(
    "--geojson", "-g", help="Filename containing the geojson object for DEM bounds"
)
@click.option(
    "--sentinel-path",
    envvar="SENTINEL_PATH",
    default="~/sentinel/",
    help="Directory containing sentinel scripts.",
    show_default=True,
)
@click.option(
    "--xrate",
    default=1,
    help="Upsample rate for DEM in X/range (default=1, no upsampling)",
    show_default=True,
)
@click.option(
    "--yrate",
    default=1,
    help="Upsample rate for DEM in Y/az (default=1, no upsampling)",
    show_default=True,
)
@click.option(
    "--unzip/--no-unzip",
    help="Pass to sentinel_stack whether to unzip Sentinel files",
    default=True,
    show_default=True,
)
@click.option(
    "--product-type",
    type=click.Choice(["RAW", "SLC", ""]),
    default=None,
    show_default=True,
    help="Level of Sentinel product (default='' will auto detect from files)",
)
@click.option(
    "--gpu",
    is_flag=True,
    help="Use the GPU version of the raw processor (not available for level 1 SLC)",
    default=False,
    show_default=True,
)
@click.option(
    "--cleanup/--no-cleanup",
    help="Rename .geos and cleanup directory to `extra_files` after .geo processing",
    default=True,
    show_default=True,
)
@click.option(
    "--max-temporal",
    type=int,
    default=500,
    show_default=True,
    help="Maximum temporal baseline for igrams (fed to sbas_list)",
)
@click.option(
    "--max-spatial",
    type=int,
    default=500,
    show_default=True,
    help="Maximum spatial baseline for igrams (fed to sbas_list)",
)
@click.option(
    "--xlooks",
    type=int,
    help="Number of looks in x/range to perform on .geo files to shrink down .int, "
    "Default is the upsampling rate, makes the igram size=original DEM size",
)
@click.option(
    "--ylooks",
    type=int,
    help="Number of looks in y/azimuth to perform on .geo files to shrink down .int, "
    "Default is the upsampling rate, makes the igram size=original DEM size",
)
@click.option(
    "--lowpass",
    type=int,
    default=1,
    show_default=True,
    help="Size of lowpass filter to use on igrams before unwrapping",
)
@click.option(
    "--max-jobs",
    type=int,
    default=None,
    help="Cap the number of snaphu processes to kick off at once."
    " If none specified, number of cpu cores is used.",
)
@click.option(
    "--max-height",
    default=10,
    show_default=True,
    help="Maximum height/max absolute phase for converting .unw files to .tif"
    " (used for contour_interval option to dishgt)",
)
@click.option("--window", default=3, help="Window size for .unw stack reference")
@click.option(
    "--ignore-geos",
    is_flag=True,
    show_default=True,
    help="Use the slclist ignore file to ignore dates "
    "(saved to slclist_ignore.txt from `view-masks`",
)
@click.option(
    "--constant-velocity",
    "-c",
    is_flag=True,
    show_default=True,
    help="Use a constant velocity for SBAS inversion solution",
)
@click.option(
    "--stackavg", is_flag=True, help="Use a stack averaging method for timeseries"
)
@click.option(
    "--alpha", default=0.0, help="Regularization parameter for SBAS inversion"
)
@click.option(
    "--difference", is_flag=True, help="Use velocity differences for regularization"
)
@click.option(
    "--deramp-order",
    default=1,
    help="Order of 2D polynomial to use to remove residual phase from unwrapped interferograms",
    show_default=True,
)
@click.option(
    "--ref-row",
    type=int,
    help="Row number of pixel to use as unwrapping reference for SBAS inversion",
)
@click.option(
    "--ref-col",
    type=int,
    help="Column number of pixel to use as unwrapping reference for SBAS inversion",
)
@click.option("--ref-station", help="Name of GPS station to use as reference")
@click.pass_obj
def process(context, **kwargs):
    """Process stack of Sentinel interferograms.

    Contains the steps from SLC .geo creation to SBAS deformation inversion

    left_lon, top_lat, dlon, dlat are used to specify the DEM bounding box.
    They may be ignored if not running step 2, and are an alternative to
    using --geojson
    """
    import insar.scripts.process

    kwargs["verbose"] = context["verbose"]
    if kwargs["left_lon"] and kwargs["geojson"]:
        raise click.BadOptionUsage(
            "Can't use both positional arguments "
            "(left_lon top_lat dlon dlat) and --geojson"
        )
    elif kwargs["geojson"]:
        with open(kwargs["geojson"]) as f:
            geojson = json.load(f)
        kwargs["geojson"] = geojson

    insar.scripts.process.main(context["path"], kwargs)


# COMMAND: view-masks
@cli.command("view-masks")
@click.option("--mask-file", "-f", default="masks.h5", help="filename of mask stack")
@click.option("--downsample", "-d", default=1, help="Amount to downsample image")
@click.option(
    "--slclist-ignore-file",
    default="slclist_ignore.txt",
    help="File to save date of missing .geos on click",
)
@click.option(
    "--print-dates/--no-print-dates",
    default=False,
    help="Print out missing dates to terminal",
)
@click.option("--cmap", default="Reds", help="colormap to display mask areas")
@click.option(
    "--vmin", type=float, default=0, help="Optional: Minimum value for imshow"
)
@click.option("--vmax", type=float, help="Optional: Maximum value for imshow")
def view_masks(
    mask_file, downsample, slclist_ignore_file, print_dates, cmap, vmin, vmax
):
    import numpy as np
    import apertools.sario
    import apertools.plotting
    import h5py

    try:
        import hdf5plugin
    except ImportError:
        pass

    slc_date_list = apertools.sario.load_slclist_from_h5(mask_file)

    def _print(series, row, col):
        dstrings = [d.strftime("%Y%m%d") for d in np.array(slc_date_list)[series]]
        print("slcs missing at (%s, %s)" % (row, col))
        print("%s" % "\n".join(dstrings))

    def _save_missing_slcs(series, row, col):
        slc_str_list = [
            g.strftime(apertools.sario.DATE_FMT)
            for g in np.array(slc_date_list)[series]
        ]
        with open(slclist_ignore_file, "w") as f:
            print("Writing %s dates: %s" % (len(slc_str_list), slc_str_list))
            for gdate in slc_str_list:
                f.write("%s\n" % gdate)

    with h5py.File(mask_file) as f:
        slc_dset = f[apertools.sario.SLC_MASK_DSET]
        with slc_dset.astype(np.bool):
            slc_masks = slc_dset[:]
        composite_mask = np.sum(slc_masks, axis=0)

    # The netcdf will store it row-reversed, since lats go downward...
    if mask_file.endswith(".nc"):
        slc_masks = slc_masks[:, ::-1, :]
        composite_mask = composite_mask[::-1, :]

    if print_dates:
        callback = _print
    elif slclist_ignore_file:
        print("Saving to %s" % slclist_ignore_file)
        callback = _save_missing_slcs

    apertools.plotting.view_stack(
        slc_masks,
        display_img=composite_mask,
        slclist=slc_date_list,
        cmap=cmap,
        label="is masked",
        title="Number of dates of missing .geo data",
        line_plot_kwargs=dict(marker="x", linestyle=" "),
        perform_shift=False,
        vmin=vmin,
        vmax=vmax,
        legend_loc=0,
        # timeline_callback=_print,
        timeline_callback=callback,
    )


# COMMAND: view-masks
@cli.command("create-east-ups")
@click.argument("config-file")
@click.option("--overwrite", is_flag=True, help="Erase current files and reprocess")
def create_east_ups(config_file, overwrite):
    from . import create_east_up

    r = create_east_up.Runner(config_file, overwrite=overwrite)
    r.run()


# COMMAND: blob
@cli.command(context_settings=dict(ignore_unknown_options=True))
@click.option("--load/--no-load", default=True, help="Load last calculated blobs")
@click.option(
    "--filename",
    "-f",
    type=str,
    default="deformation.h5",
    help="specific file to search blobs (default is deformation.h5)",
)
@click.option("--dset", help="If loading a .h5 file, which dset to load")
@click.option(
    "--positive/--no-positive",
    default=True,
    help="Search for positive (uplift) blobs " "(default True)",
)
@click.option(
    "--negative/--no-negative",
    default=True,
    help="Search for negative (subsidence) blobs " "(default True)",
)
@click.option("--title-prefix", default="")
@click.option("--blob-filename", default="blobs.h5", help="File to save found blobs")
@click.option("--row-start", default=0)
@click.option("--row-end", default=-1)
@click.option("--col-start", default=0)
@click.option("--col-end", default=-1)
@click.option(
    "--mask/--no-mask",
    default=True,
    help="Use the stack mask to ignore bad-data areas " "(default True)",
)
@click.argument("blobfunc_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_obj
def blob(
    context,
    load,
    filename,
    dset,
    positive,
    negative,
    title_prefix,
    blob_filename,
    row_start,
    row_end,
    col_start,
    col_end,
    mask,
    blobfunc_args,
    **kwargs,
):
    """Find and view blobs in deformation

    If deformation.npy and slclist.npy or .unw files are not in current directory,
    use the --path option:

        insar --path /path/to/igrams blob
    """
    import insar.blob

    extra_args = _handle_args(blobfunc_args)
    print("Extra args to blobfunc:")
    print(extra_args)
    igram_path = context["path"]
    insar.blob.make_blob_image(
        igram_path,
        filename,
        dset,
        load,
        positive,
        negative,
        title_prefix,
        blob_filename,
        row_start,
        row_end,
        col_start,
        col_end,
        context["verbose"],
        mask,
        extra_args,
    )


def _handle_args(extra_args):
    """Convert command line args into function-usable strings

    '--num-sigma' gets processed into num_sigma
    """
    keys = [arg.lstrip("--").replace("-", "_") for arg in list(extra_args)[::2]]
    vals = []
    for val in list(extra_args)[1::2]:
        # First check for string true/false, then try number
        if val.lower() in ("true", "false"):
            vals.append(val.lower() == "true")
        else:
            try:
                vals.append(float(val))
            except ValueError:
                vals.append(val)
    return dict(zip(keys, vals))


def _save_npy_file(
    imgfile,
    new_filename,
    use_mask=True,
    vmin=None,
    vmax=None,
    normalize=False,
    cmap="seismic_wide",
    shifted=True,
    preview=True,
    **kwargs,
):
    import apertools.sario
    import numpy as np

    try:
        slc_date_list, image = apertools.sario.load_deformation(
            ".", filename=imgfile, **kwargs
        )
    except ValueError:
        image = apertools.sario.load(imgfile, **kwargs)
        slc_date_list, use_mask = None, False

    if image.ndim > 2:
        # For 3D stack, assume we just want the final image
        image = image[-1]

    stack_mask = apertools.sario.load_mask(
        slc_date_list=slc_date_list, perform_mask=use_mask
    )
    image[stack_mask] = np.nan
    image[image == 0] = np.nan
    if shifted:
        cm = apertools.plotting.make_shifted_cmap(
            image,
            cmap_name=cmap,
            vmax=vmax,
            vmin=vmin,
        )
    else:
        cm = cmap
    apertools.sario.save(
        new_filename,
        image,
        cmap=cm,
        normalize=normalize,
        preview=preview,
        vmax=vmax,
        vmin=vmin,
    )


# COMMAND: validate
@cli.command()
@click.argument("slc_path")
@click.argument("defo_filename")
@click.option(
    "--kind", type=click.Choice(["errorbar", "slope", "line"]), default="errorbar"
)
@click.option(
    "--reference-station",
    "-r",
    help="GPS Station to base comparisons off. If None, just plots GPS vs InSAR",
)
@click.option(
    "--linear", is_flag=True, default=False, help="Erase current files and reprocess"
)
def validate(slc_path, defo_filename, kind, reference_station, linear):
    import apertools.gps

    apertools.gps.plot_insar_vs_gps(
        slc_path=slc_path,
        defo_filename=defo_filename,
        kind=kind,
        reference_station=reference_station,
        linear=linear,
        block=True,
    )


# COMMAND: reference
@cli.command("reference")
@click.argument("filename")
def reference(filename):
    """Print the reference location for a shift unw stack

    filename is name of .h5 unw stack file
    """
    import apertools.sario
    import apertools.latlon

    ref = apertools.sario.load_reference(unw_stack_file=filename)
    click.echo("Reference for %s: %s" % (filename, ref))

    rsc_data = apertools.sario.load_dem_from_h5(filename)
    lat, lon = apertools.latlon.rowcol_to_latlon(*ref, rsc_data=rsc_data)
    click.echo("This is equal to (%s, %s)" % (lat, lon))


# ###################################
# Preprocessing subgroup of commands:
# ###################################
@cli.group()
@click.pass_obj
def preproc(ctx):
    """Extra commands for preprocessing steps"""


@preproc.command("stacks")
@click.option(
    "--overwrite", is_flag=True, default=False, help="Erase current files and reprocess"
)
@click.pass_obj
def prepare_stacks(context, overwrite):
    """Create .h5 files of prepared stacks for timeseries

    This step is run before the final `process` step.
    Makes .h5 files for easy loading to timeseries inversion.
    """
    import insar.prepare

    igram_path = context["path"]
    insar.prepare.prepare_stacks(igram_path, overwrite=overwrite)


@preproc.command()
@click.option(
    "--delete-zips",
    is_flag=True,
    default=False,
    help="Remove .zip file after unzipping",
)
@click.pass_obj
def unzip(context, delete_zips):
    import insar.scripts.preproc

    insar.scripts.preproc.unzip_sentinel_files(context["path"], delete_zips=delete_zips)


@preproc.command("subset")
@click.option(
    "--bbox", nargs=4, type=float, help="Window lat/lon bounds: left bot right top"
)
@click.option("--out-dir", "-o", type=click.Path(exists=True))
@click.option("--in-dir", "-i", type=click.Path(exists=True))
@click.option("--start-date", "-s", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end-date", "-e", type=click.DateTime(formats=["%Y-%m-%d"]))
def subset(bbox, out_dir, in_dir, start_date, end_date):
    """Read window subset from .geos in another directory

    Writes the smaller .geos to `outpath`, along with the
    extra files going with it (elevation.dem, .orbtimings)
    """
    import apertools.sario
    from apertools.utils import force_symlink

    if abspath(out_dir) == abspath(in_dir):
        raise ValueError("--in-dir cannot be same as --out-dir")

    # dems:
    apertools.subset.copy_subset(
        bbox,
        join(in_dir, "elevation.dem"),
        join(out_dir, "elevation.dem"),
        driver="ROI_PAC",
    )
    # Fortran cant read anything but 15-space .rsc file :|
    apertools.sario.save("elevation.dem.rsc", apertools.sario.load("elevation.dem.rsc"))

    # weird params file
    with open(join(out_dir, "params"), "w") as f:
        f.write(f"{join(abspath(out_dir), 'elevation.dem')}\n")
        f.write(f"{join(abspath(out_dir), 'elevation.dem.rsc')}\n")

    # geos and .orbtimings
    for in_fname in glob.glob(join(in_dir, "*.geo.vrt")):
        cur_date = apertools.sario._parse(
            apertools.sario._strip_geoname(split(in_fname)[1])
        )
        if (end_date is not None and cur_date > end_date.date()) or (
            start_date is not None and cur_date < start_date.date()
        ):
            continue
        img = apertools.subset.read_subset(bbox, in_fname, driver="VRT")

        _, nameonly = split(in_fname)
        out_fname = join(out_dir, nameonly).replace(".vrt", "")
        # Can't write vrt?
        # copy_subset(bbox, in_fname, out_fname, driver="VRT")
        click.echo(f"Subsetting {in_fname} to {out_fname}")
        apertools.sario.save(out_fname, img)

        s, d = (
            in_fname.replace(".geo.vrt", ".orbtiming"),
            out_fname.replace(".geo", ".orbtiming"),
        )

        click.echo(f"symlinking {s} to {d}")
        force_symlink(s, d)
        # copyfile(s, d)
