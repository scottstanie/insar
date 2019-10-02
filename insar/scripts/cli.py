"""
Main command line entry point to manage all other sub commands
"""
import os
import json
import h5py
import click
import insar
import apertools
import sardem
import numpy as np
from .plot_insar import plot_image
import insar.prepare


# Main entry point:
@click.group()
@click.option('--verbose', is_flag=True)
@click.option(
    '--path',
    type=click.Path(exists=False, file_okay=False, writable=True),
    default='.',
    help="Path of interest for command. "
    "Will search for files path or change directory, "
    "depending on command.")
@click.pass_context
def cli(ctx, verbose, path):
    """Command line tools for processing insar."""
    # Store these to be passed to all sub commands
    ctx.obj = {}
    ctx.obj['verbose'] = verbose
    ctx.obj['path'] = path


# COMMAND: PROCESS
def parse_steps(ctx, param, value):
    """Allows ranges of steps, from https://stackoverflow.com/a/4726287"""
    if value is None:
        return []

    step_nums = set()
    try:
        for part in value.split(','):
            x = part.split('-')
            step_nums.update(range(int(x[0]), int(x[-1]) + 1))
    except (ValueError, AttributeError):
        raise click.BadParameter("Must be comma separated integers and/or dash separated range.")
    max_step = len(insar.scripts.process.STEPS)
    if any((num < 1 or num > max_step) for num in step_nums):
        raise click.BadParameter("must be ints between 1 and {}".format(max_step))

    return sorted(step_nums)


@cli.command()
@click.option(
    "--start",
    type=click.IntRange(min=1, max=len(insar.scripts.process.STEPS)),
    help="Choose which step to start on, then run all after. Steps: {}".format(
        insar.scripts.process.STEP_LIST),
    default=1)
@click.option(
    "--step",
    callback=parse_steps,
    help="Run a one or a range of steps and exit. "
    "Examples:\n--step 4,5,7\n--step 3-6\n--step 1,9-10",
    required=False)
@click.argument("left_lon", type=float, required=False)
@click.argument("top_lat", type=float, required=False)
@click.argument("dlon", type=float, required=False)
@click.argument("dlat", type=float, required=False)
@click.option('--geojson', '-g', help="Filename containing the geojson object for DEM bounds")
@click.option(
    "--sentinel-path",
    envvar="SENTINEL_PATH",
    default="~/sentinel/",
    help="(default=~/sentinel/) Directory containing sentinel scripts.")
@click.option(
    "--rate", "-r", default=1, help="Rate at which to upsample DEM (default=1, no upsampling)")
@click.option(
    "--unzip/--no-unzip",
    help="Pass to sentinel_stack whether to unzip Sentinel files",
    default=True)
@click.option(
    "--cleanup/--no-cleanup",
    help="Rename .geos and cleanup directory to `extra_files` after .geo processing",
    default=True)
@click.option(
    "--max-temporal",
    type=int,
    default=500,
    help="Maximum temporal baseline for igrams (fed to sbas_list)")
@click.option(
    "--max-spatial",
    type=int,
    default=500,
    help="Maximum spatial baseline for igrams (fed to sbas_list)")
@click.option(
    "--looks",
    type=int,
    help="Number of looks to perform on .geo files to shrink down .int, "
    "Default is the upsampling rate, makes the igram size=original DEM size")
@click.option(
    "--lowpass",
    type=int,
    default=1,
    help="Size of lowpass filter to use on igrams before unwrapping")
@click.option(
    "--max-jobs",
    type=int,
    default=None,
    help="Cap the number of snaphu processes to kick off at once."
    " If none specified, number of cpu cores is used.")
@click.option(
    "--max-height",
    default=10,
    help="Maximum height/max absolute phase for converting .unw files to .tif"
    " (used for contour_interval option to dishgt)")
@click.option('--window', default=3, help="Window size for .unw stack reference")
@click.option(
    '--ignore-geos',
    is_flag=True,
    help="Use the geolist ignore file to ignore dates "
    "(saved to geolist_ignore.txt from `view-masks`")
@click.option(
    '--constant-velocity',
    '-c',
    is_flag=True,
    help="Use a constant velocity for SBAS inversion solution")
@click.option('--stackavg', is_flag=True, help="Use a stack averaging method for timeseries")
@click.option('--alpha', default=0.0, help="Regularization parameter for SBAS inversion")
@click.option('--difference', is_flag=True, help="Use velocity differences for regularization")
@click.option(
    '--deramp-order',
    default=1,
    help="Order of 2D polynomial to use to remove residual phase from unwrapped interferograms"
    " (default is 1, linear ramp)")
@click.option(
    "--ref-row",
    type=int,
    help="Row number of pixel to use as unwrapping reference for SBAS inversion")
@click.option(
    "--ref-col",
    type=int,
    help="Column number of pixel to use as unwrapping reference for SBAS inversion")
@click.pass_obj
def process(context, **kwargs):
    """Process stack of Sentinel interferograms.

    Contains the steps from SLC .geo creation to SBAS deformation inversion

    left_lon, top_lat, dlon, dlat are used to specify the DEM bounding box.
    They may be ignored if not running step 2, and are an alternative to
    using --geojson
    """
    kwargs['verbose'] = context['verbose']
    if kwargs['left_lon'] and kwargs['geojson']:
        raise click.BadOptionUsage("Can't use both positional arguments "
                                   "(left_lon top_lat dlon dlat) and --geojson")
    elif kwargs['geojson']:
        with open(kwargs['geojson']) as f:
            geojson = json.load(f)
        kwargs['geojson'] = geojson

    insar.scripts.process.main(context['path'], kwargs)


# COMMAND: animate
@cli.command()
@click.option(
    "--pause",
    '-p',
    default=200,
    help="For --animate, time in milliseconds to pause"
    " between stack layers (default 200).")
@click.option(
    "--save", '-s', help="If you want to save the animation as a movie,"
    " title to save file as.")
@click.option(
    "--display/--no-display",
    help="Pop up matplotlib figure to view (instead of just saving)",
    default=True)
@click.option("--cmap", default='seismic', help="Colormap for image display.")
@click.option("--shifted/--no-shifted", default=True, help="Shift colormap to be 0 centered.")
@click.option("--file-ext", help="If not loading deformation.npy, the extension of files to load")
@click.option(
    "--intlist/--no-intlist",
    default=False,
    help="If loading other file type, also load `intlist` file  for titles")
@click.option("--db/--no-db", help="Use dB scale for images (default false)", default=False)
@click.option("--vmax", type=float, help="Maximum value for imshow")
@click.option("--vmin", type=float, help="Minimum value for imshow")
@click.pass_obj
def animate(context, pause, save, display, cmap, shifted, file_ext, intlist, db, vmin, vmax):
    """Creates animation for 3D image stack.

    If deformation.npy and geolist.npy or .unw files are not in current directory,
    use the --path option:

        insar --path /path/to/igrams animate

    Note: Default is to load deformation.npy, assuming inversion has
    been solved by `insar process --step 10`
    Otherwise, use --file-ext "unw", for example
    """
    if file_ext:
        stack = apertools.sario.load_stack(directory=context['path'], file_ext=file_ext)
        if intlist:
            intlist = (context['path'])
            titles = [
                "%s - %s" % (d1.strftime("%Y-%m-%d"), d2.strftime("%Y-%m-%d")) for d1, d2 in intlist
            ]
        else:
            titles = sorted(apertools.sario.find_files(context['path'], "*" + file_ext))
    else:
        geolist, deformation = apertools.sario.load_deformation(context['path'])
        stack = deformation
        titles = [d.strftime("%Y-%m-%d") for d in geolist]

    if db:
        stack = apertools.utils.db(stack)

    apertools.plotting.animate_stack(
        stack,
        pause_time=pause,
        display=display,
        titles=titles,
        save_title=save,
        cmap_name=cmap,
        shifted=shifted,
        vmin=vmin,
        vmax=vmax,
    )


# COMMAND: view-stack
@cli.command('view-stack')
@click.argument("filename", default='deformation.h5', required=False)
@click.option("--cmap", default='seismic', help="Colormap for image display.")
@click.option("--label", default='Centimeters', help="Label on colorbar/yaxis for plot")
@click.option("--title", help="Title for image plot")
@click.option('--row-start', default=0)
@click.option('--row-end', type=int, default=None)
@click.option('--col-start', default=0)
@click.option('--col-end', type=int, default=None)
@click.option(
    "--rowcol",
    help="Use row,col for legened entries (instead of default lat,lon)",
    is_flag=True,
    default=False)
@click.option("--mask/--no-mask", help="Mask areas that have any missing data", default=True)
@click.option("--vmin", type=float, help="Optional: Minimum value for imshow")
@click.option("--vmax", type=float, help="Optional: Maximum value for imshow")
@click.pass_obj
def view_stack(context, filename, cmap, label, title, row_start, row_end, col_start, col_end,
               rowcol, mask, vmin, vmax):
    """Explore timeseries on `filename`, a deformation stack
    If deformation.npy and geolist.npy or .unw files are not in current directory,
    use the --path option:

        insar --path /path/to/igrams view_stack

    """
    defo_path = os.path.abspath(os.path.split(filename)[0])
    geo_date_list, deformation = apertools.sario.load_deformation(defo_path, filename=filename)

    if geo_date_list is None or deformation is None:
        return

    if rowcol:
        rsc_data = None
        print("Using rows/cols")
    else:
        rsc_data = apertools.sario.load(os.path.join(defo_path, 'dem.rsc'))
        print("Using lat/lon")

    stack_mask = apertools.sario.load_mask(
        geo_date_list=geo_date_list, perform_mask=mask, directory=defo_path)

    stack_ll = apertools.latlon.LatlonImage(data=deformation, dem_rsc=rsc_data)
    stack_ll[:, stack_mask] = np.nan

    stack_ll = stack_ll[:, row_start:row_end, col_start:col_end]

    img = apertools.latlon.LatlonImage(
        data=np.mean(stack_ll[-3:], axis=0), dem_rsc=stack_ll.dem_rsc)

    apertools.plotting.view_stack(
        stack_ll,
        img,
        geolist=geo_date_list,
        title=title,
        label=label,
        cmap=cmap,
        lat_lon=not rowcol,
        vmin=vmin,
        vmax=vmax,
    )


# COMMAND: plot
@cli.command('plot')
@click.argument("filename")
@click.option("--downsample", "-d", default=1, help="Amount to downsample image")
@click.option("--cmap", default='dismph', help="Colormap for image display.")
@click.option("--title", help="Title for image plot")
@click.option(
    "--alpha", default=0.6, help="Transparency for background magnitude (if plotting insar)")
@click.option("--colorbar/--no-colorbar", default=True, help="Display colorbar on figure")
@click.option("--dset", help="Name of dset if plotting from .h5 file")
def plot(filename, downsample, cmap, title, alpha, colorbar, dset):
    """Quick plot of a single InSAR file.

    filename: Name of InSAR file to plot (possible extensions: .int, .cor, .unw, .geo,...)"

    Can downsample for easier viewing.
    Requires a .dem.rsc file to also be in same directory

    If not using the current directory, use the --path option:

        insar --path /path/to/igrams <filename>

    """
    img = apertools.sario.load(filename, downsample=downsample, dset=dset)
    plot_image(img, title=title, cmap=cmap, colorbar=colorbar, alpha=alpha)


# COMMAND: view-masks
@cli.command('view-masks')
@click.option("--downsample", "-d", default=1, help="Amount to downsample image")
@click.option(
    "--geolist-ignore-file",
    default="geolist_ignore.txt",
    help="File to save date of missing .geos on click")
@click.option(
    "--print-dates/--no-print-dates", default=False, help="Print out missing dates to terminal")
@click.option("--cmap", default="Reds", help="colormap to display mask areas")
@click.option("--vmin", type=float, default=0, help="Optional: Minimum value for imshow")
@click.option("--vmax", type=float, help="Optional: Maximum value for imshow")
@click.pass_obj
def view_masks(context, downsample, geolist_ignore_file, print_dates, cmap, vmin, vmax):
    geo_date_list = apertools.sario.load_geolist_from_h5(apertools.sario.MASK_FILENAME)

    def _print(series, row, col):
        dstrings = [d.strftime("%Y%m%d") for d in np.array(geo_date_list)[series]]
        print(".geos missing at (%s, %s)" % (row, col))
        print("%s" % '\n'.join(dstrings))

    def _save_missing_geos(series, row, col):
        geo_str_list = [
            g.strftime(apertools.sario.DATE_FMT) for g in np.array(geo_date_list)[series]
        ]
        with open(geolist_ignore_file, "w") as f:
            print("Writing %s dates: %s" % (len(geo_str_list), geo_str_list))
            for gdate in geo_str_list:
                f.write("%s\n" % gdate)

    with h5py.File(apertools.sario.MASK_FILENAME) as f:
        geo_dset = f[apertools.sario.GEO_MASK_DSET]
        composite_mask = f[apertools.sario.GEO_MASK_SUM_DSET][:]
        with geo_dset.astype(np.bool):
            geo_masks = geo_dset[:]
        composite_mask = f[apertools.sario.GEO_MASK_SUM_DSET][:]

    if print_dates:
        callback = _print
    elif geolist_ignore_file:
        print("Saving to %s" % geolist_ignore_file)
        callback = _save_missing_geos

    apertools.plotting.view_stack(
        geo_masks,
        display_img=composite_mask,
        geolist=geo_date_list,
        cmap=cmap,
        label="is masked",
        title="Number of dates of missing .geo data",
        line_plot_kwargs=dict(marker="x", linestyle=' '),
        perform_shift=False,
        vmin=vmin,
        vmax=vmax,
        legend_loc=0,
        # timeline_callback=_print,
        timeline_callback=callback,
    )


# COMMAND: blob
@cli.command(context_settings=dict(ignore_unknown_options=True))
@click.option('--load/--no-load', default=True, help='Load last calculated blobs')
@click.option(
    '--filename',
    '-f',
    type=str,
    default="deformation.h5",
    help="specific file to search blobs (default is deformation.h5)")
@click.option(
    '--positive/--no-positive',
    default=True,
    help="Search for positive (uplift) blobs "
    "(default True)")
@click.option(
    '--negative/--no-negative',
    default=True,
    help="Search for negative (subsidence) blobs "
    "(default True)")
@click.option('--title-prefix', default='')
@click.option('--blob-filename', default='blobs.h5', help='File to save found blobs')
@click.option('--row-start', default=0)
@click.option('--row-end', default=-1)
@click.option('--col-start', default=0)
@click.option('--col-end', default=-1)
@click.option(
    '--mask/--no-mask',
    default=True,
    help="Use the stack mask to ignore bad-data areas "
    "(default True)")
@click.argument('blobfunc_args', nargs=-1, type=click.UNPROCESSED)
@click.pass_obj
def blob(context, load, filename, positive, negative, title_prefix, blob_filename, row_start,
         row_end, col_start, col_end, mask, blobfunc_args, **kwargs):
    """Find and view blobs in deformation

    If deformation.npy and geolist.npy or .unw files are not in current directory,
    use the --path option:

        insar --path /path/to/igrams blob
    """
    import insar.blob
    extra_args = _handle_args(blobfunc_args)
    print('Extra args to blobfunc:')
    print(extra_args)
    igram_path = context['path']
    insar.blob.make_blob_image(
        igram_path,
        filename,
        load,
        positive,
        negative,
        title_prefix,
        blob_filename,
        row_start,
        row_end,
        col_start,
        col_end,
        context['verbose'],
        mask,
        extra_args,
    )


def _handle_args(extra_args):
    """Convert command line args into function-usable strings

    '--num-sigma' gets processed into num_sigma
    """
    keys = [arg.lstrip('--').replace('-', '_') for arg in list(extra_args)[::2]]
    vals = []
    for val in list(extra_args)[1::2]:
        # First check for string true/false, then try number
        if val.lower() in ('true', 'false'):
            vals.append(val.lower() == 'true')
        else:
            try:
                vals.append(float(val))
            except ValueError:
                vals.append(val)
    return dict(zip(keys, vals))


def _save_npy_file(imgfile,
                   new_filename,
                   use_mask=True,
                   vmin=None,
                   vmax=None,
                   normalize=False,
                   cmap='seismic',
                   shifted=True,
                   preview=True,
                   **kwargs):
    try:
        geo_date_list, image = apertools.sario.load_deformation(".", filename=imgfile, **kwargs)
    except ValueError:
        image = apertools.sario.load(imgfile, **kwargs)
        geo_date_list, use_mask = None, False

    if image.ndim > 2:
        # For 3D stack, assume we just want the final image
        image = image[-1]

    stack_mask = apertools.sario.load_mask(geo_date_list=geo_date_list, perform_mask=use_mask)
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


# COMMAND: kml
@cli.command()
@click.argument("imgfile", required=False)
@click.option(
    "--shape",
    default="box",
    help="kml shape: use 'box' for image overlay, 'polygon' for geojson square")
@click.option("--rsc", help=".rsc file containing lat/lon start and steps")
@click.option("--geojson", "-g", help="Optional: if making shape from .geojson, file to specify")
@click.option("--title", "-t", help="Title of the KML object once loaded.")
@click.option("--desc", "-d", help="Description for google Earth.")
@click.option("--output", "-o", help="File to save kml output to")
@click.option("--imgout", help="File to save image output (default: replace `imgfile` ext to .png")
@click.option("--cmap", default="seismic", help="Colormap (if saving .npy image)")
@click.option(
    "--normalize", is_flag=True, default=False, help="Center image to [-1, 1] (default false)")
@click.option(
    "--shifted/--no-shifted", default=True, help="Shift cmap to center around 0 (default true)")
@click.option("--vmax", type=float, help="Maximum value for imshow")
@click.option("--vmin", type=float, help="Minimum value for imshow")
@click.option("--ann", help=".ann file containing lat/lon start and steps for UAVSAR")
@click.option("--ext", help="extension for UAVSAR (to be used with --ann)")
@click.option("--dset", help="If loading a .h5 file, which dset to load")
@click.option(
    "--mask/--no-mask", default=False, help="If using .h5 stack, load mask and crop bad-data areas")
@click.option(
    "--preview/--no-preview", default=True, help="If using .h5 stack, view to-be-saved img")
@click.pass_obj
def kml(context, imgfile, shape, rsc, geojson, title, desc, output, imgout, cmap, shifted,
        normalize, vmax, vmin, ann, ext, dset, mask, preview):
    """Creates .kml file for some image
    IMGFILE is the image to load into Google Earth

    Example:

        insar kml 20180420_20180502.tif --rsc dem.rsc -t "My igram" -d "Kiluea eruption" -o out.kml
        insar kml grd_dismph.tif --ann trinity_all.ann --ext .int.grd -t "ground project int"

    """
    if geojson:
        with open(geojson) as f:
            gj_dict = json.load(f)
    else:
        gj_dict = None

    if rsc:
        rsc_data = apertools.sario.load(rsc)
    elif ann:
        rsc_data = apertools.parsers.parse_ann_file(ann, ext=ext, verbose=True)
    else:
        rsc_data = None

    # Check if imgfile is a .npy saved matrix
    file_ext = apertools.utils.get_file_ext(imgfile)
    if file_ext in (".npy", ".h5"):
        new_filename = imgfile.replace(file_ext, ".png") if imgout is None else imgout
        _save_npy_file(
            imgfile,
            new_filename,
            use_mask=mask,
            vmin=vmin,
            vmax=vmax,
            normalize=normalize,
            cmap=cmap,
            shifted=shifted,
            dset=dset,
            preview=preview,
        )
    else:
        new_filename = imgfile

    kml_string = apertools.kml.create_kml(
        rsc_data=rsc_data,
        img_filename=new_filename,
        gj_dict=gj_dict,
        title=title,
        desc=desc,
        kml_out=output,
        shape=shape,
    )
    print(kml_string)


# COMMAND: geotiff
@cli.command()
@click.argument("imgfile", required=False)
@click.option(
    "--shape",
    default="box",
    help="kml shape: use 'box' for image overlay, 'quad' for tiled map-overlay")
@click.option("--rsc", help=".rsc file containing lat/lon start and steps")
@click.option("--kml", help=".kml file for image (not used if .rsc used)")
@click.option(
    "--output", "-o", help="File to save .tif output to (defaults to input, replaced with tif)")
@click.option("--cmap", default="seismic", help="Colormap (if saving .npy/.h5 image)")
@click.option("--normalize", is_flag=True, default=False, help="Center image to [-1, 1]")
@click.option("--vmax", type=float, help="Maximum value for imshow")
@click.option("--vmin", type=float, help="Minimum value for imshow")
@click.option(
    "--mask/--no-mask", default=False, help="If using .h5 stack, load mask and crop bad-data areas")
@click.pass_obj
def geotiff(context, imgfile, shape, rsc, kml, output, cmap, normalize, vmax, vmin, mask):
    if rsc:
        rsc_data = apertools.sario.load(rsc)
    else:
        rsc_data = None

    # Check if imgfile is a .npy saved matrix
    file_ext = apertools.utils.get_file_ext(imgfile)
    if file_ext in (".npy", ".h5"):
        new_filename = imgfile.replace(file_ext, ".png")
        _save_npy_file(
            imgfile,
            new_filename,
            use_mask=mask,
            vmin=vmin,
            vmax=vmax,
            normalize=normalize,
            cmap=cmap,
        )
    else:
        new_filename = imgfile
    if output is None:
        output = imgfile.replace(file_ext, ".tif")
    # create_geotiff(rsc_data, kml_file, img_filename, shape='box', outfile)
    apertools.kml.create_geotiff(
        rsc_data=rsc_data,
        kml_file=kml,
        img_filename=new_filename,
        shape=shape,
        outfile=output,
    )


# COMMAND: mask
@cli.command()
@click.option("--imagefile", "-i", help="file containing image to mask")
@click.option("--dem", default="elevation.dem", help=".rsc file containing lat/lon start and steps")
@click.option("--output", "-o", required=True, help="File to save output to")
@click.pass_obj
def mask(context, imagefile, dem, output):
    """Mask an image where some elevation.dem is zero
    """
    from PIL import Image
    image = apertools.sario.load(imagefile)
    heights = apertools.sario.load(dem)

    zero_height = (heights == 0).astype(float)
    mask = np.array(Image.fromarray(zero_height).resize(image.shape))
    intmask = np.ma.array(image, mask=mask)
    apertools.sario.save(output, intmask)


# COMMAND: validate
@cli.command()
@click.argument("geo_path")
@click.argument("defo_filename")
@click.option("--kind", type=click.Choice(['errorbar', 'slope', 'line']), default="errorbar")
@click.option(
    '--reference-station',
    '-r',
    help="GPS Station to base comparisons off. If None, just plots GPS vs InSAR")
@click.option('--linear', is_flag=True, default=False, help="Erase current files and reprocess")
def validate(geo_path, defo_filename, kind, reference_station, linear):
    apertools.gps.plot_insar_vs_gps(
        geo_path=geo_path,
        defo_filename=defo_filename,
        kind=kind,
        reference_station=reference_station,
        linear=linear,
        block=True)


# COMMAND: dem-rate
@cli.command('dem-rate')
@click.option("--rsc_file", help="name of .rsc file")
@click.pass_obj
def dem_rate(context, rsc_file):
    """Print the upsample rate of a dem

    If file is not in the current directory, use the --path option:

        insar --path /path/to/igrams dem-rate

    """
    # full_file = os.path.join(context['path'], rsc_file)
    if rsc_file is None:
        rsc_file = apertools.sario.find_rsc_file(directory=context['path'])
    uprate = sardem.utils.calc_upsample_rate(rsc_filename=rsc_file)

    click.echo("%s has %.2f times the default spacing" % (rsc_file, uprate))

    default_spacing = 30.0
    click.echo("This is equal to %.2f meter spacing between pixels" % (default_spacing / uprate))


# COMMAND: reference
@cli.command('reference')
@click.argument("filename")
def reference(filename):
    """Print the reference location for a shift unw stack

    filename is name of .h5 unw stack file
    """
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


@preproc.command('stacks')
@click.option('--overwrite', is_flag=True, default=False, help="Erase current files and reprocess")
@click.pass_obj
def prepare_stacks(context, overwrite):
    """Create .h5 files of prepared stacks for timeseries

    This step is run before the final `process` step.
    Makes .h5 files for easy loading to timeseries inversion.
    """
    igram_path = context['path']
    insar.prepare.prepare_stacks(igram_path, overwrite=overwrite)


@preproc.command()
@click.option('--delete-zips', is_flag=True, default=False, help="Remove .zip file after unzipping")
@click.pass_obj
def unzip(context, delete_zips):
    insar.scripts.preproc.unzip_sentinel_files(context['path'], delete_zips=delete_zips)


@preproc.command('intmask')
@click.pass_obj
def intmask(context):
    """Create masks for .int files where invalid

    This step is run in `process.ps_sbas_igrams`, but can be
    run separately to inspect
    """
    igram_path = context['path']
    row_looks, col_looks = apertools.utils.find_looks_taken(igram_path)
    insar.timeseries.create_igram_masks(
        igram_path,
        row_looks=row_looks,
        col_looks=col_looks,
    )
