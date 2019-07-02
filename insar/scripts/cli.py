"""
Main command line entry point to manage all other sub commands
"""
import os
import json
import click
import insar
import apertools
import sardem
import numpy as np
from .plot_insar import plot_image


# Main entry point:
@click.group()
@click.option('--verbose', is_flag=True)
@click.option('--path',
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
@click.option("--start",
              type=click.IntRange(min=1, max=len(insar.scripts.process.STEPS)),
              help="Choose which step to start on, then run all after. Steps: {}".format(
                  insar.scripts.process.STEP_LIST),
              default=1)
@click.option("--step",
              callback=parse_steps,
              help="Run a one or a range of steps and exit. "
              "Examples:\n--step 4,5,7\n--step 3-6\n--step 1,9-10",
              required=False)
@click.argument("left_lon", type=float, required=False)
@click.argument("top_lat", type=float, required=False)
@click.argument("dlon", type=float, required=False)
@click.argument("dlat", type=float, required=False)
@click.option('--geojson', '-g', help="Filename containing the geojson object for DEM bounds")
@click.option("--sentinel-path",
              envvar="SENTINEL_PATH",
              default="~/sentinel/",
              help="(default=~/sentinel/) Directory containing sentinel scripts.")
@click.option("--rate",
              "-r",
              default=1,
              help="Rate at which to upsample DEM (default=1, no upsampling)")
@click.option("--unzip/--no-unzip",
              help="Pass to sentinel_stack whether to unzip Sentinel files",
              default=True)
@click.option("--cleanup/--no-cleanup",
              help="Rename .geos and cleanup directory to `extra_files` after .geo processing",
              default=True)
@click.option("--max-temporal",
              type=int,
              default=500,
              help="Maximum temporal baseline for igrams (fed to sbas_list)")
@click.option("--max-spatial",
              type=int,
              default=500,
              help="Maximum spatial baseline for igrams (fed to sbas_list)")
@click.option("--looks",
              type=int,
              help="Number of looks to perform on .geo files to shrink down .int, "
              "Default is the upsampling rate, makes the igram size=original DEM size")
@click.option("--lowpass",
              type=int,
              default=1,
              help="Size of lowpass filter to use on igrams before unwrapping")
@click.option("--max-height",
              default=10,
              help="Maximum height/max absolute phase for converting .unw files to .tif"
              " (used for contour_interval option to dishgt)")
@click.option('--window', default=3, help="Window size for .unw stack reference")
@click.option('--constant-vel',
              is_flag=True,
              help="Use a constant velocity for SBAS inversion solution")
@click.option('--alpha', default=0.0, help="Regularization parameter for SBAS inversion")
@click.option('--difference', is_flag=True, help="Use velocity differences for regularization")
@click.option(
    '--deramp-order',
    default=1,
    help="Order of 2D polynomial to use to remove residual phase from unwrapped interferograms"
    " (default is 1, linear ramp)")
@click.option("--ref-row",
              type=int,
              help="Row number of pixel to use as unwrapping reference for SBAS inversion")
@click.option("--ref-col",
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
@click.option("--pause",
              '-p',
              default=200,
              help="For --animate, time in milliseconds to pause"
              " between stack layers (default 200).")
@click.option("--save",
              '-s',
              help="If you want to save the animation as a movie,"
              " title to save file as.")
@click.option("--display/--no-display",
              help="Pop up matplotlib figure to view (instead of just saving)",
              default=True)
@click.option("--cmap", default='seismic', help="Colormap for image display.")
@click.option("--shifted/--no-shifted", default=True, help="Shift colormap to be 0 centered.")
@click.option("--file-ext", help="If not loading deformation.npy, the extension of files to load")
@click.option("--intlist/--no-intlist",
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
            intlist = insar.timeseries.read_intlist(context['path'])
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
@click.option("--filename", default='deformation.npy', help="Name of saved deformation stack")
@click.option("--cmap", default='seismic', help="Colormap for image display.")
@click.option("--label", default='Centimeters', help="Label on colorbar/yaxis for plot")
@click.option("--title", help="Title for image plot")
@click.option('--row-start', default=0)
@click.option('--row-end', default=-1)
@click.option('--col-start', default=0)
@click.option('--col-end', default=-1)
@click.option("--rowcol",
              help="Use row,col for legened entries (instead of default lat,lon)",
              is_flag=True,
              default=False)
@click.pass_obj
def view_stack(context, filename, cmap, label, title, row_start, row_end, col_start, col_end,
               rowcol):
    """Explore timeseries on deformation image.

    If deformation.npy and geolist.npy or .unw files are not in current directory,
    use the --path option:

        insar --path /path/to/igrams view_stack

    """
    geolist, deformation = apertools.sario.load_deformation(context['path'], filename=filename)
    if geolist is None or deformation is None:
        return

    if rowcol:
        rsc_data = None
    else:
        rsc_data = sardem.loading.load_dem_rsc(os.path.join(context['path'], 'dem.rsc'))

    stack = deformation[:, row_start:row_end, col_start:col_end]
    _, nrows, ncols = stack.shape
    new_rsc_data = apertools.latlon.LatlonImage.crop_rsc_data(
        rsc_data,
        row_start,
        col_start,
        nrows,
        ncols,
    )
    img = apertools.latlon.LatlonImage(data=np.mean(stack[-3:], axis=0), dem_rsc=new_rsc_data)
    img = img[row_start:row_end, col_start:col_end]

    apertools.plotting.view_stack(
        stack,
        img,
        geolist=geolist,
        title=title,
        label=label,
        cmap=cmap,
        lat_lon=not rowcol,
    )


# COMMAND: plot
@cli.command('plot')
@click.argument("filename")
@click.option("--downsample", default=1, help="Amount to downsample image")
@click.option("--cmap", default='dismph', help="Colormap for image display.")
@click.option("--title", help="Title for image plot")
@click.option("--alpha",
              default=0.6,
              help="Transparency for background magnitude (if plotting insar)")
@click.option("--colorbar/--no-colorbar", default=True, help="Display colorbar on figure")
def plot(filename, downsample, cmap, title, alpha, colorbar):
    """Quick plot of a single InSAR file.

    filename: Name of InSAR file to plot (possible extensions: .int, .cor, .unw, .geo,...)"

    Can downsample for easier viewing.
    Requires a .dem.rsc file to also be in same directory

    If not using the current directory, use the --path option:

        insar --path /path/to/igrams <filename>

    """
    img = apertools.sario.load(filename, downsample=downsample)
    plot_image(img, title=title, colorbar=colorbar, alpha=alpha)


# COMMAND: view-masks
@cli.command('view-masks')
@click.option("--downsample", default=1, help="Amount to downsample image")
@click.option("--geolist-file",
              default="geolist_missing.txt",
              help="File to save date of missing .geos on click")
@click.pass_obj
def view_masks(context, downsample, geolist_file):
    geo_file_names = insar.timeseries.read_geolist(filepath=context['path'], parse=False)

    def _print(series, row, col):
        print(".geos missing at (%s, %s): %s" % (row, col, np.array(geo_file_names)[series]))

    def _save_missing_geos(series, row, col):
        with open(geolist_file, "w") as f:
            for fname in np.array(geo_file_names)[series]:
                print("Writing %s" % fname)
                f.write("%s\n" % fname)

    geo_mask_file_names = [n + '.mask.npy' for n in geo_file_names]
    geo_masks = np.ma.array(
        apertools.sario.load_stack(file_list=geo_mask_file_names,
                                   downsample=downsample)).astype(bool)

    composite_mask = np.sum(geo_masks, axis=0)

    geolist = insar.timeseries.read_geolist(filepath=context['path'])
    apertools.plotting.view_stack(
        geo_masks,
        display_img=composite_mask,
        geolist=geolist,
        cmap="Reds",
        label="is masked",
        title="Number of dates of missing .geo data",
        line_plot_kwargs=dict(marker="x", linestyle=' '),
        perform_shift=True,
        legend_loc=0,
        # timeline_callback=_print,
        timeline_callback=_save_missing_geos,
    )


# COMMAND: blob
@cli.command(context_settings=dict(ignore_unknown_options=True))
@click.option('--load/--no-load', default=True, help='Load last calculated blobs')
@click.option('--filename',
              type=str,
              help="specific file to search blobs (default is deformation.npy)")
@click.option('--positive/--no-positive',
              default=True,
              help="Search for positive (uplift) blobs "
              "(default True)")
@click.option('--negative/--no-negative',
              default=True,
              help="Search for negative (subsidence) blobs "
              "(default True)")
@click.option('--title-prefix', default='')
@click.option('--blob-filename', default='blobs.npy', help='File to save found blobs')
@click.option('--row-start', default=0)
@click.option('--row-end', default=-1)
@click.option('--col-start', default=0)
@click.option('--col-end', default=-1)
@click.argument('blobfunc_args', nargs=-1, type=click.UNPROCESSED)
@click.pass_obj
def blob(context, load, filename, positive, negative, title_prefix, blob_filename, row_start,
         row_end, col_start, col_end, blobfunc_args, **kwargs):
    """Find and view blobs in deformation

    If deformation.npy and geolist.npy or .unw files are not in current directory,
    use the --path option:

        insar --path /path/to/igrams view_stack
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


# COMMAND: kml
@cli.command()
@click.argument("imgfile", required=False)
@click.option("--shape",
              default="box",
              help="kml shape: use 'box' for image overlay, 'polygon' for geojson square")
@click.option("--rsc", help=".rsc file containing lat/lon start and steps")
@click.option("--geojson", "-g", help="Optional: if making shape from .geojson, file to specify")
@click.option("--title", "-t", help="Title of the KML object once loaded.")
@click.option("--desc", "-d", help="Description for google Earth.")
@click.option("--output", "-o", help="File to save kml output to")
@click.option("--cmap", default="seismic", help="Colormap (if saving .npy image)")
@click.option("--normalize", is_flag=True, default=False, help="Center image to [-1, 1]")
@click.pass_obj
def kml(context, imgfile, shape, rsc, geojson, title, desc, output, cmap, normalize):
    """Creates .kml file for some image
    IMGFILE is the image to load into Google Earth

    Example:

        insar kml 20180420_20180502.tif --rsc dem.rsc -t "My igram" -d "Kiluea eruption" -o out.kml

    """

    def _save_npy_file(imgfile, new_filename):
        image = apertools.sario.load(imgfile)
        if image.ndim > 2:
            # For 3D stack, assume we just want the final image
            image = image[-1]
        apertools.sario.save(new_filename, image, cmap=cmap, normalize=normalize, preview=True)

    if geojson:
        with open(geojson) as f:
            gj_dict = json.load(f)
    else:
        gj_dict = None

    rsc_data = apertools.sario.load(rsc) if rsc else None
    # Check if imgfile is a .npy saved matrix
    if imgfile.endswith(".npy"):
        new_filename = imgfile.replace(".npy", ".png")
        _save_npy_file(imgfile, new_filename)
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


# COMMAND: mask
@cli.command()
@click.option("--imagefile", "-i", help="file containing image to mask")
@click.option("--dem", default="elevation.dem", help=".rsc file containing lat/lon start and steps")
@click.option("--output", "-o", required=True, help="File to save output to")
@click.pass_obj
def mask(context, imagefile, dem, output):
    """Mask an image where some elevation.dem is zero
    """
    image = apertools.sario.load(imagefile)
    out_image = insar.mask.mask_int(image, dem_file=dem, dem=None)
    apertools.sario.save(output, out_image)


# COMMAND: avg-stack
@cli.command('avg-stack')
def avg_stack(context, ref_row, ref_col):
    """Perform simple igram stack average to get a linear trend

    If .unw files are not in the current directory, use the --path option:

        insar --path /path/to/igrams view_stack

    If --ref-row and --ref-col not provided, most coherent patch found as reference
    """
    if not ref_row or ref_col:
        click.echo("Finding most coherent patch in stack.")
        cc_stack = apertools.sario.load_stack(directory=context['path'], file_ext=".cc")
        ref_row, ref_col = insar.timeseries.find_coherent_patch(cc_stack)
        click.echo("Using %s as .unw reference point", (ref_row, ref_col))
    insar.timeseries.avg_stack(context['path'], ref_row, ref_col)


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
        rsc_file = apertools.sario.find_rsc_file(basepath=context['path'])
    uprate = sardem.utils.calc_upsample_rate(rsc_filename=rsc_file)

    click.echo("%s has %.2f times the default spacing" % (rsc_file, uprate))

    default_spacing = 30.0
    click.echo("This is equal to %.2f meter spacing between pixels" % (default_spacing / uprate))


# ###################################
# Preprocessing subgroup of commands:
# ###################################
@cli.group()
@click.pass_obj
def preproc(ctx):
    """Extra commands for preprocessing steps"""


@preproc.command()
@click.option('--delete-zips', is_flag=True, default=False, help="Remove .zip file after unzipping")
@click.pass_obj
def unzip(context, delete_zips):
    insar.scripts.preproc.unzip_sentinel_files(context['path'], delete_zips=delete_zips)


@preproc.command('tiles')
@click.argument('data-path')
@click.option('--path-num',
              type=int,
              help="Relative orbit/path to use (None uses all within data-path)")
@click.option('--tile-size', default=0.5, help="degrees of tile size to aim for")
@click.option('--overlap', default=0.1, help="Overlap of adjacent tiles (in deg)")
@click.pass_obj
def tiles(context, data_path, path_num, tile_size, overlap):
    """Use make_tiles to create a directory structure

    Uses the current directory to make new folders.

    data_path is where the unzipped .SAFE folders are located.

    Populates the current directory with dirs and .geojson files (e.g.):
    N28.8W101.6
    N28.8W101.6.geojson
    N28.8W102.0
    N28.8W102.0.geojson
    ...
    """
    insar.scripts.preproc.create_tile_directories(data_path,
                                                  path_num=path_num,
                                                  tile_size=tile_size,
                                                  overlap=overlap,
                                                  verbose=context['verbose'])


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
