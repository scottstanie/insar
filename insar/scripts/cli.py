"""Main entry point to manage all other sub commands
"""
import os
import click
import insar


# Main entry point:
@click.group()
@click.option('--verbose', is_flag=True)
@click.option(
    '--path',
    type=click.Path(exists=False, file_okay=True, writable=True),
    default='.',
    help="Path to switch to and run command in"
)
@click.pass_context
def cli(ctx, verbose, path):
    """Help for the insar group of command"""
    # Store these to be passed to all sub commands
    ctx.obj = {}
    ctx.obj['verbose'] = verbose
    ctx.obj['path'] = path
    if path and path != ".":
        click.echo("Changing directory to {}".format(path))
        os.chdir(path)


# COMMAND: DOWNLOAD
@click.option("--date", "-r", help="Validity date for EOF to download")
@click.option(
    "--mission",
    "-m",
    type=click.Choice(["S1A", "S1B"]),
    help="Sentinel satellite to download (None gets both S1A and S1B)"
)
@click.pass_obj
def download(context, **kwargs):
    """Download EOFs for specific date, or for Sentinel files in --path.
       With arguments, searches current directory for Sentinel 1 products
    """
    pass


# COMMAND: PROCESS
@cli.command()
@click.option('--geojson', '-g', help="File containing the geojson object for DEM bounds")
@click.option(
    "--rate", "-r", default=1, help="Rate at which to upsample DEM (default=1, no upsampling)"
)
@click.option(
    "--max-height",
    default=10,
    help="Maximum height/max absolute phase for converting .unw files to .tif"
    "(used for contour_interval option to dishgt)"
)
@click.option(
    "--step",
    "-s",
    type=click.IntRange(min=1, max=len(insar.scripts.process.STEPS)),
    help="Choose which step to start on. Steps: {}".format(insar.scripts.process.STEP_LIST),
    default=1
)
@click.option(
    "--max-temporal",
    type=int,
    default=500,
    help="Maximum temporal baseline for igrams (fed to sbas_list)"
)
@click.option(
    "--max-spatial",
    type=int,
    default=500,
    help="Maximum spatial baseline for igrams (fed to sbas_list)"
)
@click.option(
    "--looks",
    type=int,
    help="Number of looks to perform on .geo files to shrink down .int, "
    "Default is the upsampling rate, makes the igram size=original DEM size"
)
@click.option(
    "--lowpass",
    type=int,
    default=1,
    help="Size of lowpass filter to use on igrams before unwrapping"
)
@click.option(
    "--ref-row",
    type=int,
    help="Row number of pixel to use as unwrapping reference for SBAS inversion"
)
@click.option(
    "--ref-col",
    type=int,
    help="Column number of pixel to use as unwrapping reference for SBAS inversion"
)
@click.pass_obj
def process(context, **kwargs):
    """Process a stack of Sentinel interferograms

    Contains the steps from SLC .geo creation to SBAS deformation inversion"""
    if context['verbose']:
        click.echo("Verbose mode")

    insar.scripts.process.main(kwargs)
