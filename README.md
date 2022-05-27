# InSAR processing and timeseries analysis tools 

Utilities for Synthetic Apeture Radar (SAR) and Interferometric SAR (InSAR) processing.
Early processing steps use the Stanford Sentinel-1 geocoded SLC processor (not included here).


Many of the utilities have been split off into the [apertools](https://github.com/scottstanie/apertools) repo.
Note that the upsampled DEM creator has moved to the [sardem](https://github.com/scottstanie/sardem) repo, and the Sentinel 1 EOF downloader has moved to the [sentineleof](https://github.com/scottstanie/sentineleof) repo, both pip installable.


## Setup and installation

```bash
git clone https://github.com/scottstanie/insar && cd insar
pip install -r requirements.txt
pip install -e .
```

This will put the executable `insar` on your path with several commands available to use.


## Command Line Interface Reference

The command line tool is located in `insar/scripts/cli.py`.

```
$ insar --help
Usage: insar [OPTIONS] COMMAND [ARGS]...

  Command line tools for processing insar.

Options:
  --verbose
  --path DIRECTORY  Path of interest for command. Will search for files path
                    or change directory, depending on command.
  --help            Show this message and exit.

Commands:
  animate     Creates animation for 3D image stack.
  process     Process stack of Sentinel interferograms.
  view-stack  Explore timeseries on deformation image.
```

Mainly used for the `process` utility:

```bash
$ insar process --help
Usage: insar process [OPTIONS] [LEFT_LON] [TOP_LAT] [DLON] [DLAT]

  Process stack of Sentinel interferograms.

  Contains the steps from SLC .geo creation to SBAS deformation inversion

  left_lon, top_lat, dlon, dlat are used to specify the DEM bounding box.
  They may be ignored if not running step 2, and are an alternative to using
  --geojson

Options:
  --start INTEGER RANGE      Choose which step to start on, then run all
                             after. Steps: 1:create_dem, 2:download_data,
                             3:run_sentinel_stack, 4:prep_igrams_dir,
                             5:create_sbas_list, 6:run_form_igrams,
                             7:record_los_vectors, 8:run_snaphu,
                             9:convert_to_tif, 10:run_sbas_inversion

  --step TEXT                Run a one or a range of steps and exit. Examples:
                             --step 4,5,7 --step 3-6 --step 1,9-10

...
```