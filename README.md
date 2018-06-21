[![Build Status](https://travis-ci.org/scottstanie/insar.svg?branch=master)](https://travis-ci.org/scottstanie/insar) 
[![Coverage Status](https://coveralls.io/repos/github/scottstanie/insar/badge.svg?branch=master)](https://coveralls.io/github/scottstanie/insar?branch=master)

# InSAR utils

Utilities for Synthetic apeture radar (SAR) and Interferometric SAR (InSAR) processing


## Setup and installation

```bash
pip install insar
```

This will put three scripts as executables on your path: `create-dem`,`download-eofs`, and `view-dem`.
Other functionality is explained below.


Or for development use (to change code and have it be reflected in what is installed):

```bash
mkvirtualenv insar
pip install -r requirements.txt
python setup.py develop
```

virtualenv is optional but recommended.


### Modules and example usage

#### dem.py
In order to download a cropped (and possibly upsampled) dem,
see `scripts/create_dem.py`


```bash
create-dem --geojson data/hawaii.geojson --rate 2 --output elevation.dem
create-dem -g data/hawaii_bigger.geojson -r 5 --output elevation.dem
```

The geojson can be any valid simple Polygon- you can get one easily from http://geojson.io , for example.

Functions for working with digital elevation maps (DEMs) are mostly contained in the `Downloader` and `Stitcher` classes.

Once you have made this, if you want to get a quick look in python, the script `script/view_dem.py` opens the file and plots with matplotlib.
You can access the script with the entry-point `view-dem`, installed with a `pip install`.
If you have multiple, you can plot them:

```bash
view-dem elevation1.dem elevation2.dem
view-dem  # Looks in the current directory for "elevation.dem"
```

The default datasource is NASA's SRTM version 3 global 1 degree data.
See https://lpdaac.usgs.gov/dataset_discovery/measures/measures_products_table/srtmgl3s_v003

This data requires a username and password from here:
https://urs.earthdata.nasa.gov/users/new

You will be prompted for a username and password when running with NASA data.
It will save into your ~/.netrc file for future use, which means you will not have to enter a username and password any subsequent times.
The entry will look like this:

```
machine urs.earthdata.nasa.gov
    login USERNAME
    password PASSWORD
```

If you want to avoid this entirely, you can [use Mapzen's data] (https://registry.opendata.aws/terrain-tiles/) by specifying
```bash
create-dem -g data/hawaii_bigger.geojson --data-source AWS
```

`--data-source NASA` is the default.

Mapzen combines SRTM data with other sources, so the .hgt files will be slightly different.
They also list that they are discontinuing some services, which is why NASA is the default.


#### eof.py

Functions for dealing with precise orbit files (POE) for Sentinel 1

```bash
$ download-eofs
```

The script without arguments will look in the current directory for .EOF files.
You can also specify dates, with or without a mission (S1A/S1B):

```bash
download-eofs --date 20180301 
download-eofs -d 2018-03-01 --mission S1A
```

Using it from python, you can pass a list of dates:

```python
from insar.eof import download_eofs

download_eofs([datetime.datetime(2018, 5, 3, 0, 0, 0)])
download_eofs(['20180503', '20180507'], ['S1A', 'S1B'])
```

#### sario.py

Input/Output functions for SAR data.
Mostly UAVSAR or DEM functions for now.

Main function: 

```python
import insar.sario
my_slc = insar.sario.load_file('/file/path/radar.slc')
my_dem = insar.sario.load_file('/file/path/elevation.hgt')
```


#### parsers.py

Classes to deal with extracting relevant data from SAR filenames.
Example:

```python
from insar.parsers import Sentinel

parser = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70.zip')
parser.start_time()
    datetime.datetime(2018, 4, 8, 4, 30, 25)

parser.mission()
    'S1A'

parser.polarization()
    'DV'
parser.full_parse()
('S1A',
 'IW',
 'SLC',
 '_',
 '1',
 'S',
 'DV',
 '20180408T043025',
 '20180408T043053',
 '021371',
 '024C9B',
 '1B70')


parser.field_meanings()
('Mission',
 'Beam',
 'Product type',
 'Resolution class',
 'Product level',
 'Product class',
 'Polarization',
 'Start datetime',
 'Stop datetime',
 'Orbit number',
 'data-take identified',
 'product unique id')

```

More will be added in the future.


#### geojson.py

Simple functions for getting handling geojson inputs:


```python
from insar.geojson import read_json, bounding_box, print_coordinates
json_dict = read_json(input_string)
```

Running the module as a script will give you both the bounding box, and the comma-joined lon,lat pairs of the polygon:

```bash
$ cat data/hawaii.geojson | python insar/geojson.py 
-155.67626953125,19.077692991868297,-154.77264404296875,19.077692991868297,-154.77264404296875,19.575317892869453,-155.67626953125,19.575317892869453,-155.67626953125,19.077692991868297
-155.67626953125 19.077692991868297 -154.77264404296875 19.575317892869453

$ cat data/hawaii.geojson 
{
  "type": "Polygon",
  "coordinates": [
    [
		  [
		    -155.67626953125,
		    19.077692991868297
		  ],
		  [
		    -154.77264404296875,
		    19.077692991868297
		  ],
		  [
		    -154.77264404296875,
		    19.575317892869453
		  ],
		  [
		    -155.67626953125,
		    19.575317892869453
		  ],
		  [
		    -155.67626953125,
		    19.077692991868297
		  ]
    ]
  ]
}
```

#### log.py

Module to make logging pretty with times and module names.

If you also `pip install colorlog`, it will become colored (didn't require this in case people like non-color logs.)

```python
from insar.log import get_log
logger = get_log()
logger.info("Better than printing")
```

```bash
[05/29 16:28:19] [INFO log.py] Better than printing
```
