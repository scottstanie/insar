# InSAR utils

Utilities for Synthetic apeture radar (SAR) and Interferometric SAR (InSAR) processing


## Setup and installation

```bash
pip install insar
```

This will put two scripts as executables on your path: `create-dem` and `download-eofs`
Other functionality is explained below.


Or for development use (to change code and have the reflected in what is installed):

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

Functions for working with digital elevation maps (DEMs) are mostly contained in the `Downloader` and `Stitcher` classes.


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
parser.start_stop_time()
    (datetime.datetime(2018, 4, 8, 4, 30, 25),
     datetime.datetime(2018, 4, 8, 4, 30, 25))

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

