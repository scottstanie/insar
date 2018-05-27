# InSAR utils

Utilities for Synthetic apeture radar (SAR) and Interferometric SAR (InSAR) processing


## Setup and installation

or for development use (to add on and not make the installation fixed):

```bash
mkvirtualenv insar
pip install -r requirements.txt
python setup.py develop
```


### Modules and example usage

#### dem.py
Functions for working with digital elevation maps (DEMs).
Contains the `Downloader` and `Stitcher` classes.

In order to download a cropped (and possibly upsampled) dem,
see `scripts/create_dem.py`


```bash
$ python scripts/create_dem.py --geojson data/hawaii.geojson --rate 2 --output elevation.dem
```


#### eof.py

Functions for dealing with precise orbit files (POE) for Sentinel 1

```python
from insar.eof import download_eofs

download_eofs('20180503')
download_eofs(datetime.datetime(2018, 5, 3, 0, 0, 0))
```

#### sario.py

Input/Output functions for SAR data.
Mostly UAVSAR functions for now.

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

