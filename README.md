[![Build Status](https://travis-ci.org/scottstanie/insar.svg?branch=master)](https://travis-ci.org/scottstanie/insar) 
[![Coverage Status](https://coveralls.io/repos/github/scottstanie/insar/badge.svg?branch=master)](https://coveralls.io/github/scottstanie/insar?branch=master)

# InSAR utils

Utilities for Synthetic apeture radar (SAR) and Interferometric SAR (InSAR) processing

Note that the upsampled DEM creator has moved to the [sardem](https://github.com/scottstanie/sardem) repo, and the Sentinel 1 EOF downloader has moved to the [sentineleof](https://github.com/scottstanie/sentineleof) repo, both pip installable.


## Setup and installation

```bash
pip install insar
```

### Some module example usage

#### sario.py

Input/Output functions for SAR data.
Contains methods to load Sentinel, UAVSAR or DEM files for now.

Main function: 

```python
import insar.sario
my_slc = insar.sario.load('/file/path/radar.slc')
my_int = insar.sario.load('/file/path/interferogram.int')
my_dem = insar.sario.load('/file/path/elevation.dem')
my_hgt = insar.sario.load('/file/path/N20W100.hgt')
```


#### parsers.py

Classes to deal with extracting relevant data from SAR filenames.
Example:

```python
from insar.parsers import Sentinel

parser = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70.zip')
parser.start_time
    datetime.datetime(2018, 4, 8, 4, 30, 25)

parser.mission
    'S1A'

parser.polarization
    'DV'
parser.full_parse
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


parser.field_meanings
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

UAVSAR parser also exists.

More will be added in the future.


#### log.py

Module to make logging pretty with times and module names.

If you also `pip install colorlog`, it will become colored (didn't require this in case people like non-color logs.)

```python
from insar.log import get_log
logger = get_log()
logger.info("Better than printing")
```

```
[05/29 16:28:19] [INFO log.py] Better than printing
```
