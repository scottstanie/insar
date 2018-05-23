# InSAR utils

Utilities for SAR and InSAR processing

So far mostly helper functions for loading the many binary data types produced by sar, downloading and manipulating auxiliary data (like DEMs or orbit files).

Also contains several ad-hoc scripts for different remote sensing projects.

```bash
    mkvirtualenv insar
    pip install -r requirements.txt
```


### Modules and example uses

#### eof.py

Functions for dealing with precise orbit files (Sentinel 1)

```python
	from insar.eof import download_eofs

    download_eofs('20180503')
    download_eofs(datetime.datetime(2018, 05, 03, 0, 0, 0))
```

#### sario.py

Input/Output functions for SAR data.
Mostly UAVSAR functions for now.

Main function: 

```python
	import insar.sario
    insar.sario.load_file('/file/path')
```


#### dem.py
Functions for working with digital elevation maps (DEMs).

Useful functions: `upsample_dem`, `upsample_dem_rsc`, `mosaic_dem`

#### parsers.py

Classes to deal with extracting relevant data from SAR filenames.
Example:

```python
	parser = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70.zip')
	parser.start_stop_time()
	parser.mission()
	parser.polarization()
	parser.full_parse()
	parser.field_meanings()
```


### Scripts:

Most useful script for now: `download_eofs.py`.
Grabs relevant EOF data for all Sentinel 1 products in the current directory.
