The files have been produced by the `insar process` command from https://github.com/scottstanie/insar

### Time series files and data formats

After Step 10 of the `insar process` command, there will be a set of files containing the interferogram input and time series outputs.
The most important files are

#### From `prepare.py`, preparing the input stacks

- unw_stack.h5 : The unwrapped interferograms
- cor_stack.h5 : The correlation estimates for each interferogram
- masks.h5 : Contains both SLC-level and interferogram-level masks for ignore nodata / bad data.
- ifg_stack.h5 : The wrapped interferograms. This file is only optional to be created, as it is not used for the SBAS inversion.


The datasets are designed for ease of use for manipulating using [xarray](https://docs.xarray.dev/en/stable/index.html), the NetCDF python library https://unidata.github.io/netcdf4-python/ (or equivalently the MATLAB netcdf functions), or the `h5py` library for HDF5 files.
The have the necessary geographic metadata in their `lat` and `lon` datasets to make GIS-ingestible files from slices of the stacks.


```
$ ncdump -h unw_stack.h5
netcdf unw_stack {
dimensions:
	ifg_idx = 188 ;
	lat = 1021 ;
	lon = 1021 ;
...
variables:
	double date(phony_dim_3) ;
		string date:units = "days since 1970-01-01T00:00:00" ;
	double ifg_dates(ifg_idx, phony_dim_4) ;
		string ifg_dates:units = "days since 1970-01-01T00:00:00" ;
	int64 ifg_idx(ifg_idx) ;
	double lat(lat) ;
	double lon(lon) ;
	double slc_dates(phony_dim_3) ;
	float stack_flat_shifted(ifg_idx, lat, lon)
```

The `lat` and `lon` dataset are 1D arrays containing the latitude and longitude coordinates.
The `date` array contains the SLC dates (stored as "days since 1970-01-01" per the NetCDF convention.)

The `stack_flat_shifted` contains the unwrapped interferograms which have been (optionally) deramped using some polynomial surface fit, and (optionally) shifted to some reference point or reference GPS station.
There are attributes of this dataset recording the deramp_order , reference (row, col) , reference_latlon, reference_window, and reference_station.


The `cor_stack.h5` has analogous datasets, with the `stack` dataset containing the 3D stack of correlation images (there is no `flat` or `shifted` for correlation files).



#### Extra files used for the timeseries
- elevation_looked.dem: a multilooked version of the DEM used to geocode the SLCs. this will be the same shape as the interferograms.
- los_enu.tif: a 3-band Geotiff containing the line-of-sight look vector (from satellite, to ground point). Bands (1,2,3) are the (east, north, up) coefficients for projecting ENU to LOS.
- slclist_ignore.txt: a text file containing the YYYYMMDD dates of SLCs to exclude for whatever reason (missing data, too much decorrelation, etc.). All interferograms containing any date from this fill will be ignored.

#### Deformation output

The final smoothed deformation data is in the `deformation.nc`. The full datasets contained can be viewed with the `ncdump` tool

```bash
$ ncdump -h deformation.nc # headers only
netcdf deformation {
dimensions:
	lat = 1021 ;
	lon = 1021 ;
	date = 34 ;
	...
variables:
	...
	float defo_noisy(date, lat, lon) ;
	...
    float defo_lowess(date, lat, lon) ;
```

The `defo_noisy` dataset is the solution of SBAS, which solves for the phase delay at each SLC date (deformation plus atmospheric noise).  `defo_lowess` Contains the LOWESS-smoothed version designed to extract the long-term deformation.

To get a single GeoTIFF file from (for example) the final deformation image, you can use `gdal_translate` to pick (in this case) the 34th band.

```bash
# the format is "NETCDF":<filename>:/<dataset name>
# the -b 34 means just pick the 34th band
$ gdal_translate -b 34 NETCDF:deformation.nc:/defo_lowess final_deformation.tif
Input file size is 1021, 1021
```