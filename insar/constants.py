from math import pi
SENTINEL_WAVELENGTH = 5.5465763  # cm
PHASE_TO_CM = SENTINEL_WAVELENGTH / (4 * pi)
DATE_FMT = "%Y%m%d"

# Input files to timeseries solution:
MASK_FILENAME = "masks.h5"
INT_FILENAME = "int_stack.h5"
UNW_FILENAME = "unw_stack.h5"
COR_FILENAME = "cor_stack.h5"


# dataset names for general 3D stacks
STACK_DSET = "stack"
STACK_MEAN_DSET = "stack_mean"
STACK_FLAT_DSET = "stack_flat"

# output file
DEFO_FILENAME = "deformation.h5"
DEFO_FILENAME_NC = "deformation.nc"
DEFO_NOISY_DSET = "defo_noisy"
MODEL_DEFO_DSET = "defo_{model}"
ATMO_DAY1_DSET = "atmo_day1"

DEM_RSC_DSET = "dem_rsc"

REFERENCE_ATTR = "reference"
REFERENCE_STATION_ATTR = "reference_station"

# Used for converting xarray polyfit coefficients to normal rates
# (xarray converts the dates to "nanoseconds sicne 1970")
# https://github.com/pydata/xarray/blob/main/xarray/core/missing.py#L273-L280
# 1e9[ns/sec] * 86400[sec/day] * 365.25[day/year] ~ [ns/year]
NS_PER_YEAR = 1e9 * 86400 * 365.25
