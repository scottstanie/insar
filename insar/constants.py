from math import pi
SENTINEL_WAVELENGTH = 5.5465763  # cm
PHASE_TO_CM = SENTINEL_WAVELENGTH / (4 * pi)


DATE_FMT = "%Y%m%d"

MASK_FILENAME = "masks.h5"
INT_FILENAME = "int_stack.h5"
UNW_FILENAME = "unw_stack.h5"
COR_FILENAME = "cor_stack.h5"

# dataset names for general 3D stacks
STACK_DSET = "stack"
STACK_MEAN_DSET = "stack_mean"
STACK_FLAT_DSET = "stack_flat"
STACK_FLAT_SHIFTED_DSET = "stack_flat_shifted"

# Mask file datasets
GEO_MASK_DSET = "geo"
GEO_MASK_SUM_DSET = "geo_sum"
IGRAM_MASK_DSET = "igram"
IGRAM_MASK_SUM_DSET = "igram_sum"

DEM_RSC_DSET = "dem_rsc"

SLCLIST_DSET = "geo_dates"
IFGLIST_DSET = "int_dates"

REFERENCE_ATTR = "reference"
REFERENCE_STATION_ATTR = "reference_station"
