# output file
DEFO_FILENAME = "deformation.h5"
DEFO_FILENAME_NC = "deformation.nc"
DEFO_NOISY_DSET = "defo_noisy"
MODEL_DEFO_DSET = "defo_{model}"
ATMO_DAY1_DSET = "atmo_day1"

# Used for converting xarray polyfit coefficients to normal rates
# (xarray converts the dates to "nanoseconds sicne 1970")
# https://github.com/pydata/xarray/blob/main/xarray/core/missing.py#L273-L280
# 1e9[ns/sec] * 86400[sec/day] * 365.25[day/year] ~ [ns/year]
NS_PER_YEAR = 1e9 * 86400 * 365.25
