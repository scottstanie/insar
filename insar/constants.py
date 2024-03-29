# output file
from apertools.sario import LOS_FILENAME


DEFO_FILENAME = "deformation.h5"
DEFO_FILENAME_NC = "deformation.nc"
DEFO_NOISY_DSET = "defo_noisy"
DEFO_LOWESS_DSET = "defo_lowess"
LINEAR_VELO_DSET = "linear_velocity"
MODEL_DEFO_DSET = "defo_{model}"
ATMO_DAY1_DSET = "atmo_day1"
COR_MEAN_DSET = "cor_mean"
COR_STD_DSET = "cor_std"
TEMP_COH_DSET = "temp_coh"

LOS_ENU_FILENAME = "los_enu.tif"
LOS_ENU_DSET = "los_enu"

# Used for converting xarray polyfit coefficients to normal rates
# (xarray converts the dates to "nanoseconds sicne 1970")
# https://github.com/pydata/xarray/blob/main/xarray/core/missing.py#L273-L280
# 1e9[ns/sec] * 86400[sec/day] * 365.25[day/year] ~ [ns/year]
NS_PER_YEAR = 1e9 * 86400 * 365.25
