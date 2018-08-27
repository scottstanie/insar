get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'tk')

import insar
from insar import sario, utils, geojson, timeseries, plotting, parsers, latlon
import sardem
import sentineleof
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import requests
