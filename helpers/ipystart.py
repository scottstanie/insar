get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'tk')

import insar
from insar import *
from insar.scripts import *
import sardem
import eof
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import requests
