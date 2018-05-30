import sys
from os.path import abspath, dirname, join, exists
try:
    import insar
except ImportError:  # add root to pythonpath if script is erroring
    sys.path.insert(0, dirname(dirname(abspath(__file__))))
from insar.dem import upsample_dem_rsc, create_kml
from insar.sario import load_dem_rsc

if __name__ == '__main__':
    rsc_data = load_dem_rsc('dem.rsc')
    print(create_kml(rsc_data, '20180420_20180502.tif'))
