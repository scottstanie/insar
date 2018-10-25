import os
import numpy as np
import sardem
from insar import blob, timeseries, latlon

if __name__ == '__main__':
    # array([ 4,  5,  7, 10, 13, 18, 25, 34, 47, 64])
    igram_path = '.'
    geolist, deformation = timeseries.load_deformation(igram_path)
    rsc_data = sardem.loading.load_dem_rsc(os.path.join(igram_path, 'dem.rsc'))
    img = latlon.LatlonImage(data=np.mean(deformation[-3:], axis=0), dem_rsc=rsc_data)

    threshold_list = [0.3, 0.5, 0.8, 1]  # For filter response
    value_threshold_list = np.linspace(0.2, 2, 10)  # Blob magnitude

    for thresh in threshold_list:
        for val_thresh in value_threshold_list:
            extra_args = {'threshold': thresh, 'value_threshold': val_thresh}
            blobs = blob._make_blobs(img, extra_args)

            blobs_name = 'blobs_{}_{}.npy'.format(thresh, val_thresh)
            print("saving", blobs_name, "size:", len(blobs))
            np.save(blobs_name, blobs)
