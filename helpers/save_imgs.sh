insar kml --no-preview --rsc dem.rsc --dset "velos/1" --vmin -20 --vmax 20 --cmap seismic_wide velocities_noprune_stackavg.h5 > velocities_noprune_stackavg.kml
insar kml --no-preview --rsc dem.rsc --dset "velos/1" --vmin -20 --vmax 20 --cmap seismic_wide velocities_prune_l2.h5 > velocities_prune_l2.kml
insar kml --no-preview --rsc dem.rsc --dset "velos/1" --vmin -20 --vmax 20 --cmap seismic_wide velocities_noprune_l2.h5 > velocities_noprune_l2.kml
insar kml --no-preview --rsc dem.rsc --dset "velos/1" --vmin -20 --vmax 20 --cmap seismic_wide velocities_noprune_l1.h5 > velocities_noprune_l1.kml
insar kml --no-preview --rsc dem.rsc --dset "velos/1" --vmin -20 --vmax 20 --cmap seismic_wide velocities_prune_l1.h5 > velocities_prune_l1.kml
insar kml --no-preview --rsc dem.rsc --dset "stddev/1" --vmin 0 --vmax 15 --cmap jet velocities_prune_l1.h5 --no-shifted --imgout velocities_stddev.png > velocities_stddev.kml
insar kml --no-preview --rsc dem.rsc --dset "stddev_raw/1" --vmin 0 --vmax 15 --cmap jet velocities_prune_l1.h5 --no-shifted --imgout velocities_stddev_raw.png > velocities_stddev_raw.kml

# TODO: add 85_prefix
