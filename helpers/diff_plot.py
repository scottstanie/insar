import apertools.sario as sario
import apertools.plotting as plotting


def diff_plot(f1='velocities_2019_linear_max800.h5', f2='velocities_2018_linear_max800.h5',
              dset="velos/1"):
    g1 = sario.load_geolist_from_h5(f1, dset="velos/1")
    g2 = sario.load_geolist_from_h5(f2, dset="velos/1")
    a1 = -sario.load(f1, dset=dset)
    a2 = sario.load(f2, dset=dset)
    c1 = a1 / 3650 * ((g1[-1] - g1[0]).days)
    c2 = a2 / 3650 * ((g2[-1] - g2[0]).days)
    plotting.plot_img_diff(arrays=[c1, c2], vm=5, vdiff=1)
    return c1, c2
