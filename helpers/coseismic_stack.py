# coding: utf-8
from datetime import date
import numpy as np
import apertools.sario as sario
from insar.prepare import remove_ramp


def stack_igrams(
    event_date=date(2020, 3, 26),
    num_igrams=None,
    use_cm=True,
    rate=False,
    outname=None,
    verbose=True,
    ref=(5, 5),
    window=5,
    ignore_geos=True,
):

    gi_file = "geolist_ignore.txt" if ignore_geos else None
    geolist, intlist = sario.load_geolist_intlist('.', geolist_ignore_file=gi_file)
    print(geolist)

    insert_idx = np.searchsorted(geolist, event_date)
    num_igrams = num_igrams or len(geolist) - insert_idx

    # Since `event_date` will fit in the sorted array at `insert_idx`, then
    # geolist[insert_idx] is the first date AFTER the event
    geo_subset = geolist[insert_idx - num_igrams:insert_idx + num_igrams]

    stack_igrams = list(zip(geo_subset[:num_igrams], geo_subset[num_igrams:]))
    stack_fnames = sario.intlist_to_filenames(stack_igrams, '.unw')
    if verbose:
        print("Using the following igrams in stack:")
        for f in stack_fnames:
            print(f)

    dts = [(pair[1] - pair[0]).days for pair in stack_igrams]
    stack = np.zeros(sario.load(stack_fnames[0]).shape).astype(float)
    cc_stack = np.zeros_like(stack)
    dt_total = 0
    for f, dt in zip(stack_fnames, dts):
        deramped_phase = remove_ramp(sario.load(f), deramp_order=1, mask=np.ma.nomask)
        stack += deramped_phase
        dt_total += dt

        cc_stack += sario.load(f.replace(".unw", ".cc"))

    # subtract the reference location:
    ref_row, ref_col = ref
    win = window // 2
    patch = stack[ref_row - win:ref_row + win + 1, ref_col - win:ref_col + win + 1]
    stack -= np.nanmean(patch)

    cc_stack /= len(stack_fnames)
    if rate:
        stack /= dt
    else:
        stack /= len(stack_fnames)

    if use_cm:
        from insar.timeseries import PHASE_TO_CM
        stack *= PHASE_TO_CM

    if outname:
        import h5py
        with h5py.File(outname, 'w') as f:
            f['stackavg'] = stack
        sario.save_dem_to_h5(outname, sario.load("dem.rsc"))
    return stack, cc_stack
