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
    cc_thresh=None,
    avg_cc_thresh=0.35,
    sigma_filter=.3,
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
    cur_phase_sum = np.zeros(sario.load(stack_fnames[0]).shape).astype(float)
    cc_stack = np.zeros_like(cur_phase_sum)
    # for pixels that get masked sometimes, lower that count in the final stack dividing
    pixel_count = np.zeros_like(cur_phase_sum, dtype=int)
    dt_total = 0
    for f, dt in zip(stack_fnames, dts):
        deramped_phase = remove_ramp(sario.load(f), deramp_order=1, mask=np.ma.nomask)
        cur_cc = sario.load(f.replace(".unw", ".cc"))

        if cc_thresh:
            bad_pixel_mask = cur_cc < cc_thresh
        else:
            # zeros => dont mask any to nan
            bad_pixel_mask = np.zeros_like(deramped_phase, dtype=bool)

        deramped_phase[bad_pixel_mask] = np.nan

        # cur_phase_sum += deramped_phase
        cur_phase_sum = np.nansum(np.stack([cur_phase_sum, deramped_phase]), axis=0)
        pixel_count += (~bad_pixel_mask).astype(int)
        dt_total += ((~bad_pixel_mask) * dt)

        cc_stack += cur_cc

    # subtract the reference location:
    ref_row, ref_col = ref
    win = window // 2
    patch = cur_phase_sum[ref_row - win:ref_row + win + 1, ref_col - win:ref_col + win + 1]
    cur_phase_sum -= np.nanmean(patch)

    if rate:
        cur_phase_sum /= dt_total
    else:
        cur_phase_sum /= pixel_count
    cc_stack /= len(stack_fnames)

    if avg_cc_thresh:
        cur_phase_sum[cc_stack < avg_cc_thresh] = np.nan

    if use_cm:
        from insar.timeseries import PHASE_TO_CM
        cur_phase_sum *= PHASE_TO_CM

    if sigma_filter:
        import insar.blob.utils as blob_utils
        cur_phase_sum = blob_utils.gaussian_filter_nan(cur_phase_sum, sigma_filter)

    if outname:
        import h5py
        with h5py.File(outname, 'w') as f:
            f['stackavg'] = cur_phase_sum
        sario.save_dem_to_h5(outname, sario.load("dem.rsc"))
    return cur_phase_sum, cc_stack
