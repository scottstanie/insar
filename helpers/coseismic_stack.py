# coding: utf-8
from datetime import datetime
import numpy as np
import apertools.sario as sario


def stack_igrams(event_date=datetime.date(2020, 3, 26)):

    geolist, intlist = sario.load_geolist_intlist('.')
    insert_idx = np.searchsorted(geolist, event_date)
    num_igrams = len(geolist) - insert_idx

    geo_subset = geolist[-(2 * num_igrams):]
    stack_igrams = list(zip(geo_subset[:num_igrams], geo_subset[num_igrams:]))
    stack_fnames = sario.intlist_to_filenames(stack_igrams, '.unw')
    dts = [(pair[1] - pair[0]).days for pair in stack_igrams]
    stack = np.sum([sario.load(f) / dt for (f, dt) in zip(stack_fnames, dts)])
    stack = np.sum(np.stack([sario.load(f) / dt for (f, dt) in zip(stack_fnames, dts)], axis=0),
                   axis=0)
    return stack
