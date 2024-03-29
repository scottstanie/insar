import numpy as np
import rasterio as rio
from rasterio.errors import RasterioIOError
import h5py
import apertools.sario as sario
from apertools.sario import DATE_FMT
from apertools.constants import P2MM_S1 as P2MM
from apertools.deramp import remove_ramp
from collections import namedtuple

# TODO: do i want this
# from dataclasses import dataclass
# @dataclass
# class Igram:
#     early: datetime.date
#     late: datetime.date = ...  # Set default value

# TODO: figure out best place for this
Igram = namedtuple("Igram", "early late")


def _default_outfile(max_temporal_baseline, min_date, max_date):
    out = "velocities"
    if max_temporal_baseline:
        out += "_max{}".format(max_temporal_baseline)
    if min_date:
        out += "_from{}".format(min_date.strftime(DATE_FMT))
    if max_date:
        out += "_to{}".format(max_date.strftime(DATE_FMT))
    out += ".h5"
    return out


def sum_phase(filenames, band=2):
    print("summing phases...")
    with rio.open(filenames[0]) as ds:
        out = ds.read(band)

    for (idx, fname) in enumerate(filenames[1:]):
        if (idx + 1) % 10 == 0:
            print("Reading {} ({} of {})".format(fname, idx + 1, len(filenames)))
        try:
            with rio.open(fname, driver="ROI_PAC") as ds:
                out += ds.read(band)
        except RasterioIOError as e:
            print(f"Error during {fname}: {e}")
            continue
    return out


    # TODO: 
    # # Use the given reference, or find one on based on max correlation
    # if reference_station is not None:
    #     ref_row, ref_col = gps.station_rowcol(station_name=reference_station, rsc_data=rsc_data)
    # else:
    #     ref_row, ref_col = reference

def run_stack(
    # unw_stack_file="unw_stack.vrt",
    outfile=None,
    ignore_geo_file=None,
    geo_dir="../",
    igram_dir=".",
    ext=".unw",
    max_temporal_baseline=900,
    min_date=None,
    max_date=None,
    ramp_order=1,
):
    if outfile is None:
        outfile = _default_outfile(max_temporal_baseline, min_date, max_date)

    # outgroup = "velos"  # Do i care about splitting up?
    outdset = "velos/1"
    # input_dset = STACK_FLAT_DSET

    # Filter igrams and dates down from baselines/bad data
    slclist, ifglist, valid_idxs = load_slclist_ifglist(  # unw_stack_file,
        ignore_geo_file,
        max_temporal_baseline,
        geo_dir=geo_dir,
        igram_dir=igram_dir,
        min_date=min_date,
        max_date=max_date,
    )

    unw_files = sario.ifglist_to_filenames(ifglist, ext)
    if not unw_files:
        print(f"no files with {ext} here, exiting.")
        return
    print("loading {} files:".format(len(unw_files)))
    print(unw_files[:5], "...")
    phase_sum = sum_phase(unw_files)
    # TODO: how to handle nodata masks...

    phase_sum = remove_ramp(phase_sum, order=ramp_order, mask=np.ma.nomask)

    timediffs = [temporal_baseline(ig) for ig in ifglist]  # units: days
    avg_velo = phase_sum / np.sum(timediffs)  # phase / day
    avg_velo2 = phase_sum / len(timediffs)  # phase / day
    out = (P2MM * avg_velo).astype("float32")  # MM / year
    print(np.max(phase_sum), np.min(phase_sum))
    print(np.max(avg_velo), np.min(avg_velo))
    print(np.max(avg_velo2), np.min(avg_velo2))
    print(np.max(out), np.min(out))

    print("Writing solution into {}:{}".format(outfile, outdset))
    with h5py.File(outfile, "w") as f:
        f.create_dataset(outdset, data=out)

    sario.save_dem_to_h5(outfile, sario.load("dem.rsc"))
    # TODO: save Igram pairs to vrt, slclist to vrt?
    # slclist = sario.find_geos(directory=igram_dir, parse=True)

    # Finally, save as a mm/year velocity

    # h5open(outfile, "cw") do f
    #     f[outdset] = permutedims(out)
    #     # TODO: Do I care to add this for stack when it's all the same?
    #     # f[_count_dset(cur_outdset)] = countstack
    # end

    # return outfile, outdset
    return out


def load_slclist_ifglist(
    # unw_stack_file,
    ignore_geo_file,
    max_temporal_baseline,
    geo_dir="../",
    igram_dir=".",
    min_date=None,
    max_date=None,
):
    slclist = sario.find_geos(directory=geo_dir, parse=True)
    ifglist = sario.find_igrams(directory=igram_dir, parse=True)
    # slclist = sario.load_slclist_from_h5(unw_stack_file)
    # ifglist = sario.load_ifglist_from_h5(unw_stack_file)

    # If we are ignoreing some indices, remove them from for all pixels
    # geo_idxs, igram_idxs = find_valid_indices(slclist, ifglist, min_date, max_date,...
    # return slclist[geo_idxs], ifglist[igram_idxs], igram_idxs
    # TODO GET INDEXES FOR UNWS
    return find_valid(
        slclist,
        ifglist,
        min_date,
        max_date,
        ignore_geo_file,
        max_temporal_baseline,
    )


def find_valid(
    geo_date_list,
    igram_date_list,
    min_date=None,
    max_date=None,
    ignore_geo_file=None,
    max_temporal_baseline=900,
):
    """Cut down the full list of interferograms and geo dates

    - Cuts date ranges (e.g. piecewise linear solution) with  `min_date` and `max_date`
    - Can ignore whole dates by reading `ignore_geo_file`
    - Prunes long time baseline interferograms with `max_temporal_baseline`
    """
    ig1 = len(igram_date_list)  # For logging purposes, what do we start with
    if not ignore_geo_file:
        print("Not ignoring any .geo dates")
        ignore_geos = []
    else:
        ignore_geos = sorted(sario.find_geos(filename=ignore_geo_file, parse=True))
        print("Ignoring the following .geo dates:")
        print(ignore_geos)

    # TODO: do I want to be able to pass an array of dates to ignore?

    # First filter by remove igrams with either date in `ignore_geo_file`
    valid_geos = [g for g in geo_date_list if g not in ignore_geos]
    valid_igrams = [
        ig
        for ig in igram_date_list
        if (ig[0] not in ignore_geos and ig[1] not in ignore_geos)
    ]
    print(f"Ignoring {ig1 - len(valid_igrams)} igrams listed in {ignore_geo_file}")

    # Remove geos and igrams outside of min/max range
    if min_date is not None:
        print(f"Keeping data after min_date: {min_date}")
        valid_geos = [g for g in valid_geos if g > min_date]
        valid_igrams = [
            ig for ig in valid_igrams if (ig[0] > min_date and ig[1] > min_date)
        ]

    if max_date is not None:
        print(f"Keeping data only before max_date: {max_date}")
        valid_geos = [g for g in valid_geos if g < max_date]
        valid_igrams = [
            ig for ig in valid_igrams if (ig[0] < max_date and ig[1] < max_date)
        ]

    # This is just for logging purposes:
    too_long_igrams = [
        ig for ig in valid_igrams if temporal_baseline(ig) > max_temporal_baseline
    ]
    print(
        f"Ignoring {len(too_long_igrams)} igrams with longer baseline "
        f"than {max_temporal_baseline} days"
    )

    # ## Remove long time baseline igrams ###
    valid_igrams = [
        ig for ig in valid_igrams if temporal_baseline(ig) <= max_temporal_baseline
    ]

    print(f"Ignoring {ig1 - len(valid_igrams)} igrams total")
    # Now go back to original full list and see what the indices were
    # used for subselecting from the unw_stack by a pixel

    valid_idxs = np.searchsorted(
        sario.ifglist_to_filenames(igram_date_list),
        sario.ifglist_to_filenames(valid_igrams),
    )
    return valid_geos, valid_igrams, valid_idxs


def temporal_baseline(igram):
    return (igram[1] - igram[0]).days


# span(dates::AbstractArray{Date, 1}) = (dates[end] - dates[1]).value
# span(igrams::AbstractArray{Igram, 1}) = (sort(igrams)[end][2] - sort(igrams)[1][1]).value
#
# _get_day_nums(dts::AbstractArray{Date, 1}) = [( d - dts[1]).value for d in dts]
