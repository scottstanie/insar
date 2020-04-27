import numpy as np
from numpy import sqrt, real, conj
from glob import glob
from apertools.utils import take_looks
import apertools.sario as sario
from apertools.log import get_log

logger = get_log()

EPS = np.finfo(np.float32).eps


def abs2(x):
    # Weird, but...
    # %timeit np.abs(b)**2
    # 13 ms ± 3.31 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    # %timeit b.real**2 + b.imag**2
    # 1.53 ms ± 2.04 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    return x.real**2 + x.imag**2


def make_igam(slc1, slc2, rowlooks, collooks):
    return take_looks(slc1 * conj(slc2), rowlooks, collooks)


def powlooks(image, rowlooks, collooks):
    return take_looks(abs2(image), rowlooks, collooks)


def make_int_cor(
    slc1,
    slc2,
    rowlooks,
    collooks,
):
    igram = make_igam(slc1, slc2, rowlooks, collooks)
    return _make_cor(igram, slc1, slc2, rowlooks, collooks)


def _make_cor(
    igram,
    slc1,
    slc2,
    rowlooks,
    collooks,
):
    ampslc1 = sqrt(powlooks(slc1, rowlooks, collooks))
    ampslc2 = sqrt(powlooks(slc2, rowlooks, collooks))
    amp = real(np.abs(igram))
    # @show extrema(ampslc1), extrema(ampslc2), extrema(amp)
    cor = real(amp / (EPS + (ampslc1 * ampslc2)))
    return cor, amp, igram


# julia> lines = readlines("sbas_list")
# 4-element Array{String,1}:
#  "./S1A_20141104.ge ./S1A_2014128.ge 24.0    29539676548307892     " ...
def form_igram_names():
    with open("sbas_list") as f:
        sbas_lines = f.read().splitlines()
    # TODO: use the parsers to get the dates...
    out = []
    for line in sbas_lines:
        early_file, late_file, temp, spatial = line.split()
        # "./S1A_20141104.ge
        igram_name = '_'.join(map(_get_date, [early_file, late_file])) + ".int"
        out.append((igram_name, str(early_file), str(late_file)))

    # Note: orting so that ALL igrams with `early_file` are formed in a row
    return sorted(out)


def _get_date(geo_name):
    # "./S1A_20141128.geo" -> "20141128"
    return geo_name.split('_')[1].split('.')[0]


def create_igrams(rowlooks=1, collooks=1):
    current_ints = glob("*.int")
    current_cors = glob("*.cc")

    fulldemrsc = sario.load("../elevation.dem.rsc")
    demrsc = take_looks(fulldemrsc, rowlooks, collooks)
    sario.save("dem.rsc", demrsc)

    cur_early_file = ""
    for (igram_name, early_file, late_file) in form_igram_names():
        cor_name = igram_name.replace(".int", ".cc")
        if (igram_name in current_ints) and (cor_name in current_cors):
            logger.debug(f"Skipping {igram_name} and {cor_name}: exists")
            continue
        else:
            logger.debog(f"Forming {igram_name} and {cor_name}")

        # Keep early in memory for all pairs: only load for new set
        if cur_early_file != early_file:
            logger.debug(f"Loading {early_file}")
            early = sario.load(early_file)
            cur_early_file = early_file

        # But we load late every time
        logger.debug(f"Loading {late_file}")
        late = sario.load(late_file)

        # TODO: check if window_read with rasterio is faster than loading huge files?

        logger.debug("Forming amps")
        ampslc1 = sqrt(powlooks(early, rowlooks, collooks))

        ampslc2 = sqrt(powlooks(late, rowlooks, collooks))
        logger.debug("Forming igram")
        igram = make_igam(early, late, rowlooks, collooks)

        logger.debug("Forming cor")
        amp = real(np.abs(igram))
        cor = real(amp / (EPS + (ampslc1 * ampslc2)))

        logger.info(f"Saving {cor_name}, {igram_name}")
        sario.save(cor_name, np.stack([amp, cor], axis=0))
        sario.save(igram_name, igram)
