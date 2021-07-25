import numpy as np
from numpy import sqrt, real, conj
from glob import glob
from apertools.utils import take_looks
import apertools.sario as sario
from apertools.log import get_log

logger = get_log()

EPS = np.finfo(np.float32).eps


def abs2(x):
    # Weird, but it seems to be faster...
    # %timeit np.abs(b)**2
    # 13 ms ± 3.31 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    # %timeit b.real**2 + b.imag**2
    # 1.53 ms ± 2.04 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    return x.real ** 2 + x.imag ** 2


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
    amp = np.abs(igram)
    # @show extrema(ampslc1), extrema(ampslc2), extrema(amp)
    cor = real(amp / (EPS + (ampslc1 * ampslc2)))
    return cor, amp, igram


# julia> lines = readlines("sbas_list")
# 4-element Array{String,1}:
#  "./S1A_20141104.ge ./S1A_2014128.ge 24.0    29539676548307892     " ...
def form_igram_names(igram_ext=".int"):
    with open("sbas_list") as f:
        sbas_lines = f.read().splitlines()
    # TODO: use the parsers to get the dates...
    out = []
    for line in sbas_lines:
        # early_file, late_file, temp, spatial = line.split()
        early_file, late_file, _, _ = line.split()
        # "./S1A_20141104.ge
        igram_name = "_".join(map(_get_date, [early_file, late_file])) + igram_ext
        out.append((igram_name, str(early_file), str(late_file)))

    # Note: orting so that ALL igrams with `early_file` are formed in a row
    return sorted(out)


def _get_date(geo_name):
    # "./S1A_20141128.geo" -> "20141128"
    return geo_name.split("_")[1].split(".")[0]


def _load_gdal(fname):
    import rasterio as rio

    with rio.open(fname) as src:
        return src.read(1)


def create_igrams(rowlooks=1, collooks=1, igram_ext=".int"):
    current_ints = glob("*" + igram_ext)
    current_cors = glob("*.cc")

    fulldemrsc = sario.load("../elevation.dem.rsc")
    demrsc = take_looks(fulldemrsc, rowlooks, collooks)
    sario.save("dem.rsc", demrsc)

    cur_early_file = ""
    for (igram_name, early_file, late_file) in form_igram_names():
        cor_name = igram_name.replace(igram_ext, ".cc")
        if (igram_name in current_ints) and (cor_name in current_cors):
            logger.debug(f"Skipping {igram_name} and {cor_name}: exists")
            continue
        else:
            logger.debug(f"Forming {igram_name} and {cor_name}")

        # Keep early in memory for all pairs: only load for new set
        if cur_early_file != early_file:
            logger.debug(f"Loading {early_file}")
            # early = sario.load(early_file)
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


def _get_weights(wsize):
    assert wsize % 2 == 1
    return 1 - np.abs(2 * (np.arange(wsize) - wsize // 2)) / (wsize + 1)


def _get_weights_square(wsize):
    w = _get_weights(wsize)
    return w.reshape((-1, 1)) * w.reshape((1, -1))


try:
    import numba
    import cupy as cp
    from cupyx.scipy.ndimage import correlate as correlate_gpu
    from scipy.ndimage import correlate
except ImportError:
    print("cupy/numba not installed, no gpu")
from apertools.utils import read_blocks, block_iterator


def softplus(x):
    xp = cp.get_array_module(x)
    return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))


def make_igram_gpu(
    early_filename,
    late_filename,
    block_size=(500, 500),
    overlaps=None,
    wsize=5,
    out_ifg="out.int",
    out_cor="out.cor",
):
    from rasterio.windows import Window
    import rasterio as rio

    if overlaps is None:
        overlaps = (wsize // 2, wsize // 2)

    out_cor = "testcor.tif"
    with rio.open(early_filename) as src:
        full_shape = src.shape
    blks1 = read_blocks(early_filename, window_shape=block_size, overlaps=overlaps)
    blks2 = read_blocks(late_filename, window_shape=block_size, overlaps=overlaps)
    blk_slices = block_iterator(src.shape, block_size, overlaps=overlaps)
    # Write empty file
    _write(out_ifg, None, early_filename, "ROI_PAC", dtype=np.complex64)
    _write(out_cor, None, early_filename, "GTiff", dtype=np.float32)

    w_cpu = _get_weights_square(wsize)
    w = cp.asarray(w_cpu)
    # w = w_cpu

    for slc1_cpu, slc2_cpu, win_slice in zip(blks1, blks2, blk_slices):
        print(f"Forming {win_slice = }")
        # slc1 = cp.asarray(slc1_cpu)
        # slc2 = cp.asarray(slc2_cpu)
        slc1 = slc1_cpu
        slc2 = slc2_cpu

        ifg = slc1 * slc2.conj()

        # Correlation
        amp1 = slc1.real ** 2 + slc1.imag ** 2
        amp2 = slc2.real ** 2 + slc2.imag ** 2
        denom = correlate_gpu(cp.sqrt(amp1 * amp2), w)
        numer = correlate_gpu(cp.abs(ifg), w)
        # denom = correlate(np.sqrt(amp1 * amp2), w)
        # numer = correlate(np.abs(ifg), w)
        cor = numer / (EPS + denom)

        ifg_cpu = cp.asnumpy(ifg)
        cor_cpu = cp.asnumpy(cor)
        # ifg_cpu = ifg
        # cor_cpu = cor

        _write(
            out_ifg,
            ifg_cpu,
            early_filename,
            "ROI_PAC",
            window=Window.from_slices(*win_slice),
            mode="r+",
        )
        _write(
            out_cor,
            cor_cpu,
            early_filename,
            "GTiff",
            window=Window.from_slices(*win_slice),
            mode="r+",
        )


def _write(outname, img, in_name, driver, mode="w", window=None, dtype=None):
    import rasterio as rio

    if dtype is None:
        dtype = img.dtype
    with rio.open(in_name) as src:
        full_height, full_width = src.shape
        transform, crs, nodata = src.transform, src.crs, src.nodata
    with rio.open(
        outname,
        mode,
        driver=driver,
        width=full_width,
        height=full_height,
        count=1,
        dtype=dtype,
        transform=transform,
        crs=crs,
        nodata=nodata,
    ) as dst:
        if img is not None:
            dst.write(img, window=window, indexes=1)


from apertools.utils import memmap_blocks


def make_igram_blocks(
    early_filename,
    late_filename,
    full_shape,
    looks=(1, 1),
    block_rows=1000,
    out_ifg="out.int",
    out_cor="out.cor",
):

    blks1 = memmap_blocks(early_filename, full_shape, block_rows, "complex64")
    blks2 = memmap_blocks(late_filename, full_shape, block_rows, "complex64")

    with open("testifg.int", "wb") as f:
        for idx, (slc1, slc2) in enumerate(zip(blks1, blks2)):
            print(f"Forming {idx = }")

            ifg = slc1 * slc2.conj()
            take_looks(ifg, *looks).tofile(f)
