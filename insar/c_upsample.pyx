from insar cimport c_upsample

def upsample(filename, rate, ncols, nrows, outfileUp="elevation.dem"):
    c_upsample.upsample(filename=filename, rate=rate, ncols=ncols, nrows=nrows, outfileUp=outfileUp)
