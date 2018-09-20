from insar import sario
from insar import utils
from sardem.utils import upsample_dem_rsc
import sys
import os

if __name__ == '__main__':
    try:
        downsample = int(sys.argv[1])
        infile_list = sys.argv[2:]
    except ValueError:
        print("Usage: %s downsamplerate [infile1 [infile2 ...]]" % sys.argv[0])
        sys.exit(1)

    for infile in infile_list:
        inpath, infilename = os.path.split(infile)
        ext = utils.get_file_ext(infile)
        outfile = infilename.replace(ext, '_small' + ext)

        f = sario.load(infile, downsample=downsample)
        print("Writing %s with size %s" % (outfile, f.shape))
        sario.save(outfile, f)

    inpath, infilename = os.path.split(infile_list[0])
    in_rsc_path = os.path.join(inpath, 'elevation.dem.rsc')
    with open('elevation_small.dem.rsc', 'w') as f:
        print("Writing elevation_small.dem.rsc")
        f.write(upsample_dem_rsc(rate=(1 / downsample), rsc_filepath=in_rsc_path))
