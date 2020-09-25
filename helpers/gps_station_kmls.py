import sys
from apertools import kml, gps, sario

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python %s [rsc_filename]" % sys.argv[0], file=sys.stderr)
        print("or: python %s [stackfile.h5]" % sys.argv[0], file=sys.stderr)
        sys.exit(1)

    if sys.argv[1].endswith(".h5"):
        rsc_data = sario.load_dem_from_h5(h5file=sys.argv[1])
        station_list = gps.stations_within_rsc(rsc_data=rsc_data)
    else:
        station_list = gps.stations_within_rsc(rsc_filename=sys.argv[1])

    print("Station list:")
    print(station_list)
    for name, lon, lat in station_list:
        kml.create_kml(
            lon_lat=(lon, lat),
            title=name,
            desc="GPS station %s" % name,
            shape="point",
            kml_out="%s.kml" % name,
        )
