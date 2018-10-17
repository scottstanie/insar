import json
from shapely.geometry import shape, Point
from collections import defaultdict
from insar import kml

geolist_files = open('geojson_list.txt').read().splitlines()
gjs = {num: json.load(open(gjf)) for num, gjf in enumerate(geolist_files)}

station_strings = [row for row in open('texas_stations.csv').read().splitlines()]
station_strings[:4]
gps_list = []
for row in station_strings:
    name, lat, lon, _ = row.split(',')
    gps_list.append((name, float(lat), float(lon)))

ps = [tup[1:] for tup in gps_list]
points = [Point(*p) for p in ps]

name_to_blocknum = defaultdict(list)
covered_stations = set()
blocks_with_gps = set()
for name, lat, lon in gps_list:
    p = Point(lon, lat)
    for num, gj in gjs.items():
        shape_gj = shape(gj)
        if shape_gj.contains(p):
            name_to_blocknum[name].append(num)
            covered_stations.add((name, lat, lon))
            blocks_with_gps.add(num)

# Now reverse the name_to_blocknum dict dict
blocknum_to_name = defaultdict(list)
for name, idlist in name_to_blocknum.items():
    for num in idlist:
        blocknum_to_name[num].append(name)

print(name_to_blocknum)
print("==========" * 10)

print("%s total geojson blocks" % len(gjs))
print('%s distinct gps stations within some block:' % len(covered_stations))
print(covered_stations)
print('%s blocks with gps:' % len(blocks_with_gps))
print(blocks_with_gps)

print("==========" * 10)
print(blocknum_to_name)

for name, lat, lon in gps_list:
    kml.create_kml(
        lon_lat=(lon, lat),
        title=name,
        desc="GPS station %s" % name,
        shape='point',
        kml_out='%s.kml' % name,
    )
