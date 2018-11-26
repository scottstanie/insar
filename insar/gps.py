import os
import glob
import datetime
import functools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from insar import los, timeseries, latlon

GPS_DIR = '/data1/scott/pecos/gps_station_data'


@functools.lru_cache()
def read_station_df(filename='/data1/scott/pecos/gps_station_data/texas_stations.csv', cache=True):
    df = pd.read_csv(filename, header=None)
    df.columns = ['name', 'lat', 'lon', 'alt']
    return df


def station_lonlat(station_name):
    df = read_station_df()
    name, lat, lon, alt = df[df['name'] == station_name].iloc[0]
    return lon, lat


def stations_within_image(image_ll):
    df = read_station_df()
    station_lon_lat_arr = df[['lon', 'lat']].values
    contains_bools = image_ll.contains(station_lon_lat_arr)
    return df[contains_bools].values


def find_stations_with_data(gps_dir=None):
    # Now also get gps station list
    if not gps_dir:
        gps_dir = GPS_DIR

    all_station_data = read_station_df()
    station_data_list = find_station_data_files(gps_dir)
    stations_with_data = [
        tup for tup in all_station_data.to_records(index=False) if tup[0] in station_data_list
    ]
    return stations_with_data


def find_station_data_files(gps_dir):
    station_files = glob.glob(os.path.join(gps_dir, '*.tenv3'))
    station_list = []
    for filename in station_files:
        _, name = os.path.split(filename)
        station_list.append(name.split('.')[0])
    return station_list


def load_gps_station(station_name, basedir=GPS_DIR):
    """Loads one gps station file's data of ENU displacement"""
    gps_data_file = os.path.join(basedir, '%s.NA12.tenv3' % station_name)
    df = pd.read_csv(gps_data_file, header=0, sep='\s+')
    df['dt'] = pd.to_datetime(df['YYMMMDD'], format='%y%b%d')

    df2015 = df[df['dt'] > datetime.datetime(2014, 12, 31)]
    df_enu = df2015[['dt', '__east(m)', '_north(m)', '____up(m)']]
    df_enu = df_enu.rename(mapper=lambda s: s.replace('_', '').replace('(m)', ''), axis='columns')
    return df_enu


def gps_to_los():
    insar_dir = '/data4/scott/delaware-basin/test2/N31.4W103.7'
    lla, xyz = los.read_los_output(os.path.join(insar_dir, 'extra_files/los_vectors.txt'))

    los_vec = np.array(xyz)[0]

    station_name = 'TXKM'
    df = load_gps_station(station_name)
    lon, lat = station_lonlat(station_name)
    enu_data = df[['east', 'north', 'up']].T
    los_gps_data = los.project_enu_to_los(enu_data, los_vec, lat, lon)
    return los_gps_data, df['dt']


def plot_gps_vs_insar():
    # WRONG DIR
    # insar_dir = '/data4/scott/delaware-basin/test2/N31.4W103.7'
    insar_dir = '/data1/scott/pecos/path85/N31.4W103.7'
    los_dir = '/data4/scott/delaware-basin/test2/N31.4W103.7/'
    enu_coeffs = los.find_enu_coeffs(-102.894010019, 31.557733084, los_dir)

    lla, xyz = los.read_los_output(os.path.join(los_dir, 'extra_files/los_vectors.txt'))
    station_name = 'TXKM'
    df = load_gps_station(station_name)
    lon, lat = station_lonlat(station_name)

    enu_data = df[['east', 'north', 'up']].T
    gps_dts = df['dt']
    # los_gps_data = project_enu_to_los(enu_data, los_vec, lat, lon)
    los_gps_data = los.project_enu_to_los(enu_data, enu_coeffs=enu_coeffs)
    print('Resetting GPS data start to 0, converting to cm:')
    los_gps_data = 100 * (los_gps_data - np.mean(los_gps_data[0:100]))
    plt.plot(gps_dts, los_gps_data, 'b.', label='gps data: %s' % station_name)

    days_smooth = 60
    los_gps_data_smooth = timeseries.moving_average(los_gps_data, days_smooth)
    plt.plot(
        gps_dts,
        los_gps_data_smooth,
        'b',
        linewidth='4',
        label='%d day smoothed gps data: %s' % (days_smooth, station_name))

    igrams_dir = os.path.join(insar_dir, 'igrams')
    defo_name = 'deformation.npy'
    geolist, deformation = timeseries.load_deformation(igrams_dir, filename=defo_name)
    defo_ll = latlon.LatlonImage(data=deformation, dem_rsc_file=os.path.join(igrams_dir, 'dem.rsc'))

    print('lon', lon, 'lat', lat, type(lat))
    print(latlon.grid_corners(**defo_ll.dem_rsc))
    # import pdb
    # pdb.set_trace()
    insar_row, insar_col = defo_ll.nearest_pixel(lat=lat, lon=lon)
    print('insar row')
    print(insar_row)
    print(insar_col)
    insar_ts = timeseries.window_stack(defo_ll, insar_row, insar_col, window_size=5, func=np.mean)

    plt.plot(geolist, insar_ts, 'rx', label='insar data', ms=5)

    days_smooth = 5
    insar_ts_smooth = timeseries.moving_average(insar_ts, days_smooth)
    plt.plot(
        geolist, insar_ts_smooth, 'r', label='%s day smoothed insar' % days_smooth, linewidth=3)

    plt.legend()
    # return geolist, insar_ts, gps_dts, los_gps_data, defo_ll
    return geolist, insar_ts, gps_dts, los_gps_data, defo_ll


# def read_station_dict(filename):
#     """Reads in GPS station data"""
#     with open(filename) as f:
#         station_strings = [row for row in f.read().splitlines()]
#
#     all_station_data = []
#     for row in station_strings:
#         name, lat, lon, _ = row.split(',')  # Ignore altitude
#         all_station_data.append((name, float(lon), float(lat)))
#     return all_station_data
