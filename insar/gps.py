import os
import glob
import datetime
import functools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from insar import los, timeseries, latlon, kml

GPS_DIR = '/data1/scott/pecos/gps_station_data'


@functools.lru_cache()
def read_station_df(filename=os.path.join(GPS_DIR, 'texas_stations.csv'), header=None):
    df = pd.read_csv(filename, header=header)
    df.columns = ['name', 'lat', 'lon', 'alt']
    return df


def station_lonlat(station_name):
    df = read_station_df()
    name, lat, lon, alt = df[df['name'] == station_name].iloc[0]
    return lon, lat


def stations_within_image(image_ll):
    """Given a LatlonImage, find gps stations contained in area"""
    df = read_station_df()
    station_lon_lat_arr = df[['lon', 'lat']].values
    contains_bools = image_ll.contains(station_lon_lat_arr)
    return df[contains_bools][['name', 'lon', 'lat']].values


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


def _clean_gps_df(df, start_year):
    df['dt'] = pd.to_datetime(df['YYMMMDD'], format='%y%b%d')

    df_recent = df[df['dt'] >= datetime.datetime(start_year, 1, 1)]
    df_enu = df_recent[['dt', '__east(m)', '_north(m)', '____up(m)']]
    df_enu = df_enu.rename(mapper=lambda s: s.replace('_', '').replace('(m)', ''), axis='columns')
    return df_enu


def load_gps_station_df(station_name, basedir=GPS_DIR, start_year=2015):
    """Loads one gps station file's data of ENU displacement since start_year"""
    gps_data_file = os.path.join(basedir, '%s.NA12.tenv3' % station_name)
    df = pd.read_csv(gps_data_file, header=0, sep='\s+')
    return _clean_gps_df(df, start_year)


def plot_gps_enu(station=None, station_df=None, days_smooth=12):
    def remove_xticks(ax):
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)

    if station is not None:
        station_df = load_gps_station_df(station)

    fig, axes = plt.subplots(3, 1)
    east_mm = 100 * (station_df['east'] - station_df['east'].iloc[0])
    north_mm = 100 * (station_df['north'] - station_df['north'].iloc[0])
    up_mm = 100 * (station_df['up'] - station_df['up'].iloc[0])
    dts = station_df['dt']
    axes[0].plot(dts, east_mm, 'b.')
    axes[0].set_ylabel('east (cm)')
    axes[0].plot(dts, timeseries.moving_average(east_mm, days_smooth), 'r-')
    axes[0].grid(True)
    remove_xticks(axes[0])

    axes[1].plot(dts, north_mm, 'b.')
    axes[1].set_ylabel('north (cm)')
    axes[1].plot(dts, timeseries.moving_average(north_mm, days_smooth), 'r-')
    axes[1].grid(True)
    remove_xticks(axes[1])

    axes[2].plot(dts, up_mm, 'b.')
    axes[2].set_ylabel('up (cm)')
    axes[2].plot(dts, timeseries.moving_average(up_mm, days_smooth), 'r-')
    axes[2].grid(True)
    remove_xticks(axes[2])

    return fig, axes


def load_gps_los_data(insar_dir, station_name=None, to_cm=True, zero_start=True):
    """Load the GPS timeseries of a station name

    Returns the timeseries, and the datetimes of the points
    """
    lon, lat = station_lonlat(station_name)
    enu_coeffs = los.find_enu_coeffs(lon, lat, insar_dir)

    df = load_gps_station_df(station_name)
    enu_data = df[['east', 'north', 'up']].T
    los_gps_data = los.project_enu_to_los(enu_data, enu_coeffs=enu_coeffs)

    if to_cm:
        print("Converting GPS to cm:")
        los_gps_data = 100 * los_gps_data

    if zero_start:
        print("Resetting GPS data start to 0")
        los_gps_data = los_gps_data - np.mean(los_gps_data[:100])
    return df['dt'], los_gps_data


def difference_gps_stations(station1, station2):
    df1 = load_gps_station_df(station1)
    df2 = load_gps_station_df(station2)
    merged = pd.merge(df1, df2, how='inner', on=['dt'], suffixes=('_1', '_2'))
    merged['d_east'] = merged['east_1'] - merged['east_2']
    merged['d_north'] = merged['north_1'] - merged['north_2']
    merged['d_up'] = merged['up_1'] - merged['up_2']
    return merged
    # return merged[['dt', 'd_east', 'd_north', 'd_up']]


def find_insar_ts(insar_dir, station_name, defo_name='deformation.npy'):
    """Get the insar timeseries closest to a GPS station

    Returns the timeseries, and the datetimes of points for plotting
    """
    igrams_dir = os.path.join(insar_dir, 'igrams')
    geolist, deformation_stack = timeseries.load_deformation(igrams_dir, filename=defo_name)
    defo_img = latlon.load_deformation_img(igrams_dir, filename=defo_name)

    lon, lat = station_lonlat(station_name)
    insar_row, insar_col = defo_img.nearest_pixel(lat=lat, lon=lon)
    # import pdb
    # pdb.set_trace()
    insar_ts = timeseries.window_stack(
        deformation_stack, insar_row, insar_col, window_size=5, func=np.mean)
    return geolist, insar_ts


def plot_gps_vs_insar():
    insar_dir = '/data1/scott/pecos/path85/N31.4W103.7'

    # station_name = 'TXKM'
    # los_dir = '/data4/scott/delaware-basin/test2/N31.4W103.7/'
    # lla, xyz = los.read_los_output(os.path.join(los_dir, 'extra_files/los_vectors.txt'))
    # enu_coeffs = los.find_enu_coeffs(-102.894010019, 31.557733084, insar_dir)

    # lon, lat = station_lonlat(station_name)
    # df = load_gps_station_df(station_name)
    # enu_data = df[['east', 'north', 'up']].T
    # los_gps_data = project_enu_to_los(enu_data, los_vec, lat, lon)
    # los_gps_data = los.project_enu_to_los(enu_data, enu_coeffs=enu_coeffs)

    defo_name = 'deformation.npy'
    igrams_dir = os.path.join(insar_dir, 'igrams')
    defo_img = latlon.load_deformation_img(igrams_dir, filename=defo_name)
    station_list = stations_within_image(defo_img)
    for station_name, lon, lat in station_list:
        plt.figure()

        gps_dts, los_gps_data = load_gps_los_data(insar_dir, station_name)

        plt.plot(gps_dts, los_gps_data, 'b.', label='gps data: %s' % station_name)

        days_smooth = 60
        los_gps_data_smooth = timeseries.moving_average(los_gps_data, days_smooth)
        plt.plot(
            gps_dts,
            los_gps_data_smooth,
            'b',
            linewidth='4',
            label='%d day smoothed gps data: %s' % (days_smooth, station_name))

        geolist, insar_ts = find_insar_ts(insar_dir, station_name, defo_name=defo_name)

        plt.plot(geolist, insar_ts, 'rx', label='insar data', ms=5)

        days_smooth = 5
        insar_ts_smooth = timeseries.moving_average(insar_ts, days_smooth)
        plt.plot(
            geolist, insar_ts_smooth, 'r', label='%s day smoothed insar' % days_smooth, linewidth=3)

        plt.legend()
    # return geolist, insar_ts, gps_dts, los_gps_data, defo_img


def plot_gps_vs_insar_diff(fignum=None, defo_name='deformation.npy'):
    insar_dir = '/data1/scott/pecos/path85/N31.4W103.7'

    igrams_dir = os.path.join(insar_dir, 'igrams')
    defo_img = latlon.load_deformation_img(igrams_dir, filename=defo_name)
    station_list = stations_within_image(defo_img)

    stat1, lon1, lat1 = station_list[0]
    stat2, lon2, lat2 = station_list[1]

    # # First way: project each ENU to LOS, then subtract
    gps_dts1, los_gps_data1 = load_gps_los_data(insar_dir, stat1, zero_start=False)
    gps_dts2, los_gps_data2 = load_gps_los_data(insar_dir, stat2, zero_start=False)
    print('first gps entries:')
    print(los_gps_data1[0])
    print(los_gps_data2[0])

    df1 = gps_dts1.to_frame()
    df1['gps1'] = los_gps_data1
    df2 = gps_dts2.to_frame()
    df2['gps2'] = los_gps_data2
    diff_df = pd.merge(df1, df2, how='inner', on=['dt'])
    # diff_df['d_gps'] = diff_df['gps1'] - diff_df['gps2']
    gps_diff_ts = diff_df['gps1'] - diff_df['gps2']
    gps_diff_ts = gps_diff_ts - np.mean(gps_diff_ts[:100])
    gps_dts = diff_df['dt']

    # # Other way: subtracting ENU components, then project diffs to LOS
    # enu_coeffs = los.find_enu_coeffs(lon1, lat1, insar_dir)
    # # Check how different they are
    # enu_coeffs2 = los.find_enu_coeffs(lon2, lat2, insar_dir)
    # import pdb
    # pdb.set_trace()
    # diff_df = difference_gps_stations(stat1, stat2)
    # enu_data = diff_df[['d_east', 'd_north', 'd_up']].T
    # gps_diff_ts = los.project_enu_to_los(enu_data, enu_coeffs=enu_coeffs)
    # gps_dts = diff_df['dt']

    print("Converting GPS to cm:")
    gps_diff_ts = 100 * gps_diff_ts

    start_mean = np.mean(gps_diff_ts[:100])
    # start_mean = np.mean(los_gps_data2[:100])
    print("Resetting GPS data start to 0 of station 2, subtract %f" % start_mean)
    gps_diff_ts = gps_diff_ts - start_mean

    plt.figure(fignum)
    plt.plot(gps_dts, gps_diff_ts, 'b.', label='gps data: %s' % stat1)

    days_smooth = 60
    los_gps_data_smooth = timeseries.moving_average(gps_diff_ts, days_smooth)
    plt.plot(
        gps_dts,
        los_gps_data_smooth,
        'b',
        linewidth='4',
        # label='%d day smoothed gps data: %s' % (days_smooth, station_name))
        label='%d day smoothed gps data: %s-%s' % (days_smooth, stat1, stat2))

    geolist, insar_ts1 = find_insar_ts(insar_dir, stat1, defo_name=defo_name)
    _, insar_ts2 = find_insar_ts(insar_dir, stat2, defo_name=defo_name)
    insar_diff = insar_ts1 - insar_ts2

    plt.plot(geolist, insar_diff, 'rx', label='insar data difference', ms=5)

    days_smooth = 5
    insar_diff_smooth = timeseries.moving_average(insar_diff, days_smooth)
    plt.plot(
        geolist,
        insar_diff_smooth,
        'r',
        label='linear %s day smoothed insar' % days_smooth,
        linewidth=3)

    plt.ylim([-2, 2])
    plt.ylabel('cm of cumulative LOS displacement')
    plt.legend()
    return los_gps_data1, los_gps_data2, gps_diff_ts, insar_ts1, insar_ts2, insar_diff


def save_station_points_kml(station_iter):
    for name, lat, lon, alt in station_iter:
        kml.create_kml(
            title=name,
            desc='GPS station location',
            lon_lat=(lon, lat),
            kml_out='%s.kml' % name,
            shape='point',
        )


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

# def gps_to_los():
#     insar_dir = '/data4/scott/delaware-basin/test2/N31.4W103.7'
#     lla, xyz = los.read_los_output(os.path.join(insar_dir, 'extra_files/los_vectors.txt'))
#
#     los_vec = np.array(xyz)[0]
#
#     station_name = 'TXKM'
#     df = load_gps_station_df(station_name)
#     lon, lat = station_lonlat(station_name)
#     enu_data = df[['east', 'north', 'up']].T
#     los_gps_data = los.project_enu_to_los(enu_data, los_vec, lat, lon)
#     return los_gps_data, df['dt']
