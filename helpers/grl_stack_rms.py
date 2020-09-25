# import h5py
import apertools.gps as gps

# import numpy as np
# import pandas as pd

station_name_list78 = [
    "NMHB",
    "TXAD",
    "TXBG",
    "TXBL",
    "TXCE",
    "TXFS",
    # "TXKM",
    "TXL2",
    "TXMC",
    "TXMH",
    "TXOE",
    "TXOZ",
    "TXS3",
    "TXSO",
]


def load_dfs():
    filenames = [
        "stack_2017_max800.h5",
        "stack_2017_max800_pruned.h5",
        "stack_2017_max800_alpha100.h5",
        "stack_2017_max800_alpha100_pruned.h5",
    ]
    dfs = []
    for fn in filenames:
        # with h5py.File(fn) as hf:
        # stacks.append(hf["stack/1"][:])
        df = gps.create_insar_gps_df(
            defo_filename=fn, station_name_list=station_name_list78
        )
        dfs.append(df)
    return dfs


def find_rms(df):
    rms_arr = []
    max_arr = []
    df.dropna(inplace=True)
    for s in station_name_list78:
        rms_arr.append(gps.rms(df[f"{s}_diff"]))
        max_arr.append(df[f"{s}_diff"].abs().max())
    return rms_arr, max_arr
    # list(zip(station_name_list78, rms_arr, max_arr))


def find_all_errors():
    # dfs = load_dfs()
    filenames = [
        "stack_2017_max800.h5",
        "stack_2017_max800_pruned.h5",
        "stack_2017_max800_alpha100.h5",
        "stack_2017_max800_alpha100_pruned.h5",
    ]
    dfs = []
    for fn in filenames:
        # with h5py.File(fn) as hf:
        # stacks.append(hf["stack/1"][:])
        df = gps.create_insar_gps_df(
            defo_filename=fn, station_name_list=station_name_list78
        )
        dfs.append(df)
    total_rms, total_max = [], []
    for df in dfs:
        rms_arr, max_arr = find_rms(df)
        total_rms.append(rms_arr)
        total_max.append(max_arr)
    # return gps.rms(total_rms), max(total_max)
    return filenames, total_rms, total_max
