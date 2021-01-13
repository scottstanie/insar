from collections import Counter
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def rose_plot_hours(hours,
                    ax=None,
                    density=False,
                    offset=0,
                    lab_unit="hours",
                    fill=True,
                    **param_dict):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    ax = plt.axes(polar=True) or subplot_kw=dict(projection='polar').

    Source (starting point):
    https://stackoverflow.com/questions/22562364/circular-histogram-for-python

    Also see:
    https://gist.github.com/gizmaa/7214002#polarplot
    """
    hours = np.array(hours)
    if ax is None:
        plt.figure()
        ax = plt.axes(polar=True)

    # Only need to hist if not even data (passing in times, not hours)
    # # Bin data and record counts
    # counts, bins = np.histogram(angles, bins=24)

    hours_uniq, counts = zip(*sorted(Counter(hours).items()))

    # dtheta = 360 / 24
    # ax.set_thetagrids(np.arange(0, 360 - dtheta, dtheta))

    # Note: minus pi/24 so that zero bin is centered at top
    angles = (2 * np.pi * np.array(hours_uniq) / 24) - (np.pi / 24)

    # Compute width of each bin
    # widths = np.diff(bins)
    width = 2 * np.pi / 24

    # By default plot density (frequency potentially misleading)
    # if density is None or density is True:
    #     # Area to assign each bin
    #     area = counts / angles.size
    #     # Calculate corresponding bin radius
    #     radius = (area / np.pi)**.5
    # else:
    #     radius = counts
    radius = counts

    # Plot data on ax
    # ax.bar(bins[:-1],
    ax.bar(angles,
           radius,
           zorder=1,
           align='edge',
           width=width,
           edgecolor='C0',
           fill=False,
           linewidth=1)

    # Set the direction of the zero angle
    # ax.set_theta_offset(offset)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # ax.set_thetagrids(np.arange(0, 24, 6))  # TODO: fix
    clock_times = ["%d:00" % d for d in np.arange(0, 24, 3)]
    ax.set_xticklabels(clock_times)

    # Remove ylabels, they are mostly obstructive and not informative
    ax.set_yticks([])

    # if lab_unit == "radians":
    #     label = [
    #         '$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$',
    #         r'$7\pi/4$'
    #     ]
    #     ax.set_xticklabels(label)
    # elif lab_unit == "hours":
    #     label = [0, 6, 12, 18]
    #     ax.set_xticklabels(label)
    return counts, radius, ax


def plot_time_vs_date(df):
    def _time_to_hour(t):
        return t.hour + t.minute / 60 + t.second / 3600

    dts = pd.to_datetime(df["datetime"])
    times = dts.apply(_time_to_hour)
    dates = dts.apply(lambda x: x.date())

    fig, ax = plt.subplots()
    ax.scatter(dates, times)
    ax.set_ylabel("Hour of day (UTC")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    # ax.set_yticks(y_ticks)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # -%m'))
    return times, dates
