#!/usr/bin/env python

from __future__ import division  # changes '/' to normal float division
import sys

if sys.platform == "linux2":
    import matplotlib as mpl

    mpl.use("Agg")
import matplotlib.pyplot as plt
import sys
import argparse
import numpy as np


def get_cli_args(arg_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "estimator",
        choices=["TS", "TSIA", "MIDAS"],
        help="Select estimator (TS: Thiel-Sen, TSIA: Thiel-Sen Interannual)",
    )
    parser.add_argument("--infile", help="Input data file")
    parser.add_argument("--interval", type=int, help="interval (TODO)")
    parser.add_argument("--period", type=float, help="period (TODO)")
    parser.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help="Tolerance for slope estimate (TODO: what is this)",
    )
    parser.add_argument("--hist", help="Save histogram plot to this filename")
    parser.add_argument("--pr", help="Plot range (Type: integer, or 'auto'. default 4)")
    args = parser.parse_args()
    if args.estimator in ("TSIA", "MIDAS") and not args.per:
        raise ValueError(
            "ERROR: TSIA and MIDAS require a period of variation to be specified"
        )
    if args.estimator in ("TSIA", "MIDAS") and not args.per:
        raise ValueError(
            "ERROR: TSIA and MIDAS require a period of variation to be specified"
        )
    if args.estimator == "TSIA" and not args.interval:
        raise ValueError("ERROR: -TSIA requires an interval to be specified")
    return args


def main(
    estimator,
    infile=None,
    data=None,
    period=0.0,
    interval=None,
    tol=0.0,
    hist=None,
    pr=None,
):
    """Median Trend Estimators:
        Author: Ben Krichman
        Created: 02/07/2018
        Contact: bkrichman@utexas.edu

    These estimators take in x and y coordinate pairs and return an estimate of
    trend, y-intercept, standard deviation, and formal error.  When running on
    command line or from a python console, inputs are entered directly after
    the run command.  When importing and running within a python script, inputs
    are given as a single list with each input besides input data entered as a
    string.

    Examples:
        console: ./med_trend_est datafile -TS -h -pr auto
        within script: out=med_trend_est.main([data,'-TS','-h','-pr','-auto'])

    Inputs:

        Required:

            Input Data (must be first input): <filename/arrayname>
                file name of file containing x and y data as columns if running
                from console or array of x and y data as columns if running
                within a python script

            Estimator Type:
                -TS: Theil-Sen (median of slopes between all points)
                -TSIA: Theil-Sen Interannual (use only points separated by
                    some number of intervals specified by the '-per' argument)
                -MIDAS: MIDAS estimator (use only points seperaed by one
                    interval specified by the '-per' argument and iterate to
                    remove outliers, see doi:10.1002/2015JB012552)

            Period (if using -TSIA or -MIDAS): -per <period>
                specifies period of variational behavior in dataset in units of
                input file (e.g. "-per 12" for monthly data with annual
                variation)

            Interval (if using -TSIA): -int <interval/'N'>
                specifies number of intervals (of length specified by -per)
                between points selected by the TSIA estimator
                *alternately specify "-int 'N'" to include any integer multiple
                of per when selecting points

        Optional:

            Tolerance (if specifying period): -tol <tolerance>, default=0
                tolerance for near-period timespans in -TSIA or -MIDAS (e.g.
                "-tol 1" for monthly data to use spans of 11-13 months)

            Histogram: -h <name>
                plot histogram of distribution of slopes between points and
                save as "name.png"

            Plot Range (if using -h): -pr <range/'auto'>, default=4
                specify how many standard deviations to include in histogram/
                alternately specify "-pr 'auto'" to include the central 90% of
                data

    Outputs:

        Slope: median estimate of linear trend of dataset

        Intercept: median estimate of y-intercept corresponding to slope (not
            necessarily defined as optimal)

        Sigma: estimate of standard deviation derived from median absolute
            deviation under assumption of near-normal distribution
            (see doi:10.1002/2015JB012552)

        Formal Error: estimate of formal error under assumption of near-normal
            distribution and autocorrelation present in data
            (see doi:10.1002/2015JB012552)
    """

    # import data to x and y arrays
    # data = args[0]
    if data is None:
        if infile is None:
            raise ValueError("Need either infile or data")
        data = np.loadtxt(args.infile)
    x = data[:, 0]
    y = data[:, 1]

    # pull period from arguments
    if estimator in ("TSIA", "MIDAS"):
        per = period
    # if "-TSIA" in map(str, args) or "-MIDAS" in map(str, args):
    # perIND = map(str, args).index("-per")
    # per = float(args[perIND + 1])

    # pull interval from arguments
    # if "-TSIA" in map(str, args):
    # intvlIND = map(str, args).index("-int")
    # intvl = args[intvlIND + 1]
    if estimator == "TSIA":
        intvl = interval

    # pull tolerance from arguments
    # if "-tol" in map(str, args):
    #     tolIND = map(str, args).index("-tol")
    #     tol = float(args[tolIND + 1])
    # else:
    #     tol = 0

    # find all slopes and timespans
    npoints = len(x)
    slopes = np.zeros([int(npoints * (npoints - 1) / 2), 2])
    ct = 0
    for i in range(0, npoints):
        for j in range(i + 1, npoints):
            slopes[ct, 0] = (y[j] - y[i]) / (x[j] - x[i])
            slopes[ct, 1] = x[j] - x[i]
            ct = ct + 1

    # Theil-Sen (find median of slopes)
    # if "-TS" in map(str, args):
    if estimator == "TS":
        typ = "Theil-Sen"
        slope = np.median(slopes[:, 0])
        mad = np.median(np.absolute(slopes[:, 0] - slope))
        sig = 1.4826 * mad
        uncertainty = 3 * np.sqrt(np.pi / 2) * sig / np.sqrt(slopes.shape[0] / 4)
        b = np.median(y - slope * x)

    # Theil-Sen Interannual (find median of slopes spanning integer number of <period>)
    # if "-TSIA" in map(str, args):
    if estimator == "TSIA":
        typ = "Theil-Sen Interannual"
        remIND = []
        if intvl == "N":
            for k in range(0, slopes.shape[0]):
                if slopes[k, 1] % per > tol and per - slopes[k, 1] % per > tol:
                    remIND.append(k)
        else:
            intvl = float(intvl)
            for k in range(0, slopes.shape[0]):
                if slopes[k, 1] < (per * intvl - tol) or slopes[k, 1] > (
                    per * intvl + tol
                ):
                    remIND.append(k)
        slopes = np.delete(slopes, remIND, 0)
        slope = np.median(slopes[:, 0])
        mad = np.median(np.absolute(slopes[:, 0] - slope))
        sig = 1.4826 * mad
        uncertainty = 3 * np.sqrt(np.pi / 2) * sig / np.sqrt(slopes.shape[0] / 4)
        b = np.median(y - slope * x)

    # MIDAS (find median of slopes spaning <period>, remove 2 sigma outliers, and iterate)
    # if "-MIDAS" in map(str, args):
    if estimator == "MIDAS":
        typ = "MIDAS"
        remIND = []
        for k in range(0, slopes.shape[0]):
            if slopes[k, 1] < (per - tol) or slopes[k, 1] > (per + tol):
                remIND.append(k)
        slopes = np.delete(slopes, remIND, 0)
        slope = np.median(slopes[:, 0])
        mad = np.median(np.absolute(slopes[:, 0] - slope))
        sig = 1.4826 * mad
        remIND = []
        for k in range(0, slopes.shape[0]):
            if slopes[k, 0] < slope - 2 * sig or slopes[k, 0] > slope + 2 * sig:
                remIND.append(k)
        slopes2 = np.delete(slopes, remIND, 0)
        slope2 = np.median(slopes2[:, 0])
        mad2 = np.median(np.absolute(slopes2[:, 0] - slope2))
        sig2 = 1.4826 * mad2
        uncertainty = 3 * np.sqrt(np.pi / 2) * sig2 / np.sqrt(slopes2.shape[0] / 4)
        b = np.median(y - slope2 * x)

    # plot histogram
    # if "-h" in map(str, args):
    if hist:
        # find plot save name
        # onamIND = map(str, args).index("-h")
        # onam = args[onamIND + 1]
        onam = hist
        # check for and set plot range
        if not pr:
            pr = 4
        elif pr == "auto":
            # if "-pr" in map(str, args):
            # prIND = map(str, args).index("-pr")
            # pr = args[prIND + 1]
            # else:
            # pr = 4
            # if pr == "auto":
            cint = np.sort(slopes[:, 0])[: int(0.95 * len(slopes))][
                int(0.05 * len(slopes)) :
            ]
            pr = int(np.ceil((cint.max() - slope) / sig))
        else:
            pr = int(pr)
        plt.ioff()
        count, bins, patches = plt.hist(
            slopes[:, 0],
            bins=np.arange(
                slope - pr * sig,
                slope + pr * sig + 2 * pr * sig / (pr * 8),
                2 * pr * sig / (pr * 8),
            ),
            edgecolor="k",
        )
        s = "$\hat{m}$: " + str("%.3g" % slope) + "\n$\sigma$: " + str("%.3g" % sig)
        plt.text(
            slope - (pr - 0.5) * sig, count.max() - count.max() / 6, s, fontsize=12
        )
        XT = np.arange(slope - pr * sig, slope + (pr + 0.01) * sig, sig)
        XLP = np.arange(-pr, pr + 1, 1)
        XL = []
        for k in XLP:
            XL.append("$" + str(k) + "\sigma$")
        XL[pr] = "$\hat{m}$"

        if estimator == "MIDAS":
            # if "-MIDAS" in map(str, args):
            # XT=np.append(XT,slope2)
            # XL=np.append(XL,'$\hat{m}_2$')
            s2 = (
                "$\hat{m}_2$: "
                + str("%.3g" % slope2)
                + "\n$\sigma_2$: "
                + str("%.3g" % sig2)
            )
            plt.text(
                slope - (pr - 0.5) * sig, count.max() - count.max() / 3, s2, fontsize=12
            )
            plt.text(
                slope - (pr - 0.5) * sig, count.max() - count.max() / 3, s2, fontsize=12
            )
            plt.axvline(x=slope - 2 * sig, color="r", ls="--")
            plt.axvline(x=slope + 2 * sig, color="r", ls="--")
            plt.axvline(x=slope2, color="m", ls="--")
            plt.text(
                slope2 - (2 * pr * sig / 70),
                0 - count.max() / 18,
                "$\hat{s}_2$",
                fontsize=10,
                color="m",
            )
        plt.xticks(XT, XL)
        plt.tick_params(length=10)
        plt.xlim(slope - pr * sig, slope + pr * sig)
        plt.xlabel("Slope Values")
        plt.ylabel("Count")
        plt.title("Slope Distribution: " + typ)
        plt.savefig(onam + ".png", dpi=400)
        plt.close()
        plt.gcf().clear()

    # change output variables to correct values
    if estimator == "MIDAS":
        # if "-MIDAS" in map(str, args):
        slope = slope2
        sig = sig2

    # return outputs
    return slope, b, sig, uncertainty


if __name__ == "__main__":
    # pass args from CL (except script name) to script
    # args = sys.argv[1:]

    # load points from file
    # infile = args[0]
    # data = np.loadtxt(infile)
    # args[0] = data

    args = get_cli_args()
    # run estimator and print output
    slope, b, sig, uncertainty = main(args)
    print(slope, b, sig, uncertainty)
