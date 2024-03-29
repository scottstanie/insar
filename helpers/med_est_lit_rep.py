#!/usr/bin/env python

from __future__ import division
import med_trend_est as MTE
import numpy as np
import matplotlib.pyplot as plt
import copy

# Repeat Literature Example
time = np.arange(0, 85, 2)
dat = (10 / 12) * time + 2 * np.sin(2 * np.pi / 12 * time)
truth = copy.copy(dat)
dat[18:] = dat[18:] + 10
dat[30:] = dat[30:] + 10
noise = 5 * np.rand(len(time)) - 2.5
ndat = dat + noise
ntruth = truth + noise
indat = np.column_stack((time, ndat))

A = np.vstack([time, np.ones(len(time))]).T
LSout = np.linalg.lstsq(A, ndat)[0]
# LSerr=np.real(np.sqrt(np.sum(np.power(ndat-LSout[0]*time,2))/(len(time)-2)/np.sum(np.power(time-np.mean(time),2))))
LSerr = np.std(ndat - LSout[0] * time) / np.sqrt(len(time))

B = np.vstack([time, np.zeros(len(time))]).T
LS0out = np.linalg.lstsq(B, ndat)[0]
# LS0err=np.real(np.sqrt(np.sum(np.power(ndat-LS0out[0]*time,2))/(len(time)-2)/np.sum(np.power(time-np.mean(time),2))))
LS0err = np.std(ndat - LS0out[0] * time) / np.sqrt(len(time))

TSIAout = MTE.main([indat, "-TSIA", "-h", "LitRep.TSIA", "-per", "12", "-int", str(1)])
MIDASout = MTE.main([indat, "-MIDAS", "-h", "LitRep.MIDAS", "-per", "12"])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
TRUTH = ax.plot(time, (10 / 12) * time, "-", color=(0.6, 0.6, 0.6))
LS = ax.plot(time, LSout[0] * time + LSout[1], "-", color=(0.8, 0, 0))
LS0 = ax.plot(time, LS0out[0] * time + LS0out[1], "--", color=(0.8, 0, 0))
TSIA = ax.plot(time, TSIAout[0] * time, "-", color="b")
MIDAS = ax.plot(time, MIDASout[0] * time, "-", color="g")
DATNS = ax.plot(time, ntruth, "x", markersize=5, linewidth=0.1, color=(0.6, 0.6, 0.6))
DAT = ax.plot(time, ndat, "x", markersize=5, linewidth=0.1, color="k")
leg = ax.legend(
    [DATNS[0], DAT[0], LS[0], LS0[0], TSIA[0], MIDAS[0], TRUTH[0]],
    [
        "Data (No Steps)",
        "Data     $\hat{m}$        $s_{\hat{m}}$ (mm/yr)",
        "LS:       "
        + str("%.2f" % (LSout[0] * 12))
        + "  "
        + str("%.3f" % (12 * LSerr)),
        "LSNI:    "
        + str("%.2f" % (LS0out[0] * 12))
        + "  "
        + str("%.3f" % (12 * LS0err)),
        "TSIA:    "
        + str("%.2f" % (TSIAout[0] * 12))
        + "  "
        + str("%.3f" % (12 * TSIAout[3])),
        "MIDAS: "
        + str("%.2f" % (MIDASout[0] * 12))
        + "  "
        + str("%.3f" % (12 * MIDASout[3])),
        "Truth:   " + str("%.2f" % 10),
    ],
)
leg.get_frame().set_alpha(1)
ax.grid("on")
plt.ylim(-5, 110)
XT = np.arange(0, 8, 1)
plt.xticks(XT * 12, XT)
plt.xlabel("Time (years)")
plt.ylabel("Position (mm)")
plt.title(
    "Linear Trend (10mm/yr) + Annual Sinusoid (2mm/yr)\n + Noise ($\sigma$=1.5mm) + Steps at 3 & 5 Years"
)
plt.savefig("LitRep.png", dpi=400)
