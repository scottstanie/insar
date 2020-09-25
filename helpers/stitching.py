from apertools import sario
import glob
import os
import numpy as np


def stitchtop(full, top):
    out = np.zeros((full.shape[0] + 600, full.shape[1]), dtype=full.dtype)
    out[-full.shape[0] :, :] = full
    out[: top.shape[0], 600:] = top
    return out


def stitchtop_bool(full, top):
    out = np.ones((full.shape[0] + 600, full.shape[1]), dtype=full.dtype)
    out[-full.shape[0] :, :] = full
    out[: top.shape[0], 600:] = top
    return out


def loop(ext):
    for topf in glob.glob("*" + ext):
        if os.path.exists("../../stitched/igrams/{}".format(topf)):
            print("skip", topf)
            continue

        try:
            top = sario.load(topf, return_amp=True)
            rest = sario.load("../../allpath85/igrams/{}".format(topf), return_amp=True)
        except FileNotFoundError:
            continue
        sario.save("../../stitched/igrams/{}".format(topf), stitchtop(rest, top))
        print("saved", topf)


def loop_cc():
    for topf in glob.glob("*cc"):
        if os.path.exists("../../stitched/igrams/{}".format(topf)):
            print("skip", topf)
            continue

        try:
            amptop, top = sario.load(topf, return_amp=True)
            amprest, rest = sario.load(
                "../../allpath85/igrams/{}".format(topf), return_amp=True
            )
        except FileNotFoundError:
            continue
        sario.save(
            "../../stitched/igrams/{}".format(topf),
            np.stack((stitchtop(amprest, amptop), stitchtop(rest, top)), axis=0),
        )
        print("saved", topf)


def loop_bool(ext=".geo.mask"):
    for topf in glob.glob("*" + ext):
        if os.path.exists("../../stitched/igrams/{}".format(topf)):
            print("skip", topf)
            continue

        try:
            top = sario.load(topf, return_amp=True)
            rest = sario.load("../../allpath85/igrams/{}".format(topf), return_amp=True)
        except FileNotFoundError:
            continue
        sario.save("../../stitched/igrams/{}".format(topf), stitchtop_bool(rest, top))
        print("saved", topf)


def loop_geo(ext=".geo"):
    for topf in glob.glob("*" + ext):
        if os.path.exists("../stitched/{}".format(topf)):
            print("skip", topf)
            continue

        try:
            top = sario.load(topf, return_amp=True)
            rest = sario.load("../allpath85/{}".format(topf))
        except FileNotFoundError:
            continue
        sario.save("../stitched/{}".format(topf), stitchtop(rest, top))
        print("saved", topf)
