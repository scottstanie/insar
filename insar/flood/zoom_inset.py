# coding: utf-8
import os
import stack
import glob

# hhfiles = glob.glob("/home/scott/Documents/Learning/research/uav-sar-data/grd/*14937*HHHH*.grd")
# hhfiles = glob.glob(os.path.expanduser("~/uav-sar-data/grd/*14937*HHHH*.grd"))
hhfiles = glob.glob(os.path.expanduser("~/uav-sar-data/stack-grd/*.grd"))
print(hhfiles)
stack.demo_zoom(hhfiles)
