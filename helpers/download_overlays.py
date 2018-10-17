import sys
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor


def find_replace(filename, old, new):
    with open(filename, 'r') as f:
        text = f.read()
    text = text.replace(old, new)
    with open(filename, 'w') as f:
        f.write(text)


def download(idx, overlay, look):
    look_numbered = 'quick-look-%s.png' % idx
    overlay_numbered = 'map-overlay-%s.kml' % idx
    subprocess.call("scp lidar.csr.utexas.edu:%s ./%s" % (overlay, overlay_numbered), shell=True)
    subprocess.call("scp lidar.csr.utexas.edu:%s ./%s" % (look, look_numbered), shell=True)
    find_replace(overlay_numbered, 'quick-look.png', look_numbered)
    return True


overlays = open('overlays-deduped.txt').read().splitlines()
quick_looks = [line.replace('map-overlay.kml', 'quick-look.png') for line in overlays]
with ThreadPoolExecutor(max_workers=10) as executor:
    procs = []
    for idx, overlay, look in zip(range(1, len(overlays) + 1), overlays, quick_looks):
        data = executor.submit(download, idx, overlay, look)
        procs.append(data)
    for p in procs:
        p.result()
