# coding: utf-8
from apertools import plotting, latlon
from insar import blob
import numpy as np
import sys

if __name__ == '__main__':
    try:
        path = sys.argv[1]
    except IndexError:
        path = '.'

    defo_img = latlon.load_deformation_img(path)
    sigma_list = blob.skblob.create_sigma_list(max_sigma=200, num_sigma=20)
    image_cube = blob.skblob.create_gl_cube(defo_img, max_sigma=200)
    gl_stack = np.moveaxis(image_cube, -1, 0)
    maxes = np.max(gl_stack, axis=(1, 2))
    titles = ['sigma=%d, max=%f' % (s, m) for s, m in zip(sigma_list, maxes)]
    plotting.animate_stack(
        gl_stack, save_title='blah.gif', pause_time=500, titles=titles, display=False)
