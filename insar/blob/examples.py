# coding: utf-8
sigma_list = blob.skblob._create_sigma_list(max_sigma=200, num_sigma=20)
image_cube = blob.skblob._create_gl_cube(defo_img, max_sigma=200)
gl_stack = np.moveaxis(image_cube, -1, 0)
maxes = np.max(gl_stack, axis=(1,2))
titles = ['sigma=%d, max=%f' % (s, m) for s,m in zip(sigma_list, maxes)]
plotting.animate_stack(gl_stack, save_title='blah.gif', pause_time=500, titles=titles, display=False)
