# coding: utf-8
img1 = latlon.load_deformation_img('./N30.6W103.6/igrams')
img2 = latlon.load_deformation_img('./N31.3W103.6/igrams')
latlon.find_img_intersections(img1, img2)
latlon.find_img_intersections(img2, img1)
img1.shape
fig, axes = plt.subplots(2, 1)
axes[0].imshow(img2)
axes[1].imshow(img1)
plt.figure()
plt.imshow(img2[463:, :])
img2.shape[0] - 463
overlap2 = img2[463:, :]
overlap1 = img1[:71], :]
overlap1 = img1[:71, :]
overlap1.shape, overlap2.shape
fig2, axes2 = plt.subplots(2, 1)
axes2[0].imshow(overlap2)
axes2[1].imshow(overlap1)
fig2, axes2 = plt.subplots(2, 1)
fig2, axes2 = plt.subplots(3, 1)
axes2[0].imshow(overlap2)
axes2[1].imshow(overlap1)
axes2[2].imshow(overlap1-overlap2)
plt.figure()
plt.imshow(overlap1-overlap2)
plt.colorbar()
plt.colormap(plotting.DISCRETE_SEISMIC5)
plt.clf()
plt.imshow(overlap1-overlap2, cmap='bwr')
plt.colorbar()
np.mean(overlap1-overlap2)
