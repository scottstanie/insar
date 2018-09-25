# coding: utf-8
defo = np.load('deformation.npy')
defo_img = latlon.LatlonImage(data=np.mean(defo[-3:], axis=0), dem_rsc_file='dem.rsc')
plt.imshow(defo_img)
timeseries = defo[:, 146, 250]
plt.plot(timeseries)
geolist = np.load('geolist.npy')
geolsit
geolsit
geolist
plt.plot(geolist, timeline, marker='o', linestyle='dashed', linewidth=1, markersize=4)
timeline = timeseries
plt.plot(geolist, timeline, marker='o', linestyle='dashed', linewidth=1, markersize=4)
plt.plot(geolist, timeline, marker='o', linestyle='dashed', linewidth=2, markersize=4)
plt.plot(geolist, timeline, marker='x', linestyle='dashed', linewidth=2, markersize=6)
plt.plot(geolist, timeline, marker='o', linestyle='dashed', linewidth=2, markersize=6)
row = 146, col=250
row, col=146, 250
lat, lon = img.rowcol_to_latlon(row, col)
lat, lon = defo_img.rowcol_to_latlon(row, col)
legend = ['Lat {:.3f}, Lon {:.3f}'.format(lat, lon)]
legend
plt.xlabel('SAR Image Date')
plt.ylabel('Centimeters of uplift')
plt.title('Deformation timeseries')
get_ipython().run_line_magic('pinfo', 'plt.savefig')
plt.savefig('defo_timeseries.png', bbox_inches='tight', transparent=True, dpi=300)
