# ----------------------------------------------------------- #
# 1. Interplate wrfout* to pressure levels of IASI;
# 2. Regrid wrfout* to higher resolution;
# 3. Filter data (circles) in Polygon;
# 4. Average wrf_data (grids) in circles;
# 5. Filter fill_value of Cx and Cross-boundary levels of WRF-Chem
# 6. Complement latitude filtered out by IASI;
# 7. Draw profile;
# ----------------------------------------------------------- #

import h5py
import string
import pyresample
import matplotlib
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
from numpy.linalg import inv
from pyresample import bilinear
# print (matplotlib.get_backend())
matplotlib.use('qt5agg')
import matplotlib.ticker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Polygon,Point
from wrf import getvar, interplevel, to_np, get_basemap, latlon_coords

# ----------------------------------------------------------- #
# Parameters
# wrfout_file: path of wrfout_file
# IASI_file: path of IASI_file
# t: timeindex in wrfout_file
# saveplace,savename
# resolution: change the resolution of wrfout* 
# Lon_min,Lon_max,Lat_min,Lat_max: crop wrfout*
# poly_1: first Polygon
# p_max: lon_max of Polygon
# title_IASI,title_WRF
# ----------------------------------------------------------- #

wrfout_file = '/data/home/wang/zx/model/chem3.9.1/WRFV3/\
test/em_real/history/lightning/BEHR/hk/2017/CFSR_nogwd/\
wrfout_d03_2017-04-19_18:00:00'

IASI_file = '/data/home/wang/zx/data/IASI/GD_files/\
IASI-MetopA_L2_O3_SOFRID-v3.5_2017042000.he5'

saveplace = '/data/home/wang/zx/codes/interplation/'
savename = 'IASI_Profiles.eps'

t = 8
resolution = 0.05
Lon_min = 109; Lon_max =118
Lat_min = 20; Lat_max = 26

# set slice
poly_1  = np.array([[ 111.3,21.5 ],
                [ 112.25,  25.  ],
                [ 112.45,  25.  ],
                [ 111.5,   21.5 ]])
p_max  = 115
delta =[0.2,0.35]


title_IASI = 'IASI Profile'
title_WRF = 'WRF-Chem Profile'

# ----------------------------------------------------------- #

# Get variables of wrfout*
fc = Dataset(wrfout_file)
lat = getvar(fc, 'XLAT',timeidx=8)
lon = getvar(fc, 'XLONG',timeidx=8)
lat_curv = to_np(lat)
lon_curv = to_np(lon)
p        = getvar(fc, 'pressure',timeidx=8)
o3       = getvar(fc, 'o3',timeidx=8)
fc.close()

# Get variables of IASI
IASI = h5py.File (IASI_file,'r')
lat   = IASI['HDFEOS/SWATHS/O3/Geolocation Fields/Latitude'][:]
lon   = IASI['HDFEOS/SWATHS/O3/Geolocation Fields/Longitude'][:]
fill_value = IASI['HDFEOS/SWATHS/O3/Data Fields/O3 Retrieval Error Covariance'].attrs['_FillValue']
# The retrieval error covariance matrix
Cx1 = IASI['HDFEOS/SWATHS/O3/Data Fields/O3 Retrieval Error Covariance'][:]
# The a priori covariance matrix
Ca1 = IASI['HDFEOS/SWATHS/O3/Data Fields/O3 Apriori Error Covariance'][:]
# Retrieval O3 profile
O3 = IASI['HDFEOS/SWATHS/O3/Data Fields/Ozone'][:]
# a priori profile
Xa = IASI['HDFEOS/SWATHS/O3/Data Fields/O3 Apriori']

# Set IASI pressure levels
plevel = np.array([1.00000000e1, 2.90000000e1, 6.90000000e1, 1.42000000e2,\
   2.61100006e2,   4.40700012e2,   6.95000000e2,   1.03700000e3,\
   1.48100000e3,   2.04000000e3,   2.72600000e3,   3.55100000e3,\
   4.52900000e3,   5.67300000e3,   6.99700000e3,   8.51800000e3,\
   1.02050000e4,   1.22040000e4,   1.43840000e4,   1.67950000e4,\
   1.94360000e4,   2.22940000e4,   2.53710000e4,   2.86600000e4,\
   3.21500000e4,   3.58280000e4,   3.96810000e4,   4.36950000e4,\
   4.78540000e4,   5.21460000e4,   5.65540000e4,   6.10600000e4,\
   6.56430000e4,   7.02730000e4,   7.49120000e4,   7.95090000e4,\
   8.39950000e4,   8.82800000e4,   9.22460000e4,   9.57440000e4,\
   9.85880000e4,   1.00543000e5,   1.01325000e5])/100

# limit IASI
mask = (lon > Lon_min) & (lon < Lon_max) & (lat > Lat_min) & (lat < Lat_max)
lon_mask = lon[mask]; lat_mask = lat[mask]
O3_mask = O3[:,mask]
Cx1 = Cx1[:,mask]
Xa = Xa[:,mask]

# Set the resolution and range of target grid
lon_new = np.arange(Lon_min,Lon_max,resolution)
lat_new = np.arange(Lat_min,Lat_max,resolution)
lon2d, lat2d = np.meshgrid(lon_new,lat_new)

orig_def = pyresample.geometry.SwathDefinition(lons=lon_curv, lats=lat_curv)
targ_def = pyresample.geometry.SwathDefinition(lons=lon2d, lats=lat2d)

# interplate to pressure level and regrid to resolution
o3_p = np.zeros((43,lat_curv.shape[0],lon_curv.shape[1]))
o3_nearest = np.zeros((43,lat_new.shape[0],lon_new.shape[0]))
# o3_gauss = np.zeros((43,lat_new.shape[0],lon_new.shape[0]))

l = 0
for level in plevel:
  o3_p[l] = interplevel(o3, p, level)
  o3_nearest[l] = pyresample.kd_tree.resample_nearest(orig_def, to_np(o3_p[l]), \
          targ_def, radius_of_influence=500000, fill_value=None)
  # o3_gauss[l] = pyresample.kd_tree.resample_gauss(orig_def, to_np(o3_p[l]), \
  #         targ_def, radius_of_influence=500000,\
  #         sigmas=250000, fill_value=None)
  l = l+1

# get IASI pixels in WRF domain
listCircle = np.stack((lon_mask,lat_mask), axis=-1)
o3_mean = np.zeros((43,lon_mask.shape[0]))
O3_comp = np.zeros((43,lon_mask.shape[0]))

# check points in each circle
for id_circle,coordinate in enumerate(listCircle):
  # narrow the field of grids to almost one circle
  mask = (lon2d > coordinate[0]-0.1) & (lon2d < coordinate[0]+0.1) &\
  (lat2d > coordinate[1]-0.1) & (lat2d < coordinate[1]+0.1)
  # draw circle (r=0.1 deg ~ 12 km)
  circle = Point(coordinate).buffer(0.1)
  # get center point of grids in rectangle (0.1 * 0.1)
  listPoint = np.stack((lon2d[mask],lat2d[mask]), axis=-1)

  # get index of chosen grid's center points in each circle
  index = []
  for id,position in enumerate(listPoint):
    point = Point (position)
    if circle.contains(point):
      index.append(id)

  # average of O3 profile in each circle
  d = plevel.shape[0]
  for l in np.arange(d):
    o3_mean[l,id_circle] = np.nanmean(o3_nearest[l][mask][index])
  # filter nan of WRF-Chem
  nonan_wrf_index = np.argwhere(~(np.isnan(o3_mean[:,id_circle]))).ravel()
  start = nonan_wrf_index[0]
  end   = nonan_wrf_index[-1]

  # Calculate averaging kernel (A = I - Cx*Ca^-1)
  Ca = np.zeros((d,d))
  inds = np.tril_indices_from(Ca)
  Ca[inds] = Ca1
  Ca[(inds[1], inds[0])] = Ca1

  Cx = np.zeros((d,d))
  inds = np.tril_indices_from(Cx)
  Cx[inds] = Cx1[:,id_circle]
  Cx[(inds[1], inds[0])] = Cx1[:,id_circle]
  Cx = Cx[start:end+1,start:end+1]
  Ca = Ca[start:end+1,start:end+1]

  # filter fill_value=-999.0
  if np.where(Cx[0] == -999.0)[0].size:
    nan_iasi_index = np.where(Cx[0] == -999.0)[0][0]
    Cx = Cx[0:nan_iasi_index,0:nan_iasi_index]
    Ca = Ca[0:nan_iasi_index,0:nan_iasi_index]
    # print (Cx.shape,Ca.shape,nan_iasi_index)
    I = np.identity(nan_iasi_index)
    A = I - Cx.dot(inv(Ca))
    # print (A.shape,o3_mean[start:end+1,id_circle].shape,Xa[start:end+1,id_circle].shape,I.shape)
    O3_comp[start:end+1,id_circle][0:nan_iasi_index] = A.dot(o3_mean[start:end+1,id_circle][0:nan_iasi_index])\
    +(I-A).dot(Xa[start:end+1,id_circle][0:nan_iasi_index])

  else:
    I = np.identity(nonan_wrf_index.shape[0])
    A = I - Cx.dot(inv(Ca))
    O3_comp[start:end+1,id_circle] = A.dot(o3_mean[start:end+1,id_circle])\
    +(I-A).dot(Xa[start:end+1,id_circle])

# Set assemble of poly
p = poly_1
poly_assemble = []
i = 0; k=1
while p[0,0] < p_max:
    if i==2:
        i=0
    poly = Polygon((p[0],p[1],p[2],p[3]))
    if 5 < k < 9:
        if k%2==0:
            p = p + [delta[i],0] + [0.05,0]
        else:
            p = p + [delta[i],0] + [0.03,0]
    elif 9 < k < 14:
        if k%2==0:
            p = p + [delta[i],0] + [0.12,0]
        else:
            p = p + [delta[i],0] + [0.08,0]
    else:
        p = p + [delta[i],0]
    i=i+1; k=k+1;
    poly_assemble.append(poly)

print ('Number of polys '+str(len(poly_assemble)))

fig, axes  = plt.subplots(int(len(poly_assemble)/2), 4,figsize=(10,18),sharey=True)
fig.subplots_adjust(wspace=0.25, hspace=0.2)

# Set axis and threshold
for ax in axes.flat:
  ax.set_yscale('log')
plt.gca().invert_yaxis()
norm = matplotlib.colors.Normalize(vmin=20,vmax=200)

ax_id = 0; cycles_id = 0
for poly in poly_assemble:
    # Get id and position of circles in each poly
    index = []
    for id,position in enumerate(listCircle):
      point = Point (position)
      if poly.contains(point):
        index.append(id)

    # Get geolocations of centers
    lat     = lat_mask[index]
    lon     = lon_mask[index]
    O3_IASI = O3_mask[:,index]
    O3_chem = O3_comp[:,index]

    # Sort ascend
    order = np.argsort(lat)
    lat = lat[order]; lon = lon[order]
    O3_IASI = O3_IASI[:,order]; O3_chem = O3_chem[:,order]

    print ('['+str(cycles_id+1)+'] ','Original lat in Polygon: ','\n',lat)

    # Calculate difference to check bad points and id
    v = abs(np.diff(lat))
    bad = v[np.argwhere(v>0.2)]
    id = np.argwhere(v>0.2)

    # Save bad points and assign null
    add_points =[]
    add_nu = np.array([])
    def average(d,k):
        return (d/k)
    for id_out,d in enumerate(bad):
        k = 2
        n = average(d,k)
        while (n>0.2):
            k = k+1
            n = average(d,k)
        add_points = np.append(add_points,np.linspace(lat[id[id_out]], lat[id[id_out]+1], num=k+1)[1:-1])
        add_nu = np.append(add_nu,(k-1))

    i_saved = np.array([])
    for nu,i in enumerate (id):
        if nu == 0:
            insert_position = i[0]+1
            for k in np.arange(add_nu[nu]):
                O3_IASI = np.insert(O3_IASI, insert_position, np.nan, axis=1)
                O3_chem = np.insert(O3_chem, insert_position, np.nan, axis=1)
                value_position = int(np.sum(add_nu[:nu]))
                lat  = np.insert(lat, insert_position, add_points[int(value_position+k)])
                insert_position += 1
            i_saved = np.append(i_saved,i[0]) 

        else:
            position_relative = int(i[0] - i_saved[nu-1])
            i_saved = np.append(i_saved,i[0])
            insert_position += position_relative

            for k in np.arange(add_nu[nu]):
                O3_IASI = np.insert(O3_IASI, insert_position, np.nan, axis=1)
                O3_chem = np.insert(O3_chem, insert_position, np.nan, axis=1)
                value_position = int(np.sum(add_nu[:nu]))
                lat  = np.insert(lat,  insert_position, add_points[int(value_position+k)])
                insert_position += 1

    print ('['+str(cycles_id+1)+'] '+'Complete lat in Polygon: ','\n',lat)

    # Mask fill_value of O3
    O3_IASI = ma.masked_values (O3_IASI, -999.0)
    O3_chem = ma.masked_values (O3_chem,0)

    # Plot
    x, y = np.meshgrid(lat,plevel[16:])
    cycles_id = cycles_id +1
    
    if cycles_id%2 == 0:
      p = axes[ax_id,2].pcolormesh(x, y, O3_IASI[16:,:]*1000,norm=norm,cmap='jet')
      p = axes[ax_id,3].pcolormesh(x, y, O3_chem[16:,:]*1000,norm=norm,cmap='jet')
      ax_id = ax_id+1
    else:
      p = axes[ax_id,0].pcolormesh(x, y, O3_IASI[16:,:]*1000,norm=norm,cmap='jet')
      p = axes[ax_id,1].pcolormesh(x, y, O3_chem[16:,:]*1000,norm=norm,cmap='jet')

    if ax_id == int(len(poly_assemble)/2)-1:
      cbar_ax = fig.add_axes([0.15, 0.07, 0.7, 0.01])
      cb = fig.colorbar(p, cax=cbar_ax,extend='both', orientation='horizontal')
      cb.ax.set_xlabel('O$_3$ (ppmv)')
      cb.ax.xaxis.set_label_position('top')

# Set ticks and label
ymajor_ticks = np.arange(200, 1200, 200)
for ax in axes.flat:
  ticks  = ax.get_xticks()
  ticks  = list(map(lambda x: '%.1f' % x, ticks))
  labels = [s + '°N' for s in ticks]
  ax.set_xticklabels(labels)
  ax.set_yticks(ymajor_ticks)
  ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
  ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

# Set number
from matplotlib.offsetbox import AnchoredText
numbers = string.ascii_lowercase[0:int(len(poly_assemble))]
numbers = [x for pair in zip(numbers,numbers) for x in pair]
for i,ax in enumerate(axes.flat):
  anchored_text = AnchoredText(numbers[i], loc=2)
  ax.add_artist(anchored_text)

# Set title and ylables
cols = ['IASI_Profile','WRF-Chem Profile']*2
rows = ['Pressure (hPa)']*int(len(poly_assemble)/2)

for ax, col in zip(axes[0], cols):
    ax.set_title(col)
for ax, row in zip(axes[:,0], rows):
    ax.set_ylabel(row, rotation=90, labelpad=2)

plt.savefig(saveplace+savename,bbox_inches = 'tight',pad_inches=0.2)