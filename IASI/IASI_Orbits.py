import numpy as np
import datetime
import matplotlib.colors
import h5py
from shapely.geometry import Polygon,Point,LinearRing
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
plt.rcParams["font.family"] = "Arial"

Lat_min = 20
Lat_max = 26
Lon_min = 109
Lon_max = 118

poly_1  = np.array([[ 111.3,21.5 ],
                [ 112.25,  25.  ],
                [ 112.45,  25.  ],
                [ 111.5,   21.5 ]])

p_max  = 115
d =[0.2,0.35]

directory   = '/data/home/wang/zx/data/IASI/GD_files/'
file_place1 = directory+'IASI-MetopA_L2_O3_SOFRID-v3.5_2017042000.he5'
file_place2 = directory+'IASI-MetopA_L2_O3_SOFRID-v3.5_2017042012.he5'

saveplace  = '/data/home/wang/zx/codes/interplation/'
savename   = 'IASI_Orbits.eps'
lat_str    = 'HDFEOS/SWATHS/O3/Geolocation Fields/Latitude'
lon_str    = 'HDFEOS/SWATHS/O3/Geolocation Fields/Longitude'
O3_str     = 'HDFEOS/SWATHS/O3/Data Fields/Ozone'
TO3_str    = 'HDFEOS/SWATHS/O3/Data Fields/O3 Total Column'
hour_str   = 'HDFEOS/SWATHS/O3/Geolocation Fields/Hour'
minute_str = 'HDFEOS/SWATHS/O3/Geolocation Fields/Minute'
second_str = 'HDFEOS/SWATHS/O3/Geolocation Fields/Second'

#set maps
def drawmap(ax):
    #define range of plot and plot provinces
    m=Basemap(llcrnrlat=Lat_min,urcrnrlat=Lat_max,llcrnrlon=Lon_min,urcrnrlon=Lon_max,resolution ='l',ax=ax)
    CHNshp = CHNshp = '/data/home/wang/zx/codes/wrfpython/data/CHN_adm_shp/CHN_adm1'
    m.readshapefile(CHNshp,'CHN',drawbounds = True)
    TWNshp = '/data/home/wang/zx/codes/wrfpython/data/TWN_adm_shp/TWN_adm0'
    m.readshapefile(TWNshp,'TWN',drawbounds = True)

    parallels = np.arange(-90.,91.,1.)
    meridians = np.arange(-180.,181.,2.)
    m.drawparallels(parallels,labels=[1,0,0,1],linewidth=0.2,xoffset=0.2,fontsize=13,fontname='Arial')
    m.drawmeridians(meridians,labels=[1,0,0,1],linewidth=0.2,yoffset=0.2,fontsize=13,fontname='Arial')

    xminor_ticks = np.arange(Lon_min,Lon_max,1)
    yminor_ticks = np.arange(Lat_min,Lat_max, 1)
    ax.set_xticks(xminor_ticks, minor=True)
    ax.set_yticks(yminor_ticks, minor=True)

    return m

def read(file,column):
    f = h5py.File (file,'r')
    lat    = f[lat_str][:]
    lon    = f[lon_str][:]
    O3     = f[O3_str][:]
    TO3    = f[TO3_str][:]
    hour   = f[hour_str][:]
    minute = f[minute_str][:]
    second = f[second_str][:]

    mask = (lon>Lon_min) & (lon<Lon_max) & (lat>Lat_min) & (lat<Lat_max)
    lon = lon[mask]
    lat = lat[mask]
    h = int(np.mean(hour[mask]))
    m = int(round(np.mean(minute[mask])))

    axes.scatter(lon, lat,s=1,c='k')
    axes.set_title('IASI_Orbits '+str(h).zfill(2)+str(m)+' UTC',fontsize=15)

    p = poly_1
    i = 0; k =1;

    while p[0,0] < p_max:
        if i==2:
            i=0
        
        poly = Polygon((p[0],p[1],p[2],p[3]))
        poly_assemble.append(poly)

        ring = LinearRing((p[0],p[1],p[2],p[3]))
        x, y = ring.xy

        # Draw coordinates and location strings
        axes.plot(x,y,linewidth=3)

        ## Add coordinates str
        # listCircle = np.stack((lon,lat), axis=-1)
        # index = []
        # for id,position in enumerate(listCircle):
        #   point = Point (position)
        #   if poly.contains(point):
        #     index.append(id)
        # lat     = lat[index]
        # lon     = lon[index]
        # for x, y in zip(lon,lat):
        #     text = str(x) + ', ' + str(y)
        #     axes.text(x, y, text,fontsize=5)

        # axes[column].scatter(lon, lat,s=1,c='k')
        # axes[column].set_title(str(h).zfill(2)+str(m)+' UTC',fontsize=15)
        # for x, y in zip(lon,lat):
        #     text = str(x) + ', ' + str(y)
        #     axes[column].text(x, y, text,fontsize=5)
        if 5 < k < 9:
            if k%2==0:
                p = p + [d[i],0] + [0.05,0]
            else:
                p = p + [d[i],0] + [0.03,0]
        elif 9 < k < 14:
            if k%2==0:
                p = p + [d[i],0] + [0.12,0]
            else:
                p = p + [d[i],0] + [0.08,0]
        else:
            p = p + [d[i],0]
        i=i+1; k=k+1;


fig, axes = plt.subplots(1, 1,figsize=(12,10))
fig.subplots_adjust(wspace=0.1, hspace=0.25)

drawmap(axes)
poly_assemble = []
read(file_place1,0)

plt.savefig(saveplace+savename,bbox_inches = 'tight')
