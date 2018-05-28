############
# Packages #
############
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely.geometry import Point
from netCDF4 import Dataset, num2date
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap


##############
# Parameters #
##############
# month: month of median wind speed
# file: ERA5 u/v file
# shp: shapefile used to crop
# Lon_min,Lon_max,Lat_min,Lat_max: boundary of map

month = 6
file = '/data/home/wang/zx/data/ERA5/wind/data/2011/ERA5-20110101-20151231-pl.nc'
shp = '/data/home/wang/zx/codes/wrfpython/data/CHN_adm_shp/CHN_adm1'

Lat_min = 20; Lat_max = 26;
Lon_min = 109; Lon_max = 118;

row=1; col = 1
figsize = [12,9]
fig, ax = plt.subplots(1, 1,figsize=(12,9))

save_dir = '/data/home/wang/zx/data/ERA5/wind/data/2011/'
outputname = 'ERA5_median_wind.png'

########
# Main #
########
def main():
    fig, ax = plt.subplots(row, col,figsize=figsize)
    lon,lat,u,v = read_ERA5(file,month)
    m = drawmap(ax,Lat_min,Lat_max,Lon_min,Lon_max)
    wind = crop_wind(m,u,v,lon,lat,Lon_min,Lon_max,Lat_min,Lat_max)
    residence = get_residence(wind)
    plot_residence(ax,lon,lat,residence,save_dir,outputname)


#############
# Functions #
#############
def read_ERA5(file,month):
    dataset = Dataset(file)

    t   = dataset.variables['time']
    u   = dataset.variables['u']
    v   = dataset.variables['v']
    lon = dataset.variables['longitude']
    lat = dataset.variables['latitude']

    # Calculate datetime
    t_unit = t.units
    t_cal = t.calendar
    datevar = num2date(t[:],units = t_unit,calendar = t_cal)
    mon = [date.month for date in datevar]

    # Filter Jun to Aug data
    id = [counter for (counter,value) in enumerate(mon) if month-1<int(value)<month+1]
    u = u[id]; v = v[id]

    return lon,lat,u,v

def drawmap(ax,Lat_min,Lat_max,Lon_min,Lon_max):
    m=Basemap(llcrnrlat=Lat_min,urcrnrlat=Lat_max,llcrnrlon=Lon_min,urcrnrlon=Lon_max)
    m.readshapefile(shp,'CHN')

    parallels = np.arange(-90.,91.,1.)
    meridians = np.arange(-180.,181.,2.)

    m.drawparallels(parallels,labels=[1,0,0,1],linewidth=0.2,xoffset=0.2,fontsize=12,fontname='Arial')
    m.drawmeridians(meridians,labels=[1,0,0,1],linewidth=0.2,yoffset=0.2,fontsize=12,fontname='Arial')

    xminor_ticks = np.arange(Lon_min,Lon_max,1)
    yminor_ticks = np.arange(Lat_min,Lat_max, 1)
    ax.set_xticks(xminor_ticks, minor=True)
    ax.set_yticks(yminor_ticks, minor=True)

    return m

def crop_wind(m,u,v,lon,lat,Lon_min,Lon_max,Lat_min,Lat_max):
    # Filter points in land
    loc = np.array(np.meshgrid(lat[:],lon[:])).T

    # Filter polygons(Guangdong) of shapefile
    polygons = []
    for info, shape in zip(m.CHN_info, m.CHN):
        if info['NAME_1'] == 'Guangdong':
            polygons.append(Polygon(np.array(shape), True))

    # Generate boolean matrix
    filter = np.zeros((loc.shape[0], loc.shape[1]), dtype=bool)
    for i in range(loc.shape[0]):
        for j in range(loc.shape[1]):
            grid_point = Point(loc[i,j][1],loc[i,j][0])
            a = []
            for p in polygons:
                a.append(p.contains(grid_point)[0])
            if not any(a):
                filter[i,j] = True

    # Calculate wind speed
    u[:,filter] = np.nan; v[:,filter] = np.nan
    wind = np.sqrt(u*u+v*v)

    return wind

def get_residence(wind):
    wind = np.nanmedian(wind, axis=0)
    print (np.nanmax(wind))

    R = 6378.1 # Radius od Earth (km)
    side = (math.pi/180)*(R-23/90*21.3)*10**3
    l = np.sqrt(2*side*side) # length of diagonal(m)
    l = np.full_like(wind, l)
    residence = (l/wind)/3600 # residence (h)

    # residence_mean = np.nanmean(residence)
    # residence_min  = np.nanmin(residence)
    # residence_max  = np.nanmax(residence)
    # print ('residence_mean',residence_mean)
    # print ('residence_min',residence_min)
    # print ('residence_max',residence_max)

    return residence

def plot_residence(ax,lon,lat,residence,save_dir,outputname):
    x,y = np.meshgrid(lon[:],lat[:])
    p = ax.pcolormesh(x, y,residence)
    # norm = mpl.colors.Normalize(vmin=6,vmax=7)
    # ticks=np.linspace(6,7,11)
    clb = fig.colorbar(p,ax=ax,cmap='jet',shrink=0.7,orientation='vertical')
    clb.set_label('median residence (h)', labelpad=5)
    plt.savefig(save_dir+outputname,bbox_inches = 'tight',pad_inches=0.2)

if __name__ == '__main__':
    main()