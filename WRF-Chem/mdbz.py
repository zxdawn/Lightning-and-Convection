############
# Packages #
############

import os,fnmatch
import numpy as np
import matplotlib
from netCDF4 import Dataset
import matplotlib.colors as colors
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from wrf import to_np, getvar, smooth2d, get_basemap, latlon_coords,extract_times,ALL_TIMES
#matplotlib.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})

##############
# Parameters #
##############

# wrf_dir: the directory of wrfout* file

# wrf_name: name of wrfout_file or one of wrfout_files

# map_dir: the dir of map

# save_dir: the directory to which to save the figure

# row: row of figure; col: column of figure

# Lon_min,Lon_max,Lat_min,Lat_max: crop wrfout*

# drawclb: whether draw colorbar

# extend : colorbar_extend: 'neither', 'both', 'min', 'max'

wrf_dir = '/nuist/u/home/yinyan/xin/work/wrfchem_3.9.1/WRFV3/test/em_real/'
wrf_name = 'wrfout_d03_2017-04-19_18:00:00'
map_dir = './data/'
save_dir = '/nuist/u/home/yinyan/xin/work/plots/wrf/try/'

Lat_min = 20; Lat_max = 26;
Lon_min = 109; Lon_max = 118;

row=1; col = 1
figsize = [12,9]

drawclb = False
extend = 'max'

########
# Main #
########

def main():
    fig, ax = plt.subplots(row, col,figsize=figsize)
    wrf_file = Dataset(wrf_dir+wrf_name)
    m = drawmap(ax,wrf_file,map_dir,Lat_min,Lat_max,Lon_min,Lon_max)
    times = extract_times(wrf_file,timeidx=ALL_TIMES)

    # Check frames of wrfout*
    if times.shape[0] > 1:
        pmdbz(wrf_file,times,m,save_dir,drawclb)
    else:
        for file in os.listdir(wrf_dir):
            if fnmatch.fnmatch(file, wrf_name[:10]+'*'):
                wrf_file = Dataset(wrf_dir+file)
                times = extract_times(wrf_file,timeidx=ALL_TIMES)
                pmdbz(wrf_file,times,m,save_dir,drawclb)

#############
# Functions #
#############

def pmdbz(wrf_file,times,m,save_dir,drawclb):
    for time in range(times.shape[0]):
        mdbz = getvar(wrf_file, "mdbz",timeidx=time)
        lats, lons = latlon_coords(mdbz)
        x, y = m(to_np(lons), to_np(lats))
        mdbz = to_np(mdbz)
        smooth_mdbz = smooth2d(mdbz, 3)
        smooth_mdbz = to_np(smooth_mdbz)

        mask = smooth_mdbz <= 0
        smooth_mdbz[mask] = np.nan

        bounds = np.arange(0,80,5)
        colors = ['#00FFFF','#009DFF','#0000FF','#0982AF','#00FF00',\
        '#08AF14','#FFD600','#FF9800','#FF0000','#DD001B','#BC0036',\
        '#79006D','#7933A0','#C3A3D4','#FFFFFF']

        im = m.contourf(x,y, smooth_mdbz,10,levels=bounds,colors=colors)

        if drawclb:
            clb = plt.colorbar(shrink=0.75,ticks=bounds,pad=0.05)
            clb.ax.tick_params(labelsize=15)
            clb.solids.set_edgecolor("face")
            for l in clb.ax.yaxis.get_ticklabels():
                l.set_family('Arial')

        name = str(times[time])[:10] + '_' + str(times[time])[11:16] + 'UTC'
        plt.title(str(times[time])[11:13]+str(times[time])[14:16] + ' UTC WRF-Chem',{'size':28},x=0.02,y=0.91,loc='left',bbox=dict(facecolor=(1,1,1,0.5),edgecolor=(0,0,0,0)))
        plt.savefig(save_dir+name+'.jpg',bbox_inches = 'tight')
        
        # remove elements if times.shape[0] > 1
        if drawclb:
            clb.remove()
        for im in im.collections:
            im.remove()

def drawmap(ax,wrf_file,map_dir,Lat_min,Lat_max,Lon_min,Lon_max):
    m = get_basemap(wrfin=wrf_file,llcrnrlat=Lat_min,urcrnrlat=Lat_max,\
    llcrnrlon=Lon_min,urcrnrlon=Lon_max,resolution ='l',ax=ax)
    CHNshp = map_dir+'CHN_adm_shp/CHN_adm1'
    TWNshp = map_dir+'TWN_adm_shp/TWN_adm0'
    m.readshapefile(CHNshp,'CHN',drawbounds = True)
    m.readshapefile(TWNshp,'TWN',drawbounds = True)

    parallels = np.arange(-90.,91.,1.)
    meridians = np.arange(-180.,181.,2.)
    m.drawparallels(parallels,labels=[1,0,0,1],linewidth=0.2,xoffset=0.2,fontsize=12,fontname='Arial')
    m.drawmeridians(meridians,labels=[1,0,0,1],linewidth=0.2,yoffset=0.2,fontsize=12,fontname='Arial')

    xminor_ticks = np.arange(Lon_min,Lon_max,1)
    yminor_ticks = np.arange(Lat_min,Lat_max, 1)
    ax.set_xticks(xminor_ticks, minor=True)
    ax.set_yticks(yminor_ticks, minor=True)

    for info, shape in zip(m.CHN_info, m.CHN):
        if info['NAME_1'] =='Guangdong':
            x, y = zip(*shape)
            m.plot(x, y, marker = None, color= 'k',linewidth=1.5)

    return m

if __name__ == '__main__':
    main()