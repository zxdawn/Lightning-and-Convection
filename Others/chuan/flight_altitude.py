############
# Packages #
############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from mpl_toolkits.basemap import Basemap

##############
# Parameters #
##############
file = 'flight_11.7.xlsx'
sheet_name = 'Sheet1'

shp = 'D:/research/python/shapefile/data/CHN_adm_shp/CHN_adm3'
save_dir = 'C:/Users/Xin/Desktop/'
outputname = 'flight_11.7.png'

row = 1; col = 1
figsize = [12,9]
fig, ax = plt.subplots(1, 1, figsize=figsize)

cmap='hsv'
linewidth=3
alpha=1.0

########
# Main #
########
def main():
    lon, lat, alt, Lat_min, Lat_max, Lon_min, Lon_max = read_file(file, sheet_name)
    m = drawmap(ax, Lat_min, Lat_max, Lon_min, Lon_max)
    multicolored_lines(ax, lon, lat, alt, cmap, linewidth, alpha)
    plt.savefig(save_dir+outputname, bbox_inches='tight', pad_inches=0.2)

#############
# Functions #
#############
def read_file(file, sheet_name):
    f   = pd.read_excel(file, sheet_name=sheet_name)
    t   = f['CreationTime'].values
    lon = f['Longitude'].values
    lat = f['Latitude'].values
    alt = f['Altitude'].values
    Lat_min, Lat_max, Lon_min, Lon_max = lat.min()-0.2, lat.max()+0.2, lon.min()-0.2, lon.max()+0.2
    return lon, lat, alt, Lat_min, Lat_max, Lon_min, Lon_max

def multicolored_lines(ax, lon, lat, alt, cmap, linewidth, alpha):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """
    alt_min = alt.min()
    alt_max = alt.max()
    norm = plt.Normalize(vmin=alt_min, vmax=alt_max)
    lc = colorline(ax, lon, lat, alt, norm, cmap, linewidth, alpha)
    cbar = plt.colorbar(lc)
    cbar.set_label('Altitude (m)')

def colorline(ax, lon, lat, alt, norm, cmap, linewidth, alpha):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    segments = make_segments(lon, lat)
    lc = mcoll.LineCollection(segments, array=alt, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)
    ax.add_collection(lc)

    return lc

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def drawmap(ax,Lat_min,Lat_max,Lon_min,Lon_max):
    m=Basemap(llcrnrlat=Lat_min,urcrnrlat=Lat_max,llcrnrlon=Lon_min,urcrnrlon=Lon_max)
    m.readshapefile(shp,'CHN')

    parallels = np.arange(-90.,91.,0.1)
    meridians = np.arange(-180.,181.,0.2)

    m.drawparallels(parallels,labels=[1,0,0,1],linewidth=0.15,xoffset=0.03,fontsize=12,fontname='Arial')
    m.drawmeridians(meridians,labels=[1,0,0,1],linewidth=0.15,yoffset=0.03,fontsize=12,fontname='Arial')

    xminor_ticks = np.arange(Lon_min,Lon_max,0.2)
    yminor_ticks = np.arange(Lat_min,Lat_max, 0.1)
    ax.set_xticks(xminor_ticks, minor=True)
    ax.set_yticks(yminor_ticks, minor=True)

    return m

if __name__ == '__main__':
    main()