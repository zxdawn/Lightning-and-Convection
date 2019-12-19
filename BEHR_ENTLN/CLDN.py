import os
import re
import glob
import calendar
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from netCDF4 import num2date, date2num
from datetime import datetime, timedelta

file_dir = '/home/xin/Documents/lightning_data/CGData/'
save_dir = '/home/xin/Documents/lightning_data/CG_nc/'
years = [str(i) for i in np.arange(2008,2016,1)]

def write_geo(save_nc, datagrp, lon_center, lat_center):
    # Create group
    geogrp = save_nc.createGroup('Geolocation_Fields')

    # Create dimension
    lon = geogrp.createDimension('lon', lon_center.shape[0])
    lon = datagrp.createDimension('lon', lon_center.shape[0])
    lat = geogrp.createDimension('lat', lat_center.shape[0])
    lat = datagrp.createDimension('lat', lat_center.shape[0])

    # Create variables
    latitudes         = geogrp.createVariable('Latitude', 'f4',('lat',),zlib=True)
    longitudes        = geogrp.createVariable('Longitude', 'f4',('lon',),zlib=True)
    latitudes.units   = 'degree_north'
    longitudes.units  = 'degree_east'

    # Write data
    latitudes[:]      = lat_center
    longitudes[:]     = lon_center


def write_NPCG(files, variable):
    index_file = 0
    for file in files:
        with open(file, 'r',encoding='GB18030') as f:
            lines = f.readlines(0)
            # Parameters of head:
            #   start_lon, start_lat, end_lon, end_lat, d_lon and d_lat
            # start_lon, start_lat, end_lon, end_lat, d_lon, d_lat \
            #     = list(map(float, re.findall(r'\d+\.\d+', lines[0])))

            # lat_center = np.arange(south,north+resolution,resolution)
            # lon_center = np.arange(west,east+resolution,resolution)

            data = np.asarray(list(map(str.split, lines[1:]))).astype(np.int)

            # variable[index_file,int((start_lon-west)/d_lon):-int((east-end_lon)/d_lon),\
            # int((start_lat-south)/d_lat):-int((north-end_lat)/d_lat)] = data

            variable[index_file,:,:] = data
        index_file += 1


def write_data(datagrp, NCG_files, PCG_files, NCG, PCG, CG):
    write_NPCG(NCG_files, NCG)
    write_NPCG(PCG_files, PCG)
    CG[:] = NCG[:] + PCG[:]


def main(save_dir, file_dir, filenames, north, south, west, east, resolution):
    # Get bins of lon/lat
    lat_center = np.linspace(south,north,(north-south)/resolution+1)
    lon_center = np.linspace(west,east,(east-west)/resolution+1)

    # Get dates
    dates_name = pd.date_range(filenames[0][6:-7].replace('_','-'), filenames[-1][6:-7].replace('_','-'),freq='D').to_pydatetime()
    dates_name = np.vectorize(lambda s: s.strftime('%Y_%m_%d'))(dates_name)

    for date in dates_name:
        print ('Saving '+date+' data .....')
        NCG_files = sorted(glob.glob(file_dir+'*AcNCG-'+date+'*'))
        PCG_files = sorted(glob.glob(file_dir+'*AcPCG-'+date+'*'))

        # Create nc file
        save_nc = Dataset(save_dir+date.replace('_','')+'.nc','w',format='NETCDF4')

        # Save geo_data
        datagrp = save_nc.createGroup('Data_fields')
        write_geo(save_nc, datagrp, lon_center, lat_center)

        # Save date
        time    = datagrp.createDimension('time', 24)
        times   = datagrp.createVariable('time','i4',('time',),zlib=True)

        sdate = datetime.strptime(filenames[0][6:-4], '%Y_%m_%d_%H')
        times.units = 'hours since '+sdate.strftime('%Y-%m-%d %H:%M:%S')
        times.calendar = 'gregorian'
        dates       = [datetime(int(date[0:4]),int(date[5:7]),int(date[8:]))+n*timedelta(hours=1) for n in range(times.shape[0])]
        times[:]    = date2num(dates,units=times.units,calendar=times.calendar)

        # Save CG
        PCG = datagrp.createVariable('PCG','i4',('time','lat','lon'),zlib=True)
        NCG = datagrp.createVariable('NCG','i4',('time','lat','lon'),zlib=True)
        CG = datagrp.createVariable('CG','i4',('time','lat','lon'),zlib=True)

        # Save all lightning data
        write_data(datagrp, NCG_files, PCG_files, NCG, PCG, CG)


if __name__ == '__main__':
    for year in years:

        filenames = sorted([filename for filename in glob.glob(file_dir+'Ac*'+year+'*dat')])
        filenames = [os.path.basename(x) for x in filenames]

        # Domain
        resolution = 0.1
        if year in ['2014','2015']:
            # Guang Dong
            north = 57; south = 17; west = 70; east = 140
        else:
            # China
            north = 26; south = 20; west = 109; east = 118

        main(save_dir, file_dir, filenames, north, south, west, east, resolution)