# Calculate LNOx based on BEHR and ENTLN data
# and save as one .nc file by swaths

# Parameters:
#   north, south, west and east
#   t_window, CRF_threshold, CF_threshold, flash_threshold, stroke_threshold and min_pixels
#   directory of BEHR and ENTLN data

# Filter conditions:
#   valid pixels >= 5 for each 1*1 grid:
#   CRF >= CRF_threshold for each OMI pixel;
#   CF  >= CF_threshold for each OMI pixel;
#   Flashes >= flash_threshold for 1*1 grid t_window before OMI overpass;
#   Strokes >= stroke_threshold for 1*1 grid t_window before OMI overpass

#   CRF_threshold and min_pixels, see Pickering et al. (2016);
#   CF_threshold, see Sarah et al. (2017);
#   flashth_reshold, stroke_threshold and t_window, see Jeff et al. (2018 submitted)
#   Method of dealing with negative and large value, see https://github.com/CohenBerkeleyLab/BEHR-core/issues/8

# Structure of .nc files:
# group: Data_fields {
#    group: flash {
#             group: 'date' {
#                group: 'Swath' {
#                    variables:......
#        }
#
#      }
#
#    }
#    group: stroke {
#             group: 'date' {
#                group: 'Swath' {
#                    variables:......
#        }
#
#      }
#
#    }
#  }
#  group: Geolocation_Fields {
#    dimensions:
#      lon = 33;
#      lat = 28;
#    variables:
#      float Latitude(lat=28);
#      float Longitude(lon=33);
#  }

# Xin Zhang <xinzhang1215@gmail.com> 17 Jul 2018

import os
import csv
import h5py
import ntpath
import fnmatch
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from netCDF4 import Dataset
from functools import partial
from datetime import datetime, timedelta

default_vals =  {'north': 50.5, 'south': 21.5,
                'west': -110.5, 'east': -76.5,
                'CRF_threshold': 0.7, 'CF_threshold': 0.4,
                'flash_threshold': 2.4, 'stroke_threshold': 2.4,
                'min_pixels': 5, 't_window': 2.4,
                'debug': 0}

bad_flags = [2,4,19] # 2:AMF error, 4:row anomaly, 19:above-cloud

behr_dir  = '/public/home/zhangxin/bigdata/BEHR_data/'
entln_dir = '/public/home/zhangxin/bigdata/ENTLN_data/'
save_dir  = '/public/home/zhangxin/bigdata/OMILNOx_data/'

def parse_args():
    '''
    Parses command line arguments given in bash. Assumes that all arguments are flag-value pairs
     using long-option nomenclature (e.g. --CRF_threshold 0.7).
    These are arguments that if not specified have reasonable default values.default.
     The key will be the flag name (without the --) and the value is the default value.
    '''
    parser = argparse.ArgumentParser(description='program to calculate LNOx from ENTLN and BEHR data')
    parser.add_argument('--north', default = default_vals['north'], type = float, help = 'north bound')
    parser.add_argument('--south', default = default_vals['south'], type = float, help = 'south bound')
    parser.add_argument('--west', default = default_vals['west'], type = float, help = 'west bound')
    parser.add_argument('--east', default = default_vals['east'], type = float, help = 'east bound')
    parser.add_argument('--CRF_threshold', default = default_vals['CRF_threshold'], type = float, help = 'CRF_threshold')
    parser.add_argument('--CF_threshold', default = default_vals['CF_threshold'], type = float, help = 'CF_threshold')
    parser.add_argument('--flash_threshold', default = default_vals['flash_threshold'], type = float, help = 'flash_threshold')
    parser.add_argument('--stroke_threshold', default = default_vals['stroke_threshold'], type = float, help = 'stroke_threshold')
    parser.add_argument('--min_pixels', default = default_vals['min_pixels'], type = float, help = 'min_pixels')
    parser.add_argument('--t_window', default = default_vals['t_window'], type = float, help = 't_window')
    parser.add_argument('--debug', default = default_vals['debug'], type = int, help = 'debug level')

    args = parser.parse_args()

    return vars(args)


def power_find(n):
    result = []
    binary = bin(n)[:1:-1]
    for x in range(len(binary)):
        if int(binary[x]):
            result.append(x)

    return result


def bin_mathod(lon, lat, bin_lon, bin_lat, method, *args):
    return [stats.binned_statistic_2d(lon, lat, arg, method, bins=[bin_lon, bin_lat]).statistic \
            for arg in args]


def check(mask, *args):
    return [arg[mask] for arg in args]


# def convert_array(*args):
    # return [np.array(arg) for arg in args]


def get_omidate(t, t_window):
    ref_time = datetime(1993, 1, 1)
    edate = ref_time + timedelta(seconds=float(t))
    sdate = edate - timedelta(hours=t_window)

    return sdate, edate


def read_entln(filename, times, t_window, lon, lat, bin_lon, bin_lat):
    df   = pd.read_csv(filename, delimiter=',')
    date = pd.to_datetime(df['timestamp'])
    lon2 = df['longitude']
    lat2 = df['latitude']

    # Filter t_window before OMI overpass
    sdate, edate = get_omidate(times.mean(), t_window)
    mask = (sdate <= date) & (date <= edate)

    # Get CG and IC flashes/strokes
    type = df['type'].loc[mask]
    CG = type[type == 0 | 40].values
    IC = type[type == 1].values

    # Get lon and lat
    lon_CG = lon2.loc[mask][type == 0 | 40].values
    lat_CG = lat2.loc[mask][type == 0 | 40].values
    lon_IC = lon2.loc[mask][type == 1].values
    lat_IC = lat2.loc[mask][type == 1].values

    # Accurate time, but very slow.
    # Because duration of each swath is ~ 9 min, I decide to use average overpass (above).
    # Xin (Aug 23, 2018)
    # 
    # CG, IC, lat_CG, lat_IC, lon_CG, lon_IC = [], [], [], [], [], []
    # for counter, t in enumerate(times):
    #     # Filter t_window before OMI overpass
    #     sdate, edate = get_omidate(t, t_window)
    #     print (sdate,edate,lon[counter],lat[counter])
    #     mask = (sdate <= date) & (date <= edate) & (lon[counter]-0.05 <= lon2) & (lon2<= lon[counter]+0.05) \
    #                 & (lat[counter]-0.05 <= lat2) & (lat2<= lat[counter]+0.05) 

    #     # Get CG and IC flashes/strokes
    #     type = df['type'].loc[mask]
    #     CG.extend(type[type == 0 | 40])
    #     IC.extend(type[type == 1])

    #     # Get lon and lat
    #     lon_CG.extend(lon2.loc[mask][type == 0 | 40])
    #     lon_IC.extend(lon2.loc[mask][type == 1])
    #     lat_CG.extend(lat2.loc[mask][type == 0 | 40])
    #     lat_IC.extend(lat2.loc[mask][type == 1])

    # Convert list to array for binned_statistic_2d
    # CG, IC, lon_CG, lon_IC, lat_CG, lat_IC = convert_array(CG, IC, lon_CG, lon_IC, lat_CG, lat_IC)

    if len(CG) == 0:
        CG_bin = np.zeros((bin_lon.shape[0]-1, bin_lat.shape[0]-1))
    else:
        CG_bin = stats.binned_statistic_2d(lon_CG, lat_CG, None, \
                'count', bins=[bin_lon,bin_lat]).statistic/1000.0 #kFlashes(kstrokes)

    if len(IC) == 0:
        IC_bin = np.zeros((bin_lon.shape[0]-1, bin_lat.shape[0]-1))
    else:
        IC_bin = stats.binned_statistic_2d(lon_IC, lat_IC, None, \
                'count', bins=[bin_lon,bin_lat]).statistic/1000.0 #kFlashes(kstrokes)

    return CG_bin, IC_bin


def read_behr_swath(f, swath, bin_lon, bin_lat, CRF_threshold, CF_threshold, t_window):
    # Read BEHR variables
    T                  = f['Data/'+swath+'/Time_2D'][:]
    lon                = f['Data/'+swath+'/Longitude'][:]
    lat                = f['Data/'+swath+'/Latitude'][:]
    CRF                = f['Data/'+swath+'/CloudRadianceFraction'][:]
    CP                 = f['Data/'+swath+'/CloudPressure'][:]
    CF                 = f['Data/'+swath+'/CloudFraction'][:]
    AMFLNOx            = f['Data/'+swath+'/BEHRAMFLNOx'][:]
    LNOx               = f['Data/'+swath+'/BEHRColumnAmountLNOxTrop'][:]
    AMFLNOx_pickering  = f['Data/'+swath+'/BEHRAMFLNOx_pickering'][:]
    LNOx_pickering     = f['Data/'+swath+'/BEHRColumnAmountLNOxTrop_pickering'][:]
    flag               = f['Data/'+swath+'/BEHRQualityFlags'][:]

    # Exclude NaN value
    T, lon, lat, CRF, CP, CF, AMFLNOx, AMFLNOx_pickering, LNOx, LNOx_pickering, flag  = \
            check(T>0, T, lon, lat, CRF, CP, CF, AMFLNOx, AMFLNOx_pickering, LNOx, LNOx_pickering, flag)

    # Filter_1.1: CRF and CF
    filter_CRF  = CRF >= CRF_threshold
    filter_CF   =  CF >= CF_threshold
    T, lon, lat, CRF, CP, AMFLNOx, AMFLNOx_pickering, LNOx, LNOx_pickering, flag = check(filter_CRF & filter_CF, T, lon, lat, CRF, CP, AMFLNOx, AMFLNOx_pickering, LNOx, LNOx_pickering, flag)

    if len(T) == 0:
        valid_pixels, CRF_bin, CP_bin, AMFLNOx_bin, AMFLNOx_pickering_bin, LNOx_bin, LNOx_pickering_bin = \
            (np.zeros((bin_lon.shape[0]-1, bin_lat.shape[0]-1)) for i in range(7))
    else:
        # Filter_1.2-1.4: quality flags and exclude negative values
        filter_quality  = [not any(x in bad_flags for x in power_find(i)) for i in flag]
        filter_positive = (LNOx>0) & (LNOx_pickering>0)
        filter_nonan    = ~np.isnan(AMFLNOx) | ~np.isnan(AMFLNOx_pickering)

        T, lon, lat, CRF, CP, AMFLNOx, AMFLNOx_pickering, LNOx, LNOx_pickering, flag = check(filter_quality & filter_positive & filter_nonan, T, lon, lat, CRF, CP, AMFLNOx, AMFLNOx_pickering, LNOx, LNOx_pickering, flag)

        if len(T) == 0:
            valid_pixels, CRF_bin, CP_bin, AMFLNOx_bin, AMFLNOx_pickering_bin, LNOx_bin, LNOx_pickering_bin = \
                (np.zeros((bin_lon.shape[0]-1, bin_lat.shape[0]-1)) for i in range(7))
        else:
            # Filter_2: Number of pixels meet Filter_1 condition.
            # This will be used as condition for valid pixels
            valid_pixels = stats.binned_statistic_2d(lon, lat, LNOx, \
                                statistic=lambda LNOx: np.count_nonzero(LNOx), \
                                bins=[bin_lon,bin_lat]).statistic

            #Bin variables
            CRF_bin, CP_bin, AMFLNOx_bin, AMFLNOx_pickering_bin, LNOx_bin, LNOx_pickering_bin\
                    = bin_mathod(lon, lat, bin_lon, bin_lat, 'mean', CRF, CP, AMFLNOx, AMFLNOx_pickering, LNOx, LNOx_pickering)

    return T, lon, lat, valid_pixels, CRF_bin, CP_bin, AMFLNOx_bin, AMFLNOx_pickering_bin, LNOx_bin, LNOx_pickering_bin


def write_geo(save_nc, lon_center, lat_center):
    # Create group
    geogrp = save_nc.createGroup('Geolocation_Fields')

    # Create dimension
    lon = geogrp.createDimension('lon', lon_center.shape[0])
    lat = geogrp.createDimension('lat', lat_center.shape[0])

    # Create variables
    latitudes         = geogrp.createVariable('Latitude', 'f4',('lat',),zlib=True)
    longitudes        = geogrp.createVariable('Longitude', 'f4',('lon',),zlib=True)
    latitudes.units   = 'degree_north'
    longitudes.units  = 'degree_east'

    # Write data
    latitudes[:]      = lat_center
    longitudes[:]     = lon_center


def write_data(kind, date_str, save_nc, swath, lon_center, lat_center, \
    CRF_bin, CP_bin, AMFLNOx_bin, AMFLNOx_pickering_bin, LNOx_bin, LNOx_pickering_bin, TL_bin):
    # Create group
    swathgrp = save_nc.createGroup('/Data_fields/'+kind+'/'+date_str+'/'+swath)

    # Create dimension
    lon = swathgrp.createDimension('lon', lon_center.shape[0])
    lat = swathgrp.createDimension('lat', lat_center.shape[0])

    # Create variables
    CRF               = swathgrp.createVariable('CloudRadianceFraction', 'f4', ('lon','lat'), fill_value=0., zlib=True)
    CP                = swathgrp.createVariable('CloudPressure', 'f4', ('lon','lat'), fill_value=0., zlib=True)
    AMFLNOx           = swathgrp.createVariable('AMFLNOx', 'f4', ('lon','lat'), fill_value=0., zlib=True)
    AMFLNOx_pickering = swathgrp.createVariable('AMFLNOx_pickering', 'f4', ('lon','lat'), fill_value=0., zlib=True)
    LNOx              = swathgrp.createVariable('LNOx', 'f4', ('lon','lat'), fill_value=0., zlib=True)
    LNOx_pickering    = swathgrp.createVariable('LNOx_pickering', 'f4', ('lon','lat'), fill_value=0.,zlib=True)

    if kind == 'flash':
        Flashes = swathgrp.createVariable('Flashes', 'f4', ('lon','lat'), fill_value=0., zlib=True)
    else:
        Strokes = swathgrp.createVariable('Strokes', 'f4', ('lon','lat'), fill_value=0., zlib=True)

    # Set units
    LNOx.units = 'molec./cm^2'
    LNOx_pickering.units = 'molec./cm^2'

    if kind == 'flash':
        Flashes.units = 'kiloFlashes'
    else:
        Strokes.units = 'kiloStrokes'

    # Write data
    CRF[:]               = CRF_bin
    CP[:]                = CP_bin
    AMFLNOx[:]           = AMFLNOx_bin
    AMFLNOx_pickering[:] = AMFLNOx_pickering_bin
    LNOx[:]              = LNOx_bin
    LNOx_pickering[:]    = LNOx_pickering_bin

    if kind == 'flash':
        Flashes[:] = TL_bin
    else:
        Strokes[:] = TL_bin


def main(behr_file, entln_file, date_str,
        north, south, west, east, 
        CRF_threshold, CF_threshold, flash_threshold, 
        stroke_threshold, min_pixels, t_window, debug):
    # Get bins of lon/lat
    bin_lon = np.arange(west, east, 1)
    bin_lat = np.arange(south, north, 1)

    # Set savefile name and group
    if ntpath.basename(entln_file).startswith('LtgFlashPortions'):
        kind = 'stroke'
        threshold = stroke_threshold
    else:
        kind = 'flash'
        threshold = flash_threshold

    name = 'omilnox_5pixel_entln'\
            +'_crf'+str(int(CRF_threshold*100))+'_cf'+str(int(CF_threshold*100))\
            +'_'+'threshold'+str(int(threshold*1000))+'_'+ntpath.basename(behr_file)[-12:-8]

    # Save .nc files
    if not os.path.isfile(save_dir+name+'.nc'):
        # Lon/lat variable is universe, just need to save once.
        save_nc = Dataset(save_dir+name+'.nc', 'w', format='NETCDF4')
        global lon_center, lat_center
        lon_center = bin_lon[:-1]+0.5
        lat_center = bin_lat[:-1]+0.5
        write_geo(save_nc, lon_center, lat_center)
    else:
        save_nc = Dataset(save_dir+name+'.nc', 'r+', format='NETCDF4')

    # Read BEHR file by swath
    print ('Reading BEHR_'+ntpath.basename(behr_file)[-12:-4]+' data for '+kind+' data'+'...')

    f = h5py.File(behr_file,'r')
    swaths = list(f['Data'])

    for swath in swaths:
        # Read BEHR and ENTLN data
        if debug > 0:
            print ('    Reading swath',swath)

        times, lon, lat, valid_pixels, CRF_bin, CP_bin, AMFLNOx_bin, AMFLNOx_pickering_bin, LNOx_bin, LNOx_pickering_bin = \
            read_behr_swath(f, swath, bin_lon, bin_lat, CRF_threshold, CF_threshold, t_window)

        if debug > 0:
            print ('    Reading ENTLN '+kind+'data ...')

        if not np.any(times):
            CG_bin = np.zeros((bin_lon.shape[0]-1, bin_lat.shape[0]-1))
            IC_bin = np.zeros((bin_lon.shape[0]-1, bin_lat.shape[0]-1))
            TL_bin = CG_bin + IC_bin
        else:
            CG_bin, IC_bin = read_entln(entln_file, times, t_window, lon, lat, bin_lon, bin_lat)
            TL_bin = CG_bin + IC_bin

            # Filter_3
            # total flashes(strokes) per grid box and moles of LNOx per grid box
            # Values set to zero in grid boxes where flash(pulse) or CRF threshold is not met
            cond = (TL_bin < threshold) | (valid_pixels < min_pixels)
            TL_bin[cond], CRF_bin[cond], CP_bin[cond], AMFLNOx_bin[cond], AMFLNOx_pickering_bin[cond], LNOx_bin[cond], LNOx_pickering_bin[cond] = [0.]*7

        # Save data to nc file
        if debug > 0:
            print ('    Save swath', swath)

        write_data(kind, date_str, save_nc, swath, lon_center, lat_center, \
                CRF_bin, CP_bin, AMFLNOx_bin, AMFLNOx_pickering_bin, LNOx_bin, LNOx_pickering_bin, TL_bin)

    save_nc.close()
    f.close()


if __name__ == '__main__':
    args = parse_args()

    behr_files = [behr_dir+behr_file for behr_file in os.listdir(behr_dir) if fnmatch.fnmatch(behr_file, 'OMI_BEHR-DAILY_US_v3-0B*.hdf')]
    behr_files.sort()

    for behr_file in behr_files:
        # Get date string
        date_str    = behr_file[-12:-4]

        # Get entln filename
        flash_file  = entln_dir+'LtgFlash'+date_str+'.csv'
        pulse_file  = entln_dir+'LtgFlashPortions'+date_str+'.csv'
        entln_files = [flash_file, pulse_file]

        # Process BEHR and ENTLN data
        for entln_file in entln_files:
            main(behr_file, entln_file, date_str, **args)