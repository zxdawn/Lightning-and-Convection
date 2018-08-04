# Calculate LNOx based on BEHR and ENTLN data
#
# Filter conditions:
    # valid pixels >= 5 for each 1*1 grid;
    # CRF >= CRF_threshold for each OMI pixel;
    # CF  >= CF_threshold for each OMI pixel;
    # Flashes >= flashthreshold for 1*1 grid 3h before OMI passtime;
    # Strokes >= strokethreshold for 1*1 grid 3h before OMI passtime
#
# Parameters:
#   north, south, west and east
#   CRF_threshold, CF_threshold, flashthreshold, strokethreshold and min_pixels
#   directory of BEHR and ENTLN data
#
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
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

default_vals =  {'north': 50.5, 'south': 21.5,
                'west': -110.5, 'east': -76.5,
                'CRF_threshold': 0.7, 'CF_threshold': 0.4,
                'flashthreshold': 3, 'strokethreshold': 3,
                'min_pixels': 5}

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
    parser.add_argument('--flashthreshold', default = default_vals['flashthreshold'], type = float, help = 'flashthreshold')
    parser.add_argument('--strokethreshold', default = default_vals['strokethreshold'], type = float, help = 'strokethreshold')
    parser.add_argument('--min_pixels', default = default_vals['min_pixels'], type = float, help = 'min_pixels')

    args = parser.parse_args()

    return vars(args)


def read_entln(filename, sdate, edate, bin_lon, bin_lat):
    df = pd.read_csv(filename, delimiter=',')

    # Filter 3h before OMI passtime
    date = pd.to_datetime(df['timestamp'])
    mask = (sdate <= date) & (date <= edate)

    # Get CG and IC flashes/pulses
    type = df['type'].loc[mask]
    CG = type[type == 0 | 40]
    IC = type[type == 1]

    # Get lon and lat
    lon_CG = df['longitude'].loc[mask][type == 0 | 40]
    lat_CG = df['latitude'].loc[mask][type == 0 | 40]
    lon_IC = df['longitude'].loc[mask][type == 1]
    lat_IC = df['latitude'].loc[mask][type == 1]

    CG_bin = stats.binned_statistic_2d(lon_CG, lat_CG, CG, \
                'sum', bins=[bin_lon,bin_lat]).statistic/1000 #kFlashes(kpulses)
    IC_bin = stats.binned_statistic_2d(lon_IC, lat_IC, IC, \
                'sum', bins=[bin_lon,bin_lat]).statistic/1000 #kFlashes(kpulses)

    return CG_bin, IC_bin


def read_behr_swath(f, swath, bin_lon, bin_lat, CRF_threshold, CF_threshold):
    # Read BEHR variables
    T    = f['Data/'+swath+'/Time'][:]
    lon  = f['Data/'+swath+'/Longitude'][:]
    lat  = f['Data/'+swath+'/Latitude'][:]
    CRF  = f['Data/'+swath+'/CloudRadianceFraction'][:]
    CF   = f['Data/'+swath+'/CloudFraction'][:]
    LNOx = f['Data/'+swath+'/BEHRColumnAmountLNOxTrop'][:]
    LNOx_pickering = f['Data/'+swath+'/BEHRColumnAmountLNOxTrop_pickering'][:]

    # Get OMI passtime
    ref_time = datetime(1993, 1, 1)
    T_mean = np.mean(T[0])
    edate = ref_time + timedelta(seconds=float(T_mean))
    sdate = edate - timedelta(hours=3)

    # Filter_1: CRF and CF
    filter_CRF = CRF < CRF_threshold
    filter_CF  =  CF < CF_threshold
    filter = filter_CRF | filter_CF
    LNOx[filter] = 0.
    LNOx_pickering[filter] = 0.

    lon_1D = lon.ravel(); lat_1D = lat.ravel(); LNOx_1D = LNOx.ravel(); LNOx_pickering_1D = LNOx_pickering.ravel()

    # Filter_2: pixels meet Filter_1 condtion
    valid_pixels = stats.binned_statistic_2d(lon_1D, lat_1D, LNOx_1D, \
                        statistic=lambda LNOx_1D: np.count_nonzero(LNOx_1D), \
                        bins=[bin_lon,bin_lat]).statistic

    # Bin LNOx
    LNOx_bin = stats.binned_statistic_2d(lon_1D, lat_1D, LNOx_1D, \
                    # statistic=lambda LNOx_1D: np.nansum(LNOx_1D), \
                    'sum',bins=[bin_lon,bin_lat]).statistic

    LNOx_pickering_bin = stats.binned_statistic_2d(lon_1D, lat_1D, LNOx_pickering_1D, \
                    'sum',bins=[bin_lon,bin_lat]).statistic

    return sdate, edate, valid_pixels, LNOx_bin, LNOx_pickering_bin


def write_nc(name, save_nc, swath, bin_lon, bin_lat, LNOx_bin, LNOx_pickering_bin, TL_bin):
    # Create group
    swathgrp = save_nc.createGroup(swath)

    # Get lon/lat
    lon_center = bin_lon[:-1]+0.5
    lat_center = bin_lat[:-1]+0.5

    # Create dimension
    lon = swathgrp.createDimension('lon', lon_center.shape[0])
    lat = swathgrp.createDimension('lat', lat_center.shape[0])

    # Create variables
    latitudes      = swathgrp.createVariable('Latitude', 'f4',('lat',))
    longitudes     = swathgrp.createVariable('Longitude', 'f4',('lon',))
    LNOx           = swathgrp.createVariable('LNOx', 'f4', ('lon','lat'), fill_value=0.)
    LNOx_pickering = swathgrp.createVariable('LNOx_pickering', 'f4', ('lon','lat'), fill_value=0.)

    if name.split('_')[2] == 'entlnflash':
        Flashes = swathgrp.createVariable('Flashes', 'f4', ('lon','lat'), fill_value=0.)
    else:
        Strokes = swathgrp.createVariable('Strokes', 'f4', ('lon','lat'), fill_value=0.)

    # Set units
    latitudes.units = 'degree_north'
    longitudes.units = 'degree_east'
    LNOx.units = 'molec./cm^2'
    LNOx_pickering.units = 'molec./cm^2'

    if name.split('_')[2] == 'entlnflash':
        Flashes.units = 'kiloFlashes'
    else:
        Strokes.units = 'kiloStrokes'

    # Write data
    latitudes[:]      = lat_center
    longitudes[:]     = lon_center
    LNOx[:]           = LNOx_bin
    LNOx_pickering[:] = LNOx_pickering_bin

    if name.split('_')[2] == 'entlnflash':
        Flashes[:] = TL_bin
    else:
        Strokes[:] = TL_bin


def process_data(behr_file, entln_file, bin_lon, bin_lat, Lon_min, Lon_max, Lat_min, Lat_max,
        CRF_threshold, CF_threshold, flashthreshold, strokethreshold, min_pixels):
    # Set savefile name and group
    if ntpath.basename(entln_file).startswith('LtgFlashPortions'):
        kind = 'stroke'
        threshold = strokethreshold
    else:
        kind = 'flash'
        threshold = flashthreshold

    name = 'omilnox_5pixel_entln'+kind\
            +'_crf'+str(int(CRF_threshold*100))+'_cf'+str(int(CF_threshold*100))\
            +'_'+kind+'threshold'+str(int(threshold*100))+'_'+ntpath.basename(behr_file)[-12:-4]

    save_nc = Dataset(name+'.nc', 'w', format='NETCDF4')

    # Read BEHR file by swath
    print ('Reading BEHR data ...')
    f = h5py.File(behr_file,'r')
    swaths = list(f['Data'])

    for swath in swaths:
        # Read BEHR and ENTLN data
        print ('    Reading swath',swath)
        sdate, edate, valid_pixels, LNOx_bin, LNOx_pickering_bin = read_behr_swath(f, swath, bin_lon, bin_lat, CRF_threshold, CF_threshold)
        print ('Reading ENTLN data ...')
        CG_bin, IC_bin = read_entln(entln_file, sdate, edate, bin_lon, bin_lat)
        TL_bin = CG_bin + IC_bin

        # Filter
        # total flashes(pulses) per grid box and moles of LNOx per grid box
        # Values set to zero in grid boxes where flash(pulse) or CRF threshold is not met
        cond = (TL_bin < threshold) | (valid_pixels < min_pixels)
        TL_bin[cond] = 0.; LNOx_bin[cond] = 0.; LNOx_pickering_bin = 0.

        # Save to nc file
        print ('Save swath', swath)
        write_nc(name, save_nc, swath, bin_lon, bin_lat, LNOx_bin, LNOx_pickering_bin, TL_bin)


def main(entln_file, north, south, west, east, 
        CRF_threshold, CF_threshold, flashthreshold, 
        strokethreshold, min_pixels):
    bin_lon  = np.arange(west, east, 1)
    bin_lat  = np.arange(south, north, 1)

    process_data(behr_file, entln_file, bin_lon, bin_lat, west, east, south, north,\
        CRF_threshold, CF_threshold, flashthreshold, strokethreshold, min_pixels)


if __name__ == '__main__':
    # Set directory of BEHR and ENTLN data
    # Read arguments
    behr_dir  = '/public/home/zhangxin/bigdata/BEHR_data/'
    entln_dir = '/public/home/zhangxin/bigdata/ENTLN_data/'
    args = parse_args()

    for behr_file in os.listdir(behr_dir):
        if fnmatch.fnmatch(behr_file, 'OMI_BEHR-DAILY_US_v3-0B*.hdf'):
            # Get date string
            date_str    = behr_file[-12:-4]

            # Get entln filename
            flash_file  = entln_dir+'LtgFlash'+date_str+'.csv'
            pulse_file  = entln_dir+'LtgFlashPortions'+date_str+'.csv'
            entln_files = [flash_file, pulse_file]

            # Process BEHR and ENTLN data
            behr_file = behr_dir + behr_file
            for entln_file in entln_files:
                main(entln_file, **args)