# BEHR_OMINO2 Guide for non_Berkeley

## Clone [BEHR_Core](https://github.com/CohenBerkeleyLab/BEHR-core.git) and other repositories

```
mkdir BEHR
cd BEHR
git clone -b develop --single-branch https://github.com/CohenBerkeleyLab/BEHR-core.git
cd BEHR-core
./setup.sh --develop
```
### Structure of repositories:

```
├── BEHR
│   ├── BEHR-core
│   ├── BEHR-core-utils
│   ├── BEHR-PSM-Gridding
│   ├── Matlab-Gen-Utils
│   ├── MatlabPythonInterface
│   └── WRF_Utils
```

If you don't have permission to change Matlab path, you should add BEHR to Matlab path every time:

`addpath(genpath('your_BEHR_path'))`

## Create behr_paths.m

1. Run `BEHR_initial_setup` in Matlab;
2. You will find *behr_paths.m* in BEHR-core-utils/Utils/Constants;
3. Change paths in behr_paths.m by yourself. Check that all the paths listed in behr_paths.m are
   correct. You can validate them by running behr_paths.ValidatePaths() in Matlab, it should return 1 (0 means at least one path is invalid).

### Note

1. **sp_mat_dir** should have subdirectories for each region to be produced (e.g. "us" - must be lower case)
2. **behr_mat_dir** should have subdirectories for each region to be produced and within each region directories "daily" and "monthly".
3. **myd06_dir** should contain folders for each year with MYD06_L2 files in them.
4. **mcd43d_dir** should contain folders for each year with MCD43D* files in them.
5. **wrf_profiles** it assumes that WRF files are organized in a directory structure behr_paths.wrf_profiles/<region>/<yyyy>/<mm> where <region> is the region being retrieved (passed as the "region" input), <yyyy> is the four-digit year and <mm> is the two-digit month. If you specify wrf_output_path explicitly, then it assumes all WRF output files are in that directory.

## Download data

Check Section 4.3 of BEHR_Read_me file.

### Download url
**GLOBE database**

https://www.ngdc.noaa.gov/mgg/topo/gltiles.html , https://www.ngdc.noaa.gov/mgg/topo/elev/esri/hdr/

**Land_Water_Mask_7Classes_UMD.hdf**

ftp://rsftp.eeos.umb.edu/data02/Gapfilled/

**OMNO2 and OMPIXCOR**

https://mirador.gsfc.nasa.gov/

**MCD43D07, MCD43D08, MCD43D09 and MCD43D31**

https://e4ftl01.cr.usgs.gov/MOTA/

**MYD06_L2**

https://search.earthdata.nasa.gov/

**sortscript.sh**

https://github.com/CohenBerkeleyLab/OtherSatelliteUtils/blob/master/OMI%20Utils/DOMINO/sortscript.sh

Modify sortscript.sh:

```
For OMNO2:
    36     y=${fname:18:4}
    37     m=${fname:23:2}
For OMPIXCOR:
    36     y=${fname:21:4}
    37     m=${fname:26:2}
```

### Structure example of data files:

```
├── GLOBE_Database
│   ├── a10g
│   ├── a10g.hdr
│   ├── ...........
│   ├── p10g
│   └── p10g.hdr
├── MODIS
│   ├── Land_Water_Mask_7Classes_UMD.hdf
│   ├── MCD43D
│   │   └── 2017
│   └── MYD06_L2
│       └── 2017
├── OMI
│   ├── OMNO2
│   │   └── version_3_3_0
│   │       ├── 2017
│   │       │   └── 04
│   │       └── download_staging
│   │           ├── download
│   │           └── sortscript.sh
│   └── OMPIXCOR
│       └── version_003
│           ├── 2017
│           │   └── 04
│           └── download_staging
│               ├── download
│               └── sortscript.sh
```

## Read data

1. Set date_start, date_end, region and DEBUG_LEVEL in `*/BEHR/BEHR-core/Read_Data/read_main.m`;
2. Run `read_main.m`
3. Check SP_Files.

## Recalculate AMF and Tropospheric Column

1. Initialize parameters of BEHR_main.m
2. [Set python environment](https://github.com/zxdawn/BEHR-PSM-Gridding) for BEHR-PSM-Gridding
3. Run `BEHR_main.m`

## Reference

[BEHR_Readme.pdf](https://github.com/CohenBerkeleyLab/BEHR-core/tree/develop/Documentation)