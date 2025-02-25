"""
Recipe for gap-filling MOSAiC Met City data with advection-corrected ASFS.
"""

import os
import json
import datetime
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

# Load mdf_toolkit
# if this is first time running, go to the root of the mdf-toolkit directory 
# and run ```pip install -e .``` first
from mdftoolkit.MDF_toolkit import MDF_toolkit as MDF

'''
The archived met data are available in daily files, loading so many
files in python can be prohibitively slow. So before running this
script, we need to concatenate the daily files for each datasource.
Within the directories containing their respective data, run the
following nco (https://nco.sourceforge.net/) commands:
metcity:
ncrcat -v base_time,time_offset,temp_2m,rh_2m,atmos_pressure_2m,wspd_u_mean_2m,wspd_v_mean_2m,down_long_hemisp,down_short_hemisp,zenith_true,zenith_apparent,lat_tower,lon_tower -t 4 mosmet.metcity.level3.4.1min.20??????.000000.nc mosmet.metcity.level3.4.1min.mdfsubset.nc

asfs50
ncrcat -v base_time,time_offset,temp,rh,atmos_pressure,wspd_u_mean,wspd_v_mean,down_long_hemisp,down_short_hemisp,zenith_true,zenith_apparent,lat,lon -t 4 mosmet.asfs50.level3.4.1min.20??????.000000.nc mosmet.asfs50.level3.4.1min.mdfsubset.nc

asfs40
ncrcat -v base_time,time_offset,temp,rh,atmos_pressure,wspd_u_mean,wspd_v_mean,down_long_hemisp,down_short_hemisp,zenith_true,zenith_apparent,lat,lon -t 4 mosmet.asfs40.level3.4.1min.20??????.000000.nc mosmet.asfs40.level3.4.1min.mdfsubset.nc

asfs30
ncrcat -v base_time,time_offset,temp,rh,atmos_pressure,wspd_u_mean,wspd_v_mean,down_long_hemisp,down_short_hemisp,zenith_true,zenith_apparent,lat,lon -t 4 mosmet.asfs30.level3.4.1min.20??????.000000.nc mosmet.asfs30.level3.4.1min.mdfsubset.nc

For shiprad data, we first need to average the secondly data to minutely
This is easiest to do on ARM's jupyterhub. See the notebook
'preprocess_shiprad.ipynb' or use the preprocessed data in the archive.
'''

## Parameters
# Specify which dates belong to Drift Track 1 and Drift Track 2
end_dt_drift_1 = pd.to_datetime("2020-07-31 12:00:00", utc=False)
start_dt_drift_2 = pd.to_datetime("2020-08-27 12:00:00", utc=False)
# Parameters that control bias correction and filling
overlap_minutes = 180 # 3 hour overlap
relaxation_minutes = 5 # so that almost all transition is in 15 min.
# Residual gaps in radiation after filling with ASFS that are less than
# this threshold are interpolated
rad_interp_thrhld_minutes = 15 

# Specify which variables correspond to MDF var names
var_map_dict = {'lat'   : 'lat_tower',
                'lon'   : 'lon_tower',
                'uas'   : 'wspd_u_mean_2m',
                'vas'   : 'wspd_v_mean_2m',
                'tas'   : 'temp_2m',
                'hus'   : 'specific_humidity_2m',
                'rlds'  : 'down_long_hemisp',
                'rsds'  : 'down_short_hemisp',
                }
# Specify variable correspondence between Met City and ASFS
varname_dict = {'lat_tower': 'lat',
                'lon_tower': 'lon',
                'temp_2m': 'temp', 
                'rh_2m': 'rh', 
                'atmos_pressure_2m': 'atmos_pressure',
                'wspd_u_mean_2m': 'wspd_u_mean', 
                'wspd_v_mean_2m': 'wspd_v_mean',
                'down_long_hemisp': 'down_long_hemisp', 
                'down_short_hemisp': 'down_short_hemisp',
                'mixing_ratio_2m': None,
                'zenith_true': 'zenith_true'}


# Load paths
with open(os.path.join('..', 'utils', 'paths.json')) as fp:
    paths_dict = json.load(fp)

## Functions
def find_gap_edge(da, start, end, first=True):
    """
    Convenience function for finding the timestamp of a gap start/end

    Parameters:
    -----------
    da: xarry DataArray
        DataArray to find gap in
    start: str or datetime.datetime
        Time to start search
    end: str or datetime.datetime
        Time to end search
    first: bool, optional
        If true return the first missing time, if False return the last one

    Returns:
    --------
    Timestamp

    """

    if type(start) is pd.Timestamp:
        dt_start = start
    else:
        dt_start = datetime.datetime.fromisoformat(start)
    if type(end) is pd.Timestamp:
        dt_end = end
    else:
        dt_end = datetime.datetime.fromisoformat(end)

    da_search = da.sel(time=slice(dt_start, dt_end))
    tup_na = np.nonzero(da_search.isnull().values)
    if first:
        return da_search.get_index('time')[tup_na[0][0]]
    else:
        return da_search.get_index('time')[tup_na[0][-1]]

# Function for gap filling
def gap_fill(param, fill_source, gap_start, gap_end, offset_seconds, 
             ds_fill, source_dict, overlap_minutes, relaxation_minutes,
             varname_dict, primary_source='metcity', plot=False, 
             plotpath=None):
    """
    Fill a gap for an individual parameter.
    
    Parameters:
    -----------
    param: hashable
        Name of the parameter in Met City data to fill.
    fill_source: hashable
        Source to fill from, must match a key in source_dict.
    gap_start: str
        Timestamp (ISO format) of first missing data point in gap.
    gap_end: str
        Timestamp (ISO format) of the last missing data point in gap.
    offset_seconds: numeric
        How many seconds to shift filling record to match gap.
    ds_fill: xarray dataset
        Dataset to fill gap in (Met City data).
    source_dict: dict
        Dict containing datasets to fill gap from (e.g., ASFS's).
    overlap_minutes: numeric
        How many minutes before and after gap to use for bias correction.
    relaxation_minutes: numeric
        Exponential decay constant for relaxing fill values to Met City data.
    varname_dict: dict
        Mapping from variable names in Met City data to other datasets.
    primary_source: hashable, optional
        Key for the primary datasource (typically 'metcity')
    plot: bool, optional
        Whether or not to generate a plot for this gap
    pltpath: str, optional
        Where to save figures to, optional, if None do not write figures.
        
    Returns:
    --------
    xarray DataSet with the gap filled.

    Raises:
    -------
    Warns if missing values at edges of gap

    """

    # Get time periods
    td_buffer = datetime.timedelta(minutes=overlap_minutes)
    dt_gap_start = gap_start.to_pydatetime() #datetime.datetime.fromisoformat(gap_start)
    dt_gap_end = gap_end.to_pydatetime() #datetime.datetime.fromisoformat(gap_end)
    dt_buffer_start = dt_gap_start - td_buffer
    dt_buffer_end = dt_gap_end + td_buffer

    # Get shifted asfs dataset
    # convert seconds to nearest minute
    td_offset = datetime.timedelta(minutes=round(offset_seconds/60.0))
    np_offset = np.timedelta64(td_offset)
    da_source = source_dict[fill_source].sel(
        time=slice(dt_buffer_start - td_offset, 
                dt_buffer_end - td_offset))[varname_dict[param]].copy(deep=True)
    new_time = da_source.time.values + np_offset
    da_source['time'] = new_time

    # Bias Correction
    bias = (source_dict[primary_source].sel(time=slice(dt_buffer_start, 
                                                          dt_buffer_end)
                                              )[param] 
                - da_source).mean(skipna=True).values
    da_source += bias

    # Relaxation
    da_gap_fill = da_source.sel(time=slice(dt_gap_start, dt_gap_end))
    da_gap_fill.name = param
    length = da_gap_fill.size
    minutes_forward = np.arange(length) + 1
    minutes_reverse = minutes_forward[::-1]
    try:
        start_diff = (source_dict[primary_source][param].sel(
            time=dt_gap_start - datetime.timedelta(minutes=1)).values
                    - da_source.sel(time=dt_gap_start - 
                                    datetime.timedelta(minutes=1)).values)
    except:
        start_diff = np.NaN
    try:
        end_diff = (source_dict[primary_source][param].sel(
            time=dt_gap_end + datetime.timedelta(minutes=1)).values
                    - da_source.sel(time=dt_gap_end +
                                    datetime.timedelta(minutes=1)).values)
    except KeyError:
        end_diff = np.NaN
    if not np.isnan(start_diff):
        da_gap_fill += start_diff * np.exp(
            -1*minutes_forward/relaxation_minutes)
    else:
        warnings.warn("Missing start value in " + param + " at " 
                      + dt_gap_start.strftime('%Y%m%d%H%M'))        
    if not np.isnan(end_diff):
        da_gap_fill += end_diff * np.exp(
            -1*minutes_reverse/relaxation_minutes)
    else:
        warnings.warn("Missing end value in " + param + " at " 
                      + dt_gap_end.strftime('%Y%m%d%H%M'))
    
    # Gap filling
    arr_src = np.full(da_gap_fill.size, fill_source, dtype='object')
    da_gap_src = xr.DataArray(arr_src, coords={'time':da_gap_fill.time},
                              name=param + '_src')
    ds_gap_fill = xr.merge([da_gap_fill, da_gap_src])

    # Update dataset with filled values
    ds_fill_var = ds_gap_fill.combine_first(ds_fill[[param, 
                                                     param+'_src']])
    ds_fill.update(ds_fill_var)

    # Plot if desired
    if plot:
        f, ax = plt.subplots(1, 1, figsize=(8,6))
        ds_fill.sel(time=slice(dt_buffer_start, dt_buffer_end))[param].plot(
            ax=ax, label='Filled')
        try:
            source_dict[fill_source].sel(time=slice(dt_buffer_start - td_offset, 
                    dt_buffer_end - td_offset))[varname_dict[param]].plot(
                    ax=ax, label=fill_source)
        except TypeError:
            pass
        try:
            source_dict[primary_source].sel(time=slice(dt_buffer_start, 
                                                            dt_buffer_end)
                                            )[param].plot(ax=ax, label='MetCity')
        except TypeError:
            pass
        ax.legend()
        # write output if requested
        if plotpath is not None:
            fname = (dt_gap_start.strftime('%Y%m%d%H%M') + '_' +
                     dt_gap_end.strftime('%Y%m%d%H%M') + '_' +
                     param)
            f.savefig(os.path.join(plotpath, fname))
            plt.close(f)
    return ds_fill

def find_gaps(ds, param):
    """
    Convenience function for finding gaps in a dataset

    Parameters:
    -----------
    ds: xarray Dataset
        The dataset to find the gaps in.
    param: string
        Name of the variable to find gaps in.
    
    Returns:
    --------
    numpy array Nx2 where each row is a gap
    """

    da = ds[param]
    gaps = []
    in_gap = False
    gap = np.zeros(2, dtype='datetime64[ns]')
    last_time = 0
    for row in da:
        if np.isnan(row.item()):
            if not in_gap:
                gap[0] = row.time.item()
                in_gap = True
        else:
            if in_gap:
                gap[1] = last_time
                in_gap = False
                gaps.append(deepcopy(gap))
        last_time = row.time.item()
    gaps_np = np.array(gaps)

    # Need to eliminate times that the ship was not present
    end_co1 = np.datetime64("2020-05-16")
    start_co2 = np.datetime64("2020-06-19")
    end_co2 = np.datetime64("2020-07-31")
    start_co3 = np.datetime64("2020-08-21")
    gaps_co1 = gaps_np[gaps_np[:,1]<=end_co1,:]
    gaps_co2 = gaps_np[np.logical_and(gaps_np[:,0]>start_co2, gaps_np[:,1]<=end_co2),:]
    gaps_co3 = gaps_np[gaps_np[:,0]>start_co3,:]
    gaps_co = np.concatenate([gaps_co1, gaps_co2, gaps_co3])

    return gaps_co

def interpolate_gap(ds_fill, param, gap_start, gap_end):
    """
    Convenience function for interpolating a gap in a dataset

    Parameters:
    -----------
    ds_fill: xarray Dataset
        The dataset to interpolate the gap in.
    param: string
        Name of the variable to interpolate.
    gap_start: np.datetime64
        Start of the gap (first missing value)
    gap_end: np.datetime64
        End of the gap (last missing value)

    Returns:
    --------
    xarray Dataset with the gap interpolated.
    """

    # Get dataset for gap and time points before and after
    ds_gap_fill = ds_fill.sel(
        time=slice(gap_start - np.timedelta64(1, 'm'), 
                   gap_end + np.timedelta64(1, 'm'))
        )[[param, param+'_src']]
    
    # Interpolate and update _src
    ds_gap_fill[param] = ds_gap_fill[param].interpolate_na(dim='time')
    ds_gap_fill[param+'_src'].sel(
        time=slice(gap_start, gap_end)).values[:] = 'interpolated'
    
    # Update dataset with filled values
    ds_fill_var = ds_gap_fill.combine_first(ds_fill[[param, 
                                                     param+'_src']])
    ds_fill.update(ds_fill_var)

    return ds_fill

def calc_mix_ratio(temp, RHw, press):
    """
    Function for calculating mixing ratio from C. Cox

    !!! Temperature must be in Kelvin !!!


    Calculations based on Appendix B of the PTU/HMT manual to be mathematically consistent with the
    derivations in the on onboard electronics. Checked against Ola's code and found acceptable
    agreement (<0.1% in MR). RHi calculation is then made following Hyland & Wexler (1983), which
    yields slightly higher (<1%) compared a different method of Ola's
    """
    
    # calculate saturation vapor pressure (Pws) using two equations sets, Wexler (1976) eq 5 & coefficients
    c0    = 0.4931358
    c1    = -0.46094296*1e-2
    c2    = 0.13746454*1e-4
    c3    = -0.12743214*1e-7
    omega = temp - ( c0*temp**0 + c1*temp**1 + c2*temp**2 + c3*temp**3 )

    # eq 6 & coefficients
    bm1 = -0.58002206*1e4
    b0  = 0.13914993*1e1
    b1  = -0.48640239*1e-1
    b2  = 0.41764768*1e-4
    b3  = -0.14452093*1e-7
    b4  = 6.5459673
    Pws = np.exp( ( bm1*omega**-1 + b0*omega**0 + b1*omega**1 + b2*omega**2 + 
                   b3*omega**3 ) + b4*np.log(omega) ) # [Pa]

    Pw = RHw*Pws/100 # # actual vapor pressure (Pw), eq. 7, [Pa]

    x = 1000*0.622*Pw/((press*100)-Pw) # mixing ratio by weight (eq 2), [g/kg]

    return x

## Load Met City data, create dataset filled for filled data
# Load Met City data
ds_metcity = xr.open_dataset(os.path.join(paths_dict['raw_data'], 
    'metcity', 'mosmet.metcity.level3.4.1min.mdfsubset.nc'),
    engine='netcdf4')
# Create fixed frequency time index: adds na's where no file is present
dti = pd.date_range(start=ds_metcity.indexes['time'][0],
                    end=ds_metcity.indexes['time'][-1], freq='min')
ds_metcity = ds_metcity.reindex(indexers={'time': dti})
# Create dataset for filled values
ds_fill = ds_metcity.copy(deep=True)
# Add variables for the source of filled parameters
arr_src = np.full(ds_fill.dims['time'], 'metcity', dtype='object')
for key in varname_dict.keys():
    ds_fill[key + '_src'] =('time',arr_src)

## Create dataframe to guide which gaps get filled
# Load gap filling guide
df_gap_fill = pd.read_excel("gap_fill_met_city.xlsx", parse_dates=[1, 2])

# Find gap edges and update df_gap_fill_mod if needed
df_gap_fill_mod = df_gap_fill.copy(deep=True)
buffer = pd.Timedelta(minutes=10)
for i in df_gap_fill_mod.index:
    found_edge = False
    tbl_start = df_gap_fill_mod.at[i,'gap_start']
    while found_edge is False:
        search_start = tbl_start - buffer
        search_end = tbl_start + buffer
        edge_start = find_gap_edge(ds_metcity[df_gap_fill_mod.at[i,'param']], 
                                start=search_start, end=search_end,
                                first=True)
        # If we haven't found the edge...
        if search_start==edge_start:
            tbl_start = tbl_start - buffer
        else:
            found_edge=True
            df_gap_fill_mod.at[i,'gap_start'] = edge_start
    found_edge = False
    tbl_end = df_gap_fill_mod.at[i,'gap_end']
    while found_edge is False:
        search_start = tbl_end - buffer
        search_end = tbl_end + buffer
        edge_end = find_gap_edge(ds_metcity[df_gap_fill_mod.at[i,'param']], 
                                start=search_start, end=search_end,
                                first=False)
        # If we haven't found the edge...
        if search_end==edge_end:
            tbl_end = tbl_end + buffer
        else:
            found_edge=True
            df_gap_fill_mod.at[i,'gap_end'] = edge_end

# Dict to contain all data sources used for standard gap filling
source_dict = {'metcity': ds_metcity}

## Fill gaps with ASFS data
# Load ASFS data
for asfs_name in ['asfs30', 'asfs40', 'asfs50']:
    source_dict[asfs_name] = xr.open_dataset(os.path.join(
        paths_dict['raw_data'], asfs_name, 
        'mosmet.'+asfs_name+'.level3.4.1min.mdfsubset.nc'),
        engine='netcdf4')

# Fill identified gaps with ASFS data
for index, row in df_gap_fill_mod.iterrows():
    ds_fill = gap_fill(row['param'], row['station'], 
                        row['gap_start'], row['gap_end'], 
                        row['offset'], ds_fill, source_dict, 
                        overlap_minutes, relaxation_minutes, 
                        varname_dict, primary_source='metcity', 
                        plot=False, plotpath='.')

## Finish filling radiation gaps

# Gap filling in shortwave radiation based on C. Cox analysis
# Using values provided by C. Cox
rsds_ccox = pd.read_excel("rsds_ccox_analysis.xlsx", parse_dates=[0, 1])
for index, row in rsds_ccox.iterrows():
    dti = pd.date_range(start=row['START'], end=row['END'], freq='min')
    if type(row['RSDS'])==float:
        arr_dat = np.array([row['RSDS']])
    else:
        arr_dat = np.array(row['RSDS'].split()).astype(float)
    arr_src = np.full(dti.size, 'cox_analysis', dtype='object')
    ds_gap = xr.Dataset({'down_short_hemisp': ('time', arr_dat),
                         'down_short_hemisp_src': ('time', arr_src)},
                           coords={'time': dti})
    ds_fill_var = ds_gap.combine_first(ds_fill[['down_short_hemisp', 
                                                'down_short_hemisp_src']])
    ds_fill.update(ds_fill_var)

# From 2019-10-15 to 2019-10-24 05:29 Use ASFS40 with 0.64 Wm2
# subtracted from it for times that solar zenith angle > 96 degrees
da_gap_fill = source_dict['asfs40']['down_short_hemisp'].sel(time=
    slice('2019-10-15 00:00:00', '2019-10-24 05:29:00')).copy(deep=True)
da_gap_fill -= 0.64
arr_src = np.full(da_gap_fill.size, 'asfs40', dtype='object')
da_gap_src = xr.DataArray(arr_src, coords={'time':da_gap_fill.time},
                            name='down_short_hemisp_src')
ds_gap_fill = xr.merge([da_gap_fill, da_gap_src])
ds_fill_var = ds_gap_fill.combine_first(ds_fill[['down_short_hemisp', 
                                            'down_short_hemisp_src']])
ds_fill.update(ds_fill_var)

# Anytime the solar zenith angle is below a critical threshold set
# downwelling shortwave to zero, for simplicity let's use the civil twilight
# (sza > 96 degrees) applied to the solar zenith angle
# First we have to produce a continuous record of solar zenith angle
da_sza = ds_fill['zenith_true']
# And interpolate to fill missing values, use linear interpolation
da_sza = da_sza.interpolate_na(dim='time', method='linear')
crit_sza = 96
da_sza.name = 'interp_sza'
ds_fill = xr.merge([ds_fill, da_sza])
ds_fill['down_short_hemisp'][(ds_fill['interp_sza'] >= crit_sza)] = 0.0
ds_fill['down_short_hemisp_src'][(ds_fill['interp_sza'] >= crit_sza)
                                 ] = 'sza'

# 2020-04-14 to 2020-06-01 for residual gaps prioritize ASFS30, then
# ASFS50, then interpolate gaps shorter than threshold, then use shiprad
gaps_sw = find_gaps(ds_fill.sel(time=slice('2020-04-14', '2020-06-01')),
                     'down_short_hemisp')
for i in np.arange(gaps_sw.shape[0]):
    # Check whether gap is complete in ASFS30
    if not np.isnan(source_dict['asfs30']['down_short_hemisp'].sel(
        time=slice(gaps_sw[i,0], gaps_sw[i,1])).values).any():
        ds_fill = gap_fill('down_short_hemisp', 'asfs30', 
            pd.to_datetime(gaps_sw[i,0]), 
            pd.to_datetime(gaps_sw[i,1]), 
            0, ds_fill, source_dict, overlap_minutes, 
            relaxation_minutes, varname_dict, primary_source='metcity',
            plot=False)
    elif not np.isnan(source_dict['asfs50']['down_short_hemisp'].sel(
        time=slice(gaps_sw[i,0], gaps_sw[i,1])).values).any():
        ds_fill = gap_fill('down_short_hemisp', 'asfs50', 
            pd.to_datetime(gaps_sw[i,0]), 
            pd.to_datetime(gaps_sw[i,1]), 
            0, ds_fill, source_dict, overlap_minutes, 
            relaxation_minutes, varname_dict, primary_source='metcity',
            plot=False)

# Gap filling longwave radiation during the November storm from
# C. Cox analysis, here are notes from Chris:
#Two files attached on LWD for Nov 16-26 at MOSAiC.
#*LWDfill: I made an attempt to compile an LWD for the CO based on the tower, ASFS50 and ASFS40. Let me know what you think.
#header: yyyy mm dd HH MM lwd flag
#lwd is the best guess
#flag is the source: 1 = tower, 2 = asfs40, 3 = asfs50. 22 or 33 means that I time-shifted the value by some amount.
#*AERIBT: the AERI brightness temps I mentioned.
#header: yyyy mm dd HH MM bt h
#bt is the brightness temp at a cloud-sensitive microwindow. It will have much larger variance than LWD, but should correlate. A good way to compare to the LWD is to convert the LWD to BT = (LWD/5.67e-8)^0.25
df_novstorm_lwd = pd.read_csv(os.path.join(paths_dict['proc_data'], 
                                           'novstormLWDfill.txt'),
                             sep=",", names=['year', 'month', 'day', 
                            'hour', 'minute', 'down_long_hemisp', 'flag'])
df_novstorm_lwd['time'] = pd.to_datetime(df_novstorm_lwd[[
    'year', 'month','day', 'hour', 'minute']])
def flag_converter(flag):
    if flag == 1:
        return 'metcity'
    elif flag == 2:
        return 'asfs40'
    elif flag == 3:
        return 'asfs50'
    elif flag == 22:
        return 'asfs40'
    elif flag == 33:
        return 'asfs50'
    else:
        return ''
df_novstorm_lwd['down_long_hemisp_src'] = df_novstorm_lwd['flag'].map(
    flag_converter)
ds_gap_fill = df_novstorm_lwd.set_index('time')[['down_long_hemisp', 
                                'down_long_hemisp_src']].to_xarray()
ds_fill_var = ds_gap_fill.combine_first(ds_fill[['down_long_hemisp', 
                                            'down_long_hemisp_src']])
ds_fill.update(ds_fill_var)

# Find remaining gaps in downwelling radiation:
# shortwave
gaps_sw = find_gaps(ds_fill, 'down_short_hemisp')
# longwave
gaps_lw = find_gaps(ds_fill, 'down_long_hemisp')

# For gaps that are longer than the interpolation threshold fill with 
# shiprad
# Compute gap duration
dur_gaps_sw_min = (gaps_sw[:,1] - gaps_sw[:,0]).astype(float)/(1e9*60)
dur_gaps_lw_min = (gaps_lw[:,1] - gaps_lw[:,0]).astype(float)/(1e9*60)
long_gaps_sw = gaps_sw[dur_gaps_sw_min >= rad_interp_thrhld_minutes, :]
long_gaps_lw = gaps_lw[dur_gaps_lw_min >= rad_interp_thrhld_minutes, :]

ds_shipradS1 = xr.open_dataset(os.path.join(paths_dict['raw_data'], 
    'shiprad', 'mosshipradS1.b1.subset.20191015_20200918.nc'), 
    engine='netcdf4')

# Specify variable correspondence between Met City and shiprad
varname_dict_shiprad = {'lat_tower': 'lat',
                        'lon_tower': 'lon',
                        'temp_2m': 'temp', 
                        'down_long_hemisp': 'downwelling_longwave', 
                        'down_short_hemisp': 'spn1_total_corr'}

df_gap_sw = pd.DataFrame({'param': 'down_short_hemisp',
                          'gap_start': long_gaps_sw[:,0],
                          'gap_end': long_gaps_sw[:,1],
                          'station': 'shipradS1',
                          'offset': 0})
df_gap_lw = pd.DataFrame({'param': 'down_long_hemisp',
                          'gap_start': long_gaps_lw[:,0],
                          'gap_end': long_gaps_lw[:,1],
                          'station': 'shipradS1',
                          'offset': 0})
df_gap_shiprad = pd.concat([df_gap_sw, df_gap_lw])

# Fill gaps
source_dict['shipradS1'] = ds_shipradS1
for index, row in df_gap_shiprad.iterrows():
    ds_fill = gap_fill(row['param'], row['station'], row['gap_start'], 
             row['gap_end'], row['offset'], 
             ds_fill, source_dict, overlap_minutes, relaxation_minutes,
             varname_dict_shiprad, primary_source='metcity', plot=False)

# Find remaining gaps and interpolate  short enough ones
# Find remaining gaps in downwelling radiation:
# shortwave
gaps_sw = find_gaps(ds_fill, 'down_short_hemisp')
gaps_lw = find_gaps(ds_fill, 'down_long_hemisp')
dur_gaps_sw_min = (gaps_sw[:,1] - gaps_sw[:,0]).astype(float)/(1e9*60)
dur_gaps_lw_min = (gaps_lw[:,1] - gaps_lw[:,0]).astype(float)/(1e9*60)
short_gaps_sw = gaps_sw[dur_gaps_sw_min < rad_interp_thrhld_minutes, :]
short_gaps_lw = gaps_lw[dur_gaps_lw_min < rad_interp_thrhld_minutes, :]
for i in np.arange(short_gaps_sw.shape[0]):
    ds_fill = interpolate_gap(ds_fill, 'down_short_hemisp', 
                              short_gaps_sw[i,0], short_gaps_sw[i,1])
for i in np.arange(short_gaps_lw.shape[0]):
    ds_fill = interpolate_gap(ds_fill, 'down_long_hemisp', 
                              short_gaps_lw[i,0], short_gaps_lw[i,1])

## Create dataframe for export and compute specific humidity
df_out = ds_fill.to_dataframe()
# Convert air T to Kelvin
df_out['temp_2m'] = df_out['temp_2m'] + 273.15
# Create mixing ratio column
df_out['mixing_ratio_2m'] = calc_mix_ratio(df_out.temp_2m, df_out.rh_2m,
                                           df_out.atmos_pressure_2m)
df_out['specific_humidity_2m'] = (df_out.mixing_ratio_2m/1000)/(
    1 + df_out.mixing_ratio_2m/1000)

# Clip negative shortwave values
df_out['down_short_hemisp'][df_out.down_short_hemisp < 0] = 0.0

# If the value is nan, set _src to nan
def set_src_nan(df, param):
    df[param+'_src'][df[param].isnull()] = np.NaN
set_src_nan(df_out, 'temp_2m')
set_src_nan(df_out, 'rh_2m')
set_src_nan(df_out, 'atmos_pressure_2m')
set_src_nan(df_out, 'wspd_u_mean_2m')
set_src_nan(df_out, 'wspd_v_mean_2m')
set_src_nan(df_out, 'down_long_hemisp')
set_src_nan(df_out, 'down_short_hemisp')

## Convert into MDF
# First drift track (legs 1-4)
df_out_drift_1 = df_out.loc[:end_dt_drift_1]
MDF_out = MDF(supersite_name='MOSAiC_atm_drift1', verbose=True)
global_atts = {
    "title"                    : "MOSAiC surface meteorology",
    "Conventions"              : "MDF, CF (where possible)",
    "standard_name_vocabulary" : "",
    "contributor_name"         : "Christopher J. Cox",
    "contributor_email"        : "christopher.j.cox@noaa.gov",
    "institution"              : "NOAA",
    "creator_name"             : "David Clemens-Sewall",
    "creator_email"            : "david.clemens-sewall@noaa.gov",
    "project"                  : "MOSAiC Model Forcing Dataset working group",
    "summary"                  : "BETA gap filled with in-situ measurements",
    "id"                       : "doi:10.18739/A2RN30916",
    "license"                  : "CC-0", 
    "metadata_link"            : "https://doi.org/10.5281/zenodo.10819496",
    "references"               : ("Continuous observations of the surface " +
                                  "energy budget and meteorology over the " +
                                  "Arctic sea ice during MOSAiC/" +
                                  "https://doi.org/10.1038/s41597-023-02415-5" +
                                  ";  Shipboard Portable Radiation Package/" +
                                  "https://doi.org/10.5439/1437994"),
    "time_coverage_start"      : "{}".format(df_out_drift_1.index[0]),
    "time_coverage_end"        : "{}".format(df_out_drift_1.index[-1]),
    "naming_authority"         : "___", 
    "standard_name_vocabulary" : "___", 
    "keywords"                 : "surface energy budget, arctic, polar",
    "calendar"                 : "standard",
}
MDF_out.update_global_atts(global_atts)  

modf_df = MDF.map_vars_to_mdf(df_out_drift_1, var_map_dict, drop=True)
MDF_out.add_data_timeseries(modf_df, cadence="time01")
MDF_out.write_files(output_dir=paths_dict['proc_data'],
                    fname_only_underscores=True)

# Second drift track (leg 5)
df_out_drift_2 = df_out.loc[start_dt_drift_2:]
MDF_out = MDF(supersite_name='MOSAiC_atm_drift2', verbose=True)
global_atts = {
    "title"                    : "MOSAiC surface meteorology",
    "Conventions"              : "MDF, CF (where possible)",
    "standard_name_vocabulary" : "",
    "contributor_name"         : "Christopher J. Cox",
    "contributor_email"        : "christopher.j.cox@noaa.gov",
    "institution"              : "NOAA",
    "creator_name"             : "David Clemens-Sewall",
    "creator_email"            : "david.clemens-sewall@noaa.gov",
    "project"                  : "MOSAiC Model Forcing Dataset working group",
    "summary"                  : "BETA gap filled with in-situ measurements",
    "id"                       : "doi:10.18739/A2RN30916",
    "license"                  : "CC-0", 
    "metadata_link"            : "https://doi.org/10.5281/zenodo.10819496",
    "references"               : ("Continuous observations of the surface " +
                                  "energy budget and meteorology over the " +
                                  "Arctic sea ice during MOSAiC/" +
                                  "https://doi.org/10.1038/s41597-023-02415-5" +
                                  ";  Shipboard Portable Radiation Package/" +
                                  "https://doi.org/10.5439/1437994"),
    "time_coverage_start"      : "{}".format(df_out_drift_2.index[0]),
    "time_coverage_end"        : "{}".format(df_out_drift_2.index[-1]),
    "naming_authority"         : "___", 
    "standard_name_vocabulary" : "___", 
    "keywords"                 : "surface energy budget, arctic, polar",
    "calendar"                 : "standard",
}
MDF_out.update_global_atts(global_atts)  

modf_df = MDF.map_vars_to_mdf(df_out_drift_2, var_map_dict, drop=True)
MDF_out.add_data_timeseries(modf_df, cadence="time01")
MDF_out.write_files(output_dir=paths_dict['proc_data'],
                    fname_only_underscores=True)


## After here is not needed for creating MDF
# For summaries print amount of data and sources
print('Drift 1:')
print('Air Temperature')
print(df_out_drift_1['temp_2m_src'].value_counts(dropna=False)/df_out_drift_1.shape[0])
print('Atmospheric Pressure')
print(df_out_drift_1['atmos_pressure_2m_src'].value_counts(dropna=False)/df_out_drift_1.shape[0])
print('Relative Humidity')
print(df_out_drift_1['rh_2m_src'].value_counts(dropna=False)/df_out_drift_1.shape[0])
print('Wind Velocity')
print(df_out_drift_1['wspd_u_mean_2m_src'].value_counts(dropna=False)/df_out_drift_1.shape[0])
print('Downwelling Longwave')
print(df_out_drift_1['down_long_hemisp_src'].value_counts(dropna=False)/df_out_drift_1.shape[0])
print('Downwelling Shortwave')
print(df_out_drift_1['down_short_hemisp_src'].value_counts(dropna=False)/df_out_drift_1.shape[0])

print('Drift 2:')
print('Air Temperature')
print(df_out_drift_2['temp_2m_src'].value_counts(dropna=False)/df_out_drift_2.shape[0])
print('Atmospheric Pressure')
print(df_out_drift_2['atmos_pressure_2m_src'].value_counts(dropna=False)/df_out_drift_2.shape[0])
print('Relative Humidity')
print(df_out_drift_2['rh_2m_src'].value_counts(dropna=False)/df_out_drift_2.shape[0])
print('Wind Velocity')
print(df_out_drift_2['wspd_u_mean_2m_src'].value_counts(dropna=False)/df_out_drift_2.shape[0])
print('Downwelling Longwave')
print(df_out_drift_2['down_long_hemisp_src'].value_counts(dropna=False)/df_out_drift_2.shape[0])
print('Downwelling Shortwave')
print(df_out_drift_2['down_short_hemisp_src'].value_counts(dropna=False)/df_out_drift_2.shape[0])
