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
from scipy.stats import linregress
from copy import deepcopy

# Load mdf_toolkit
# if this is first time running, go to the root of the mdf-toolkit directory 
# and run ```python setup.py install``` first
from mdftoolkit.MDF_toolkit import MDF_toolkit as MDF

# Load paths
with open(os.path.join('..', 'utils', 'paths.json')) as fp:
    paths_dict = json.load(fp)

def select_variables(ds):
    """
    Select only surface variables of interest to reduce data volume
    """

    varnames = ['temp_2m', 'rh_2m', 'atmos_pressure_2m',
                'wspd_u_mean_2m', 'wspd_v_mean_2m',
                'down_long_hemisp', 'down_short_hemisp',
                'zenith_true', 'zenith_apparent']
    # add qc flags
    for i in range(6):
        varnames.append(varnames[i] + '_qc')
    
    return ds[varnames]

def select_variables_asfs(ds):
    """
    Select only surface variables of interest to reduce data volume
    """

    varnames = ['temp', 'rh', 'atmos_pressure',
                'wspd_u_mean', 'wspd_v_mean',
                'down_long_hemisp', 'down_short_hemisp',
                'zenith_true', 'zenith_apparent']
    # add qc flags
    for i in range(6):
        varnames.append(varnames[i] + '_qc')
    
    return ds[varnames]

def preprocess_shiprad(ds):
    """
    Selects variables we want from shiprad and averages to minutely freq
    """

    return ds['spn1b_total1'].resample(time='min').mean()

# Load Met City, ASFS, and shiprad data
ds_metcity = xr.open_mfdataset(
 paths=os.path.join(paths_dict['raw_data'], 'metcity', '*1min*.nc'),
 chunks='auto', preprocess= select_variables,
 concat_dim='time', engine='netcdf4', combine="nested")

ds_asfs30 = xr.open_mfdataset(
 paths=os.path.join(paths_dict['raw_data'], 'asfs30', '*1min*.nc'),
 chunks='auto', preprocess= select_variables_asfs,
 concat_dim='time', engine='netcdf4', combine="nested")

ds_asfs40 = xr.open_mfdataset(
 paths=os.path.join(paths_dict['raw_data'], 'asfs40', '*1min*.nc'),
 chunks='auto', preprocess= select_variables_asfs,
 concat_dim='time', engine='netcdf4', combine="nested")

ds_asfs50 = xr.open_mfdataset(
 paths=os.path.join(paths_dict['raw_data'], 'asfs50', '*1min*.nc'),
 chunks='auto', preprocess= select_variables_asfs,
 concat_dim='time', engine='netcdf4', combine="nested")

# Met City is the primary dataset, but it contains a few gaps, fill these gaps
# with Na values
dti = pd.date_range(start=ds_metcity.indexes['time'][0],
                    end=ds_metcity.indexes['time'][-1], freq='min')
ds_metcity = ds_metcity.reindex(indexers={'time': dti})

# Preliminary work filling missing radiation with ship data.
# Need to revisit once ARM uploads longwave data
# For now just use unshaded shortwave at the center detector ('spn1b_total')
da_shipradS1 = xr.open_mfdataset(
    paths=os.path.join(paths_dict['raw_data'], 'shiprad', 'mosshipradS1*.nc'),
    chunks='auto', preprocess=preprocess_shiprad, concat_dim='time',
    engine='netcdf4', combine='nested')
# read into memory for convenience
da_shipradS1 = da_shipradS1.load()
# The first and last indices are duplicated, just drop for now
da_shipradS1 = da_shipradS1.drop_duplicates('time')

# Load gap filling guide
df_gap_fill = pd.read_excel("gap_fill_met_city.xlsx", parse_dates=[1, 2])

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
        warnings.warn("Missing start value at")        
    if not np.isnan(end_diff):
        da_gap_fill += end_diff * np.exp(
            -1*minutes_reverse/relaxation_minutes)
    else:
        warnings.warn("Missing end value at")
    
    # Gap filling
    arr_src = np.full(da_gap_fill.size, fill_source, dtype='object')
    da_gap_src = xr.DataArray(arr_src, coords={'time':da_gap_fill.time},
                              name=param + '_src')
    ds_gap_fill = xr.merge([da_gap_fill, da_gap_src])

    # Update dataset with filled values
    ds_fill = ds_gap_fill.combine_first(ds_fill)

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
    
# Create dicts needed for gap filling
varname_dict = {'temp_2m': 'temp', 
                'rh_2m': 'rh', 
                'atmos_pressure_2m': 'atmos_pressure',
                'wspd_u_mean_2m': 'wspd_u_mean', 
                'wspd_v_mean_2m': 'wspd_v_mean',
                'down_long_hemisp': 'down_long_hemisp', 
                'down_short_hemisp': 'down_short_hemisp',
                'mixing_ratio_2m': None}
source_dict = {'metcity': ds_metcity,
               'asfs30': ds_asfs30,
               'asfs40': ds_asfs40,
               'asfs50': ds_asfs50,
               }

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
# This results in a few small changes to the end timestamps of gaps
print(df_gap_fill.compare(df_gap_fill_mod))

# Create dataset with filled values
ds_fill = ds_metcity.copy(deep=True)
# Add variables for the source of filled parameters
arr_src = np.full(ds_fill.dims['time'], 'metcity', dtype='object')
for key in varname_dict.keys():
    ds_fill[key + '_src'] =('time',arr_src)

# Parameters that control bias correction and filling
overlap_minutes = 180 # 3 hour overlap 60*24*14 # 2 week overlap
relaxation_minutes = 5 # so that almost all transition is in 15 min.

# Fill identified gaps with ASFS data
for index, row in df_gap_fill_mod.iterrows():
    ds_fill = gap_fill(row['param'], row['station'], row['gap_start'], 
             row['gap_end'], row['offset'], 
             ds_fill, source_dict, overlap_minutes, relaxation_minutes,
             varname_dict, primary_source='metcity', plot=True,
             plotpath='/home/dcsewall/figures/gap_fill_met_city')

# Finish filling shortwave radiation gaps
# First, anytime the solar zenith angle is below a critical threshold set
# downwelling shortwave to zero, for simplicity let's use the civil twilight
# (sza > 96 degrees) applied to the solar zenith angle
# First we have to produce a continuous record of solar zenith angle
# Let's take the mean of all available met city and asfs data
da_sza = xr.concat([ds_metcity['zenith_true'], ds_asfs30['zenith_true'], 
                    ds_asfs40['zenith_true'], ds_asfs50['zenith_true']]
                    , dim='src')
da_sza = da_sza.mean(dim='src', skipna=True)
# And interpolate to fill missing values, use linear interpolation
da_sza = da_sza.chunk(dict(time=-1)).interpolate_na(dim='time', 
                                                    method='linear')
# The results are not perfect, but they look sufficient for this purpose
crit_sza = 96
da_sza.name = 'interp_sza'
ds_fill = xr.merge([ds_fill, da_sza])
ds_fill['down_short_hemisp'][(ds_fill['interp_sza'] >= crit_sza).compute()] = 0.0
 
# Fill gaps in radiation with shiprad
# just shortwave for now

# First we need to find all gaps in shortwave
# Simplest if we just move the entire array into memory
da_sw = ds_fill['down_short_hemisp'].compute()
gaps = []
in_gap = False
gap = np.zeros(2, dtype='datetime64[ns]')
last_time = 0
for row in da_sw:
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

df_gap_sw = pd.DataFrame({'param': 'down_short_hemisp',
                          'gap_start': gaps_co[:,0],
                          'gap_end': gaps_co[:,1],
                          'station': 'shipradS1',
                          'offset': 0})

# Create dataset for shiprad same as asfs
da_shipradS1.name = 'down_short_hemisp'
arr_src = np.full(da_shipradS1.sizes['time'], 'shipradS1', dtype='object')
da_src = xr.DataArray(arr_src, coords={'time':da_shipradS1.time}, 
                      name='down_short_hemisp_src')
ds_shipradS1 = xr.merge([da_shipradS1, da_src])

# Fill gaps
# Parameters that control bias correction and filling
overlap_minutes = 180 # 3 hour overlap 60*24*14 # 2 week overlap
relaxation_minutes = 5 # so that almost all transition is in 15 min.
source_dict = {'metcity': ds_metcity,
               'shipradS1': ds_shipradS1}
for index, row in df_gap_sw.iterrows():
    ds_fill = gap_fill(row['param'], row['station'], row['gap_start'], 
             row['gap_end'], row['offset'], 
             ds_fill, source_dict, overlap_minutes, relaxation_minutes,
             varname_dict, primary_source='metcity', plot=True,
             plotpath='/home/dcsewall/figures/gap_fill_met_city')

# function for mixing ratio
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

# Create dataframe for export
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

# Convert into MDF, for now ignore tagging source of the values
# instantiate class that will analyze and output data in MDF format
MDF_out = MDF(supersite_name='MOSAiC', verbose=True)
global_atts = {
    "title"                    : "MOSAiC surface meteorology",
    "Conventions"              : "MDF, CF (where possible)",
    "standard_name_vocabulary" : "",
    "contributor_name"         : "Christopher J. Cox",
    "contributor_email"        : "christopher.j.cox@noaa.gov",
    "institution"              : "NOAA",
    "creator_name"             : "David Clemens-Sewall",
    "creator_email"            : "dcsewall@ucar.edu",
    "project"                  : "MOSAiC Single Column Modeling Working Group",
    "summary"                  : "BETA gap filled with in-situ measurements",
    "id"                       : "DOI: XXXXXXXX",
    "license"                  : "CC-0", 
    "metadata_link"            : "http://www.doi-metatadat-link.org/mydoiiiii",
    "references"               : ("Continuous observations of the surface " +
                                  "energy budget and meteorology over the " +
                                  "Arctic sea ice during MOSAiC/" +
                                  "https://doi.org/10.1038/s41597-023-02415-5" +
                                  ";  Shipboard Portable Radiation Package/" +
                                  "https://doi.org/10.5439/1437994"),
    "time_coverage_start"      : "{}".format(df_out.index[0]),
    "time_coverage_end"        : "{}".format(df_out.index[-1]),
    "naming_authority"         : "___", 
    "standard_name_vocabulary" : "___", 
    "keywords"                 : "surface energy budget, arctic, polar",
    "calendar"                 : "standard",
}
MDF_out.update_global_atts(global_atts)  

# Specify which variables correspond
var_map_dict = {'uas'   : 'wspd_u_mean_2m',
                'vas'   : 'wspd_v_mean_2m',
                'tas'   : 'temp_2m',
                'hus'   : 'specific_humidity_2m',
                'rlds'  : 'down_long_hemisp',
                'rsds'  : 'down_short_hemisp',
                }
modf_df = MDF.map_vars_to_mdf(df_out, var_map_dict, drop=True)
MDF_out.add_data_timeseries(modf_df, cadence="time01")
MDF_out.write_files(output_dir=paths_dict['proc_data'])


### After here is not needed for creating MDF
# For summaries print amount of data and sources
print('Air Temperature')
print(df_out['temp_2m_src'].value_counts(dropna=False)/df_out.shape[0])
print('Atmospheric Pressure')
print(df_out['atmos_pressure_2m_src'].value_counts(dropna=False)/df_out.shape[0])
print('Relative Humidity')
print(df_out['rh_2m_src'].value_counts(dropna=False)/df_out.shape[0])
print('Wind Velocity')
print(df_out['wspd_u_mean_2m_src'].value_counts(dropna=False)/df_out.shape[0])
print('Downwelling Longwave')
print(df_out['down_long_hemisp_src'].value_counts(dropna=False)/df_out.shape[0])
print('Downwelling Shortwave')
print(df_out['down_short_hemisp_src'].value_counts(dropna=False)/df_out.shape[0])