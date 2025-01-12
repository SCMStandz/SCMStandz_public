"""
Recipe for adding KAZR-derived precipitation to MOSAiC MDF

Three distinct datasets are used for different time periods:
2019-11-01 to 2020-05-14: Matrosov et al. (2022) https://doi.org/10.1525/elementa.2021.00101
2020-06-18 to 2020-07-31: Smith et al. (2024) https://doi.org/10.5194/egusphere-2024-1977
2020-08-20 to 2020-09-20: Persson (pers. comm. Jan. 2025) KAZR precipitation retrievals
"""

import os
import json
import datetime
import warnings
import numpy as np
import xarray as xr
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Load mdf_toolkit
from mdftoolkit.MDF_toolkit import MDF_toolkit as MDF

# Load paths
with open(os.path.join('..', 'utils', 'paths.json')) as fp:
    paths_dict = json.load(fp)

# Load kazr data
def proc_kazr(ds):
    """
    Process the time_offset from the kazr data
    """

    ds['seconds'] = ds['time_offset'] + ds['base_time']
    ds = ds.assign_coords({'time': pd.to_datetime(ds['seconds'], 
                                                  unit='s', origin='unix')})
    return ds[['snowfall_rate1', 'snowfall_rate2']]

ds_kazr = xr.open_mfdataset(os.path.join(paths_dict['raw_data'], 
                                        'kazr', '*.nc'),
                           preprocess=proc_kazr, decode_times=False)

# Convert snowfallrate1 to precipitation in kg/m2 s
df_winter_prec = ds_kazr['snowfall_rate1'].to_dataframe().resample('T').mean()/3600
df_winter_prec.rename(columns={'snowfall_rate1': 'prsn'}, inplace=True)

# Load summertime liquid precipitation from Smith et al. (2024)
df_summer = pd.read_csv(os.path.join(paths_dict['raw_data'], 
                                    'MOSAiCSummerLiquidPrecip.txt'),
                       delim_whitespace=True, header=6)
# convert time index to datetime
df_summer.index = (pd.to_datetime('2019-12-31') +
                   pd.to_timedelta(df_summer.Yearday.values, 'd')).round('min')
df_summer.index.rename('time', inplace=True)
# Average radar and PWD rain rate estimates and convert to kg m-2 s-1
df_summer['prlq'] = (df_summer.RR_rad + df_summer.RR_pwd)/(2.0 * 3600)

# Combine winter and summer precipitation
df_prec = pd.merge(df_winter_prec, df_summer.prlq, left_index=True,
                   right_index=True, how='outer').fillna(0.0)
df_prec['pr'] = df_prec['prsn'] + df_prec['prlq']

# Load existing MDF file for drift1 that doesn't have snowfall
fn = "MOSAiC_atm_drift1_MDF_20191015_20200731.nc"
filename = os.path.join(paths_dict['proc_data'], fn)
MDF_out = MDF.load_mdf(filename)

# Update global attributes
curr_references = MDF_out._global_atts['references']
curr_contrib_name = MDF_out._global_atts['contributor_name']
curr_contrib_email = MDF_out._global_atts['contributor_email']
MDF_out.update_global_atts({'references': curr_references +
                           '; High temporal resolution estimates of Arctic' +
                            'snowfall rates emphasizing gauge and radar-based'+
                            ' retrievals from the MOSAiC expedition/'+
                            'https://doi.org/10.1525/elementa.2021.00101;'+
                            'Formation and fate of freshwater on an ice ' +
                            'floe in the Central Arctic/'+
                            'https://doi.org/10.5194/egusphere-2024-1977',
                            'contributor_name': curr_contrib_name +
                            '; Matthew Shupe',
                            'contributor_email': curr_contrib_email +
                            '; matthew.shupe@noaa.gov'})

# Add to MDF
# Note that we are missing data for the leg 3-4 transition. Explicitly
# set these values to zero to prevent interpolation. A future update
# to the dataset will fill this gap.
df_prec = df_prec.reindex(MDF_out._subsites[None].var_data['time01']
                          [0].index, fill_value=0.0)

MDF_out.add_data_timeseries(df_prec, cadence="time01")
# Rename MDF and write files
MDF_out._name = "MOSAiC_atm_drift1_precip"
MDF_out.write_files(output_dir=paths_dict['proc_data'],
                    fname_only_underscores=True)

# For drift2, use KAZR estimates from Persson (pers. comm. Jan. 2025)
mat_dict = loadmat(os.path.join(paths_dict['raw_data'], 
                                'precip_rates_KAZR_PWD_233_265.mat'))
df_drift2 = pd.DataFrame({'yd_avg': mat_dict['yd_avg'].squeeze(),
                          'snwrt': mat_dict['snwrt'].squeeze(),
                          'rnrt': mat_dict['rnrt'].squeeze(),
                          'iprt': mat_dict['iprt'].squeeze()})

df_drift2_pwd = pd.DataFrame({'yd_pwd1': mat_dict['yd_pwd1'].squeeze(),
                              'pcprat_pwd1': mat_dict['pcprat_pwd1'].squeeze()})
# Create time index
df_drift2.index = (pd.to_datetime('2019-12-31') +
                   pd.to_timedelta(df_drift2.yd_avg.values, 'd'))
df_drift2.index.rename('time', inplace=True)
df_drift2_pwd.index = (pd.to_datetime('2019-12-31') +
                   pd.to_timedelta(df_drift2_pwd.yd_pwd1.values, 'd')).round('min')
df_drift2_pwd.index.rename('time', inplace=True)
# Convert to minutely means
df_drift2 = df_drift2.resample('T').mean()
# Drop yd_avg and convert to kg m-2 s-1
df_drift2.drop(columns='yd_avg', inplace=True)
df_drift2 = df_drift2/3600.0 # convert mm/hr to kg/m2/s
df_drift2_pwd.drop(columns='yd_pwd1', inplace=True)
df_drift2_pwd['pcprat_pwd1'] /= 3600.0 # convert mm/hr to kg/m2/s
# Fill times with no data with zeros
df_drift2 = df_drift2.fillna(0.0)
df_drift2_pwd = df_drift2_pwd.fillna(0.0)
# Create total precipitation and snowfall columns
# We will consider the precipitation classified as ice pellets to be
# snowfall since it is solid precipitation, note that this may
# introduce errors when considering the density and optical properties
# of the newly-fallen precipitation.
df_drift2['pr'] = df_drift2['snwrt'] + df_drift2['rnrt'] + df_drift2['iprt']
df_drift2['prsn'] = df_drift2['snwrt'] + df_drift2['iprt']

# Load existing MDF file for drift1 that doesn't have snowfall
fn = "MOSAiC_atm_drift2_MDF_20200827_20200918.nc"
filename = os.path.join(paths_dict['proc_data'], fn)
MDF_out = MDF.load_mdf(filename)

# Update global attributes
curr_references = MDF_out._global_atts['references']
curr_contrib_name = MDF_out._global_atts['contributor_name']
curr_contrib_email = MDF_out._global_atts['contributor_email']
MDF_out.update_global_atts({'references': curr_references +
                            '; Atmospheric Radiation Measurement (ARM)' +
                            ' user facility. 2019. Ka ARM Zenith Radar/' +
                            'http://dx.doi.org/10.5439/1615726',
                            'contributor_name': curr_contrib_name +
                            '; Ola Persson',
                            'contributor_email': curr_contrib_email +
                            '; ola.persson@noaa.gov'})


# Add to MDF
# Note that we are missing data for the leg 3-4 transition. Explicitly
# set these values to zero to prevent interpolation. A future update
# to the dataset will fill this gap.
df_drift2 = df_drift2.reindex(MDF_out._subsites[None].var_data['time01']
                          [0].index, fill_value=0.0)

MDF_out.add_data_timeseries(df_drift2, cadence="time01")
# Rename MDF and write files
MDF_out._name = "MOSAiC_atm_drift2_precip"
MDF_out.write_files(output_dir=paths_dict['proc_data'],
                    fname_only_underscores=True)