"""
Recipe for adding pseudo-precipitation from Ian Raphael
"""

import os
import json
import datetime
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# Load mdf_toolkit
# if this is first time running, go to the root of the mdf-toolkit directory 
# and run ```python setup.py install``` first
from mdftoolkit.MDF_toolkit import MDF_toolkit as MDF

# Load paths
with open(os.path.join('..', 'utils', 'paths.json')) as fp:
    paths_dict = json.load(fp)

# Load existing MDF file that doesn't have snowfall
fn = "MOSAiC_MDF_20191005-20201001.nc"
filename = os.path.join(paths_dict['proc_data'], fn)
MDF_out = MDF.load_mdf(filename)

# Load accumulation timeseries from Ian
snow_fp = os.path.join(paths_dict['raw_data'], 'stakes')
df_snow_fyi = pd.read_csv(os.path.join(snow_fp, 'FYI_snow_mosaic.csv'))
df_snow_syi = pd.read_csv(os.path.join(snow_fp, 'SYI_snow_mosaic copy.csv'))
df_snow_sheba = pd.read_csv(os.path.join(snow_fp, 'snow_sheba.csv'))

# Parameters for converting accumulation to precipitation
fyi_ow_date = "2019-11-16"
rhos = 330 # kg/m3 snow density
dt = 60 # timestamp in seconds

# Create timestamp column
df_snow_fyi['time01'] = pd.to_datetime(df_snow_fyi['Date'])
df_snow_syi['time01'] = pd.to_datetime(df_snow_syi['Date'])
df_snow_sheba['time01'] = pd.to_datetime(df_snow_sheba['Date'])

# We need SHEBA time to be at the same time as MOSAiC,
# We can just add 22 years
sheba_start = df_snow_sheba.at[0, 'time01']
td = sheba_start.replace(year=2019) - sheba_start
df_snow_sheba['time01'] = df_snow_sheba['time01'] + td

# precipitation rate is the diff of accumulation
# m/min * 1 min / 60 s * 330 kg / m3 = kg / s m2
df_snow_fyi['prsn'] = df_snow_fyi['Median snow "accumulation" (m)'].diff() * rhos / dt
df_snow_syi['prsn'] = df_snow_syi['Median snow "accumulation" (m)'].diff() * rhos / dt
df_snow_sheba['prsn'] = df_snow_sheba['Median snow "accumulation" (m)'].diff() * rhos / dt

# It looks like losses, presumably evaporation, in Icepack consume
# about 10% of the snow for the FYI case, so let's increase precipitation by 10%
df_snow_fyi['prsn'] = 1.1 * df_snow_fyi['prsn']

# also create precipitation rates from before data start
dt_fyi_ow = datetime.datetime.fromisoformat(fyi_ow_date)
prsn_pre = df_snow_fyi.at[0, 'Initial snow depth'] * rhos / (
    df_snow_fyi.at[0, 'time01'] - dt_fyi_ow).total_seconds()
dti = pd.date_range(start=MDF_out._subsites[None].var_data['time01'][0].index[0],
    end=df_snow_fyi.at[0, 'time01'], freq='min', name='time01', 
    inclusive='left')
se_pre_fyi = pd.Series(data=np.zeros(dti.size), index=dti, name='prsn')
se_pre_fyi.loc[dt_fyi_ow:dti[-1]] = prsn_pre
# SYI doesn't get preceding snow
se_pre_syi = pd.Series(data=np.zeros(dti.size), index=dti, name='prsn')

# merge to get combined precipitation rates
df_fyi_prsn = pd.concat([pd.DataFrame(se_pre_fyi), 
                        df_snow_fyi[['time01', 'prsn']].set_index('time01')])
df_syi_prsn = pd.concat([pd.DataFrame(se_pre_syi), 
                        df_snow_syi[['time01', 'prsn']].set_index('time01')])
# SHEBA starts early enough no need for pre-snow
df_sheba_prsn = df_snow_sheba[['time01', 'prsn']].set_index('time01')
# For SHEBA set the precipitation rate of the last day to 0
df_sheba_prsn.loc["2020-05-10":"2020-05-12", 'prsn'] = 0

# Add to MDF
MDF_out.add_data_timeseries(df_fyi_prsn, cadence="time01")
# Update global attributes
curr_references = MDF_out._global_atts['references']
MDF_out.update_global_atts({'references': curr_references +
                           '; Measurements of sea ice point-mass-balance ' +
                           'using hotwire thickness gauges and ablation ' +
                           'stakes during the Multidisciplinary drifting ' +
                           'Observatory for the Study of Arctic Climate ' + 
                           '(MOSAiC) Expedition in the Central Arctic ' +
                           '(2019-2020)/https://doi.org/10.18739/A2NK36626'})
curr_summary = MDF_out._global_atts['summary']
MDF_out.update_global_atts({'summary': curr_summary +
                            ', prsn is pseudo-precipitation derived from ' +
                            'accumulation at FYI stakes sites assuming snow' +
                            ' density of 330 kg/m3'})

# Rename MDF and write files
MDF_out._name = "MOSAiC_Raphael_snow_fyi"
MDF_out.write_files(output_dir=paths_dict['proc_data'])

# Repeat for SYI
del MDF_out
fn = "MOSAiC_MDF_20191005-20201001.nc"
filename = os.path.join(paths_dict['proc_data'], fn)
MDF_out = MDF.load_mdf(filename)
MDF_out.add_data_timeseries(df_syi_prsn, cadence="time01")
# Update global attributes
curr_references = MDF_out._global_atts['references']
MDF_out.update_global_atts({'references': curr_references +
                           '; Measurements of sea ice point-mass-balance ' +
                           'using hotwire thickness gauges and ablation ' +
                           'stakes during the Multidisciplinary drifting ' +
                           'Observatory for the Study of Arctic Climate ' + 
                           '(MOSAiC) Expedition in the Central Arctic ' +
                           '(2019-2020)/https://doi.org/10.18739/A2NK36626'})
curr_summary = MDF_out._global_atts['summary']
MDF_out.update_global_atts({'summary': curr_summary +
                            ', prsn is pseudo-precipitation derived from ' +
                            'accumulation at SYI stakes sites assuming snow' +
                            ' density of 330 kg/m3'})
# Rename MDF and write files
MDF_out._name = "MOSAiC_Raphael_snow_syi"
MDF_out.write_files(output_dir=paths_dict['proc_data'])

# Repeat for SHEBA
del MDF_out
fn = "MOSAiC_MDF_20191005-20201001.nc"
filename = os.path.join(paths_dict['proc_data'], fn)
MDF_out = MDF.load_mdf(filename)
MDF_out.add_data_timeseries(df_sheba_prsn, cadence="time01")
# Update global atts
MDF_out.update_global_atts({'summary': 'This file is a hybrid of MOSAiC '+
                            'surface energy balance terms with psuedo-' +
                            'precipitation derived from accumulation at' +
                            ' SHEBA stakes sites. Use with caution'})
# Rename MDF and write files
MDF_out._name = "MOSAiC_Raphael_snow_sheba"
MDF_out.write_files(output_dir=paths_dict['proc_data'])