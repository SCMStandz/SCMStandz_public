"""
Recipe for adding KAZR-derived precipitation to MOSAiC MDF
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

# Load kazr data
def proc_kazr(ds):
    """
    Process the time_offset from the kazr data
    """

    ds['seconds'] = ds['time_offset'] + ds['base_time']
    ds = ds.assign_coords({'time': pd.to_datetime(ds['seconds'], 
                                                  unit='s', origin='unix')})
    return ds[['snowfall_rate1', 'snowfall_rate2']]

ds_raw = xr.open_mfdataset(os.path.join(paths_dict['raw_data'], 
                                        'kazr', '*.nc'),
                           preprocess=proc_kazr, decode_times=False)

# Convert snowfallrate1 to precipitation in kg/m2 s
df_prec = ds_raw['snowfall_rate1'].to_dataframe().resample('T').mean()/3600
df_prec.rename(columns={'snowfall_rate1': 'prsn'}, inplace=True)

# Update global attributes
curr_references = MDF_out._global_atts['references']
MDF_out.update_global_atts({'references': curr_references +
                           '; High temporal resolution estimates of Arctic' +
                            'snowfall rates emphasizing gauge and radar-based'+
                            ' retrievals from the MOSAiC expedition/'+
                            'https://doi.org/10.1525/elementa.2021.00101'})

# Add to MDF
MDF_out.add_data_timeseries(df_prec, cadence="time01")
# Rename MDF and write files
MDF_out._name = "MOSAiC_kazr_snow"
MDF_out.write_files(output_dir=paths_dict['proc_data'])
