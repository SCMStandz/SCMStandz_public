"""
Recipe for converting ocean mixed layer data into MDF
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

# Load ocean data
fn = "MOSAiC_dailyaverage_skalars.nc"
ds_raw = xr.open_dataset(os.path.join(paths_dict["raw_data"], fn))

# Subset to variables of interest
ds = ds_raw[['day', 'SST', 'SSS', 'MLD', 'SSTf', 'ustar', 'Fh_thermo']]

# Create datetimeindex
# from Kiki, first measurement day is October 6, 2019, measurements
# are somewhat randomly distributed throughout each day so let's set the 
# the timestamp at noon
start_dt = pd.to_datetime("2019-10-06 12:00:00", utc=True)
day_offset = pd.to_timedelta((ds.day - ds.day[0]).values, unit='D')
dti = pd.DatetimeIndex(data=start_dt+day_offset)
ds = ds.assign_coords({"time": ('nprofile', dti.values)})
ds = ds.drop_vars('day')
ds = ds.swap_dims({'nprofile':'time'})

# Write to MDF
# Create dataframe for export
df_out = ds.to_dataframe()
# Fill missing dates with NaN values
dti = pd.date_range(df_out.index[0], df_out.index[-1], freq='D')
df_out = df_out.reindex(index=dti)
# Convert into MDF, for now ignore tagging source of the values
# instantiate class that will analyze and output data in MDF format
MDF_out = MDF(supersite_name='MOSAiC_ocn', verbose=True)
global_atts = {
    "title"                    : "MOSAiC ocean mixed layer",
    "Conventions"              : "MDF, CF (where possible)",
    "standard_name_vocabulary" : "",
    "contributor_name"         : "Kirstin Schulz",
    "contributor_email"        : "kiki.schulz@utexas.edu",
    "institution"              : "University of Texas",
    "creator_name"             : "David Clemens-Sewall",
    "creator_email"            : "dcsewall@ucar.edu",
    "project"                  : "MOSAiC Single Column Model Working Group",
    "summary"                  : "BETA ocean measurements",
    "id"                       : "DOI: XXXXXXXX",
    "license"                  : "CC-0", 
    "metadata_link"            : "http://www.doi-metatadat-link.org/mydoiiiii",
    "references"               : ("The Eurasian Arctic Ocean along the MOSAiC"+
                                  " drift (2019-2020): An interdisciplinary " +
                                  "perspective on properties and processes/" +
                                  "https://doi.org/10.31223/X5TT2W"),
    "time_coverage_start"      : "{}".format(df_out.index[0]),
    "time_coverage_end"        : "{}".format(df_out.index[-1]),
    "naming_authority"         : "___", 
    "standard_name_vocabulary" : "___", 
    "keywords"                 : "ocean, arctic, polar",
    "calendar"                 : "standard",
}
MDF_out.update_global_atts(global_atts)  

# Specify which variables correspond
var_map_dict = {'so'   : 'SSS',
                'mlotst': 'MLD',
                'ustar'   : 'ustar',
                'hfsot' : 'Fh_thermo',
                'tos' : 'SST',
                'tosf' : 'SSTf',
                }
modf_df = MDF.map_vars_to_mdf(df_out, var_map_dict, drop=True)
MDF_out.add_data_timeseries(modf_df, cadence="time1440")
MDF_out.write_files(output_dir=paths_dict['proc_data'])