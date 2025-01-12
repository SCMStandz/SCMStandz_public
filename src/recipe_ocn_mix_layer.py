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
# and run ```pip install -e .``` first
from mdftoolkit.MDF_toolkit import MDF_toolkit as MDF

# Load paths
with open(os.path.join('..', 'utils', 'paths.json')) as fp:
    paths_dict = json.load(fp)

# Specify which dates belong to Drift Track 1 and Drift Track 2
start_dt_drift_1 = pd.to_datetime("2019-10-06 12:00:00", utc=True)
end_dt_drift_1 = pd.to_datetime("2020-07-31 12:00:00", utc=True)
start_dt_drift_2 = pd.to_datetime("2020-08-21 12:00:00", utc=True)

# Load ocean data
fn = "MOSAiC_dailyaverage_skalars.nc"
ds = xr.open_dataset(os.path.join(paths_dict["raw_data"], fn))


# Specify corresponding variable names
var_map_dict = {'lat': 'latitude',
                'lon': 'longitude',
                'so'   : 'SSS',
                'mlotst': 'MLD',
                'xlot': 'mixL',
                'siu' : 'uice',
                'ustar'   : 'ustar',
                'hfsot' : 'Fh_thermo',
                'hfsoh' : 'Fh_halo',
                'tos' : 'SST',
                'tosf' : 'SSTf',
                }

# Create datetimeindex
# from Kiki, first measurement day is October 6, 2019, measurements
# are somewhat randomly distributed throughout each day so let's set the 
# the timestamp at noon
day_offset = pd.to_timedelta((ds.day - ds.day[0]).values, unit='D')
dti = pd.DatetimeIndex(data=start_dt_drift_1+day_offset)
ds = ds.assign_coords({"time": ('nprofile', dti.values)})
ds = ds.drop_vars('day')
ds = ds.swap_dims({'nprofile':'time'})

# Create dataframe for export
df_out = ds.to_dataframe()
# Fill missing dates with NaN values
dti = pd.date_range(df_out.index[0], df_out.index[-1], freq='D')
df_out = df_out.reindex(index=dti)

# Convert into MDF
# First drift track (legs 1-4)
df_out_drift_1 = df_out.loc[:end_dt_drift_1]
MDF_out = MDF(supersite_name='MOSAiC_ocn_drift1', verbose=True)
global_atts = {
    "title"                    : "MOSAiC ocean mixed layer drift 1",
    "Conventions"              : "MDF, CF (where possible)",
    "standard_name_vocabulary" : "",
    "contributor_name"         : "Kirstin Schulz",
    "contributor_email"        : "kiki.schulz@utexas.edu",
    "institution"              : "University of Texas",
    "creator_name"             : "David Clemens-Sewall",
    "creator_email"            : "david.clemens-sewall@noaa.gov",
    "project"                  : "MOSAiC Model Forcing Dataset Working Group",
    "summary"                  : "BETA ocean measurements",
    "id"                       : "DOI: XXXXXXXX",
    "license"                  : "CC-0", 
    "metadata_link"            : "http://www.doi-metatadat-link.org/mydoiiiii",
    "references"               : ("The Eurasian Arctic Ocean along the MOSAiC"+
                                  " drift (2019-2020): An interdisciplinary " +
                                  "perspective on properties and processes/" +
                                  "https://doi.org/10.1525/elementa.2023.00114"),
    "time_coverage_start"      : "{}".format(df_out_drift_1.index[0]),
    "time_coverage_end"        : "{}".format(df_out_drift_1.index[-1]),
    "naming_authority"         : "___", 
    "standard_name_vocabulary" : "___", 
    "keywords"                 : "ocean, arctic, polar",
    "calendar"                 : "standard",
}
MDF_out.update_global_atts(global_atts)  
modf_df = MDF.map_vars_to_mdf(df_out_drift_1, var_map_dict, drop=True)
MDF_out.add_data_timeseries(modf_df, cadence="time1440")
MDF_out.write_files(output_dir=paths_dict['proc_data'],
                    fname_only_underscores=True)

# Second drift track (leg 5)
df_out_drift_2 = df_out.loc[start_dt_drift_2:]
MDF_out = MDF(supersite_name='MOSAiC_ocn_drift2', verbose=True)
global_atts = {
    "title"                    : "MOSAiC ocean mixed layer drift 1",
    "Conventions"              : "MDF, CF (where possible)",
    "standard_name_vocabulary" : "",
    "contributor_name"         : "Kirstin Schulz",
    "contributor_email"        : "kiki.schulz@utexas.edu",
    "institution"              : "University of Texas",
    "creator_name"             : "David Clemens-Sewall",
    "creator_email"            : "david.clemens-sewall@noaa.gov",
    "project"                  : "MOSAiC Model Forcing Dataset Working Group",
    "summary"                  : "BETA ocean measurements",
    "id"                       : "doi:10.18739/A2RN30916",
    "license"                  : "CC-0", 
    "metadata_link"            : "https://doi.org/10.5281/zenodo.10819496",
    "references"               : ("The Eurasian Arctic Ocean along the MOSAiC"+
                                  " drift (2019-2020): An interdisciplinary " +
                                  "perspective on properties and processes/" +
                                  "https://doi.org/10.1525/elementa.2023.00114"),
    "time_coverage_start"      : "{}".format(df_out_drift_2.index[0]),
    "time_coverage_end"        : "{}".format(df_out_drift_2.index[-1]),
    "naming_authority"         : "___", 
    "standard_name_vocabulary" : "___", 
    "keywords"                 : "ocean, arctic, polar",
    "calendar"                 : "standard",
}
MDF_out.update_global_atts(global_atts)  
modf_df = MDF.map_vars_to_mdf(df_out_drift_2, var_map_dict, drop=True)
MDF_out.add_data_timeseries(modf_df, cadence="time1440")
MDF_out.write_files(output_dir=paths_dict['proc_data'],
                    fname_only_underscores=True)