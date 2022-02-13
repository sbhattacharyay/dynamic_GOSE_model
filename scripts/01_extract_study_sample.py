#### Master Script 1: Extract study sample from CENTER-TBI dataset ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Load and filter CENTER-TBI dataset
# III. Characterise ICU stay timestamps
# IV. Prepare ICU stay windows for dynamic modelling

### I. Initialisation
import os
import sys
import time
import glob
import random
import datetime
import warnings
import itertools
import numpy as np
import pandas as pd
import pickle as cp
import seaborn as sns
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings(action="ignore")

### II. Load and filter CENTER-TBI dataset
# Load CENTER-TBI dataset to access ICU discharge date/times
CENTER_TBI_demo_info = pd.read_csv('../CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter patients who were enrolled in the ICU
CENTER_TBI_demo_info = CENTER_TBI_demo_info[CENTER_TBI_demo_info.PatientType == 3]

# Filter patients who are or are above 16 years of age
CENTER_TBI_demo_info = CENTER_TBI_demo_info[CENTER_TBI_demo_info.Age >= 16]

# Filter patients who have non-missing GOSE scores
CENTER_TBI_demo_info = CENTER_TBI_demo_info[~CENTER_TBI_demo_info.GOSE6monthEndpointDerived.isna()]

### III. Characterise ICU stay timestamps
# Select columns that indicate ICU admission and discharge times
CENTER_TBI_ICU_datetime = CENTER_TBI_demo_info[['GUPI','ICUAdmDate','ICUAdmTime','ICUDischDate','ICUDischTime']]

# Compile date and time information and convert to datetime
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUAdmDate', 'ICUAdmTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'][CENTER_TBI_ICU_datetime.ICUAdmDate.isna() | CENTER_TBI_ICU_datetime.ICUAdmTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'][CENTER_TBI_ICU_datetime.ICUDischDate.isna() | CENTER_TBI_ICU_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

CENTER_TBI_ICU_datetime['ICUDurationHours'] = (CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] - CENTER_TBI_ICU_datetime['ICUAdmTimeStamp']).astype('timedelta64[h]')

# For missing timestamps, cross-check with information available in other study
missing_timestamp_GUPIs = CENTER_TBI_ICU_datetime.GUPI[CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'].isna() | CENTER_TBI_ICU_datetime['ICUDischTimeStamp'].isna()].values

static_study_icu_timestamps = pd.read_csv('../../ordinal_GOSE_prediction/ICU_adm_disch_timestamps.csv')
static_study_icu_timestamps['ICUAdmTimeStamp'] = pd.to_datetime(static_study_icu_timestamps['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
static_study_icu_timestamps['ICUDischTimeStamp'] = pd.to_datetime(static_study_icu_timestamps['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
static_study_icu_timestamps = static_study_icu_timestamps[static_study_icu_timestamps.GUPI.isin(missing_timestamp_GUPIs)]

CENTER_TBI_ICU_datetime = CENTER_TBI_ICU_datetime[~CENTER_TBI_ICU_datetime.GUPI.isin(static_study_icu_timestamps.GUPI)]
CENTER_TBI_ICU_datetime = CENTER_TBI_ICU_datetime.append(static_study_icu_timestamps,ignore_index=True)

# Sort timestamps by GUPI
CENTER_TBI_ICU_datetime = CENTER_TBI_ICU_datetime.sort_values(by='GUPI',ignore_index = True)

# Filter out patients with less than 24 hours of ICU stay
CENTER_TBI_ICU_datetime = CENTER_TBI_ICU_datetime[CENTER_TBI_ICU_datetime.ICUDurationHours >= 24].reset_index(drop=True)

# Create directory to store study timestamp information
os.makedirs('../timestamps',exist_ok=True)

# Save timestamps as CSV
CENTER_TBI_ICU_datetime.to_csv('../timestamps/ICU_adm_disch_timestamps.csv',index = False)

### IV. Prepare ICU stay windows for dynamic modelling
# Define length in hours
WINDOW_HOURS = 2

# Define empty lists to store window information
FROM_ADM_STUDY_WINDOWS = []
FROM_DISCH_STUDY_WINDOWS = []

# Iterate through study patients and append window timestamps to list
for i in range(CENTER_TBI_ICU_datetime.shape[0]):
    
    # Extract current patient information
    curr_GUPI = CENTER_TBI_ICU_datetime.GUPI[i]
    curr_ICUAdmTimeStamp = CENTER_TBI_ICU_datetime.ICUAdmTimeStamp[i]
    curr_ICUDischTimeStamp = CENTER_TBI_ICU_datetime.ICUDischTimeStamp[i]
    
    # Calculate total ICU duration and number of study windows
    secs_ICUStay = (curr_ICUDischTimeStamp - curr_ICUAdmTimeStamp).total_seconds()
    total_bin_count = int(np.ceil(secs_ICUStay/(WINDOW_HOURS*3600)))
    
    # Derive start and end timestamps for study windows from admission
    from_adm_start_timestamps = [curr_ICUAdmTimeStamp + datetime.timedelta(hours=x*WINDOW_HOURS) for x in range(total_bin_count)]
    from_adm_end_timestamps = [ts - datetime.timedelta(microseconds=1) for ts in from_adm_start_timestamps[1:]]
    from_adm_end_timestamps.append(curr_ICUDischTimeStamp+datetime.timedelta(microseconds=1))
    
    # Derive start and end timestamps for study windows from discharge
    from_disch_start_timestamps = [curr_ICUDischTimeStamp - datetime.timedelta(hours=(x+1)*WINDOW_HOURS) for x in range(total_bin_count)]
    logicals = [ts < curr_ICUAdmTimeStamp for ts in from_disch_start_timestamps]
    if ~np.all(~np.array(logicals)):
        from_disch_start_timestamps[np.where(logicals)[0][0]] = curr_ICUAdmTimeStamp
    from_disch_end_timestamps = [ts - datetime.timedelta(microseconds=1) for ts in from_disch_start_timestamps[:-1]]
    from_disch_end_timestamps.insert(0,curr_ICUDischTimeStamp+datetime.timedelta(microseconds=1))
    
    # Compile current patient information into a dataframe
    from_adm_GUPI_windows = pd.DataFrame({'GUPI':curr_GUPI,
                                          'TimeStampStart':from_adm_start_timestamps,
                                          'TimeStampEnd':from_adm_end_timestamps,
                                          'WindowIdx':[i+1 for i in range(total_bin_count)],
                                          'WindowTotal':total_bin_count
                                     })
    
    from_disch_GUPI_windows = pd.DataFrame({'GUPI':curr_GUPI,
                                            'TimeStampStart':from_disch_start_timestamps,
                                            'TimeStampEnd':from_disch_end_timestamps,
                                            'WindowIdx':[i+1 for i in range(total_bin_count)],
                                            'WindowTotal':total_bin_count
                                 })
    
    # Append current GUPI dataframe to running compiled lists
    FROM_ADM_STUDY_WINDOWS.append(from_adm_GUPI_windows)
    FROM_DISCH_STUDY_WINDOWS.append(from_disch_GUPI_windows)
    
# Concatenate lists of dataframes into single study window dataframe
FROM_ADM_STUDY_WINDOWS = pd.concat(FROM_ADM_STUDY_WINDOWS,ignore_index=True)
FROM_DISCH_STUDY_WINDOWS = pd.concat(FROM_DISCH_STUDY_WINDOWS,ignore_index=True)

# Save window dataframe lists into `timestamps` directory
FROM_ADM_STUDY_WINDOWS.to_csv('../timestamps/from_adm_window_timestamps.csv',index = False)
FROM_DISCH_STUDY_WINDOWS.to_csv('../timestamps/from_disch_window_timestamps.csv',index = False)