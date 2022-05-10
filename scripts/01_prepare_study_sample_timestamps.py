#### Master Script 1: Extract and prepare study sample timestamps from CENTER-TBI dataset ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Load and filter CENTER-TBI dataset
# III. Characterise ICU stay timestamps and filling missing timestamp values

### I. Initialisation
import os
import sys
import time
import tqdm
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
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings(action="ignore")

### II. Load and filter CENTER-TBI dataset
# Load CENTER-TBI dataset to access ICU discharge date/times
CENTER_TBI_demo_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter patients who were enrolled in the ICU
CENTER_TBI_demo_info = CENTER_TBI_demo_info[CENTER_TBI_demo_info.PatientType == 3]

# Filter patients who are or are above 16 years of age
CENTER_TBI_demo_info = CENTER_TBI_demo_info[CENTER_TBI_demo_info.Age >= 16]

# Filter patients who have non-missing GOSE scores
CENTER_TBI_demo_info = CENTER_TBI_demo_info[~CENTER_TBI_demo_info.GOSE6monthEndpointDerived.isna()]

### III. Characterise ICU stay timestamps and filling missing timestamp values
# Select columns that indicate ICU admission and discharge times
CENTER_TBI_ICU_datetime = CENTER_TBI_demo_info[['GUPI','ICUAdmDate','ICUAdmTime','ICUDischDate','ICUDischTime']].reset_index(drop=True)

# Compile date and time information and convert to datetime
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUAdmDate', 'ICUAdmTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'][CENTER_TBI_ICU_datetime.ICUAdmDate.isna() | CENTER_TBI_ICU_datetime.ICUAdmTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'][CENTER_TBI_ICU_datetime.ICUDischDate.isna() | CENTER_TBI_ICU_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

CENTER_TBI_ICU_datetime['ICUDurationHours'] = (CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] - CENTER_TBI_ICU_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# To fill in missing timestamps, first check all available timestamps in nonrepeatable time info
non_repeatable_date_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/NonRepeatableDateInfo/data.csv',na_values = ["NA","NaN"," ", ""]).dropna(axis=1,how='all')
non_repeatable_date_info = non_repeatable_date_info[non_repeatable_date_info.GUPI.isin(CENTER_TBI_demo_info.GUPI)].reset_index(drop=True)
non_repeatable_date_info = non_repeatable_date_info[['GUPI']+[col for col in non_repeatable_date_info.columns if 'Date' in col]].select_dtypes(exclude='number')
non_repeatable_date_info = non_repeatable_date_info.melt(id_vars='GUPI',var_name='desc',value_name='date').dropna(subset=['date']).sort_values(by=['GUPI','date']).reset_index(drop=True)
non_repeatable_date_info['desc'] = non_repeatable_date_info['desc'].str.replace('Date','')

non_repeatable_time_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/NonRepeatableTimeInfo/data.csv',na_values = ["NA","NaN"," ", ""]).dropna(axis=1,how='all')
non_repeatable_time_info = non_repeatable_time_info[non_repeatable_time_info.GUPI.isin(CENTER_TBI_demo_info.GUPI)].reset_index(drop=True)
non_repeatable_time_info = non_repeatable_time_info[['GUPI']+[col for col in non_repeatable_time_info.columns if ('Time' in col)|('Hour' in col)]].select_dtypes(exclude='number')
non_repeatable_time_info = non_repeatable_time_info.melt(id_vars='GUPI',var_name='desc',value_name='time').dropna(subset=['time']).sort_values(by=['GUPI','time']).reset_index(drop=True)
non_repeatable_time_info['desc'] = non_repeatable_time_info['desc'].str.replace('Time','')

non_repeatable_datetime_info = pd.merge(non_repeatable_date_info,non_repeatable_time_info,how='outer',on=['GUPI','desc']).dropna(how='all',subset=['date','time'])
non_repeatable_datetime_info = non_repeatable_datetime_info.sort_values(by=['GUPI','date','time']).reset_index(drop=True)
os.makedirs('/home/sb2406/rds/hpc-work/timestamps/',exist_ok=True)
non_repeatable_datetime_info.to_csv('/home/sb2406/rds/hpc-work/timestamps/nonrepeatable_timestamps.csv',index=False)

# To fill in missing timestamps, second check all available timestamps in repeatable time info
repeatable_datetime_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/RepeatableDateTimeInfo/data.csv',na_values = ["NA","NaN"," ", ""]).dropna(axis=1,how='all')
repeatable_datetime_info = repeatable_datetime_info[repeatable_datetime_info.GUPI.isin(CENTER_TBI_demo_info.GUPI)].reset_index(drop=True)
repeatable_datetime_info['RowID'] = list(range(1,repeatable_datetime_info.shape[0]+1))
repeatable_datetime_info = repeatable_datetime_info.merge(CENTER_TBI_ICU_datetime,how='left',on='GUPI')

## Fix cases in which a missing date or time can be inferred from another variable
# Hourly values (HV) features
missing_HV_date = repeatable_datetime_info[(repeatable_datetime_info.HVDate.isna())&(~repeatable_datetime_info.HourlyValueTimePoint.isna())&(repeatable_datetime_info.HourlyValueTimePoint != 'None')].reset_index(drop=True)
missing_HV_date = missing_HV_date[['RowID','GUPI','ICUAdmDate','HourlyValueTimePoint','HVDate']]
missing_HV_date['ICUAdmDate'] = pd.to_datetime(missing_HV_date['ICUAdmDate'],format = '%Y-%m-%d')
missing_HV_date['HourlyValueTimePoint'] = missing_HV_date['HourlyValueTimePoint'].astype('int')
missing_HV_date['HVDate'] = (missing_HV_date['ICUAdmDate'] + pd.to_timedelta((missing_HV_date['HourlyValueTimePoint']-1),'days')).astype(str)

for curr_rowID in tqdm.tqdm(missing_HV_date.RowID,'Fixing missing HVDate values'):
    curr_Date = missing_HV_date.HVDate[missing_HV_date.RowID == curr_rowID].values[0]
    repeatable_datetime_info.HVDate[repeatable_datetime_info.RowID == curr_rowID] = curr_Date
    
repeatable_datetime_info.HVTime[(repeatable_datetime_info.HVTime.isna())&(~repeatable_datetime_info.HVHour.isna())] = repeatable_datetime_info.HVHour[(repeatable_datetime_info.HVTime.isna())&(~repeatable_datetime_info.HVHour.isna())]

# Daily values (DV) features (remove ward [non-ICU] observations)
ward_daily_vitals = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/DailyVitals/data.csv',na_values = ["NA","NaN"," ", ""])
ward_daily_vitals = ward_daily_vitals[ward_daily_vitals.PatientLocation == 'Ward'].reset_index(drop=True)
ward_daily_vitals = ward_daily_vitals[['GUPI','DVTimepoint','DVDate']]

repeatable_datetime_info = pd.merge(repeatable_datetime_info,ward_daily_vitals,how='left',on=['GUPI','DVTimepoint','DVDate'],indicator=True)
repeatable_datetime_info = repeatable_datetime_info[repeatable_datetime_info._merge != 'both'].drop(columns=['_merge']).reset_index(drop=True)

missing_DV_date = repeatable_datetime_info[(repeatable_datetime_info.DVDate.isna())&(~repeatable_datetime_info.DVTimepoint.isna())&(repeatable_datetime_info.DVTimepoint != 'None')].reset_index(drop=True)
missing_DV_date = missing_DV_date[['RowID','GUPI','ICUAdmDate','DVTimepoint','DVDate']]
missing_DV_date['ICUAdmDate'] = pd.to_datetime(missing_DV_date['ICUAdmDate'],format = '%Y-%m-%d')
missing_DV_date['DVTimepoint'] = missing_DV_date['DVTimepoint'].astype('int')
missing_DV_date['DVDate'] = (missing_DV_date['ICUAdmDate'] + pd.to_timedelta((missing_DV_date['DVTimepoint']-1),'days')).astype(str)

for curr_rowID in tqdm.tqdm(missing_DV_date.RowID,'Fixing missing DVDate values'):
    curr_Date = missing_DV_date.DVDate[missing_DV_date.RowID == curr_rowID].values[0]
    repeatable_datetime_info.DVDate[repeatable_datetime_info.RowID == curr_rowID] = curr_Date
    
# Remove follow-up information
repeatable_datetime_info = repeatable_datetime_info[repeatable_datetime_info.TimePoint.isna()].reset_index(drop=True)

# Outcomes
non_baseline_outcomes = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/Outcomes/data.csv',na_values = ["NA","NaN"," ", ""])
non_baseline_outcomes = non_baseline_outcomes[non_baseline_outcomes != 'Base'].reset_index(drop=True)
non_baseline_outcomes = non_baseline_outcomes[['GUPI','Timepoint']]

repeatable_datetime_info = pd.merge(repeatable_datetime_info,non_baseline_outcomes,how='left',on=['GUPI','Timepoint'],indicator=True)
repeatable_datetime_info = repeatable_datetime_info[repeatable_datetime_info._merge != 'both'].drop(columns=['_merge']).reset_index(drop=True)

# TIL
missing_TIL_date = repeatable_datetime_info[(repeatable_datetime_info.TILDate.isna())&(~repeatable_datetime_info.TILTimepoint.isna())&(repeatable_datetime_info.TILTimepoint != 'None')].reset_index(drop=True)
missing_TIL_date = missing_TIL_date[['RowID','GUPI','ICUAdmDate','TILTimepoint','TILDate']]
missing_TIL_date['ICUAdmDate'] = pd.to_datetime(missing_TIL_date['ICUAdmDate'],format = '%Y-%m-%d')
missing_TIL_date['TILTimepoint'] = missing_TIL_date['TILTimepoint'].astype('int')
missing_TIL_date['TILDate'] = (missing_TIL_date['ICUAdmDate'] + pd.to_timedelta((missing_TIL_date['TILTimepoint']-1),'days')).astype(str)

for curr_rowID in tqdm.tqdm(missing_TIL_date.RowID,'Fixing missing TILDate values'):
    curr_Date = missing_TIL_date.TILDate[missing_TIL_date.RowID == curr_rowID].values[0]
    repeatable_datetime_info.TILDate[repeatable_datetime_info.RowID == curr_rowID] = curr_Date

# Pivot repeatable date and time information to separate long format dataframes
repeatable_date_info = repeatable_datetime_info[['GUPI','RowID']+[col for col in repeatable_datetime_info.columns if 'Date' in col]].drop(columns=['ICUAdmDate']).dropna(axis=1,how='all')
repeatable_date_info['RowID'] = repeatable_date_info['RowID'].astype(str)
repeatable_date_info = repeatable_date_info.select_dtypes(exclude='number').melt(id_vars=['GUPI','RowID'],var_name='desc',value_name='date').dropna(subset=['date']).sort_values(by=['GUPI','date']).reset_index(drop=True)
repeatable_date_info['desc'] = repeatable_date_info['desc'].str.replace('Date','')

repeatable_time_info = repeatable_datetime_info[['GUPI','RowID']+[col for col in repeatable_datetime_info.columns if 'Time' in col]].dropna(axis=1,how='all')
repeatable_time_info['RowID'] = repeatable_time_info['RowID'].astype(str)
repeatable_time_info = repeatable_time_info.select_dtypes(exclude='number').melt(id_vars=['GUPI','RowID'],var_name='desc',value_name='time').dropna(subset=['time']).sort_values(by=['GUPI']).reset_index(drop=True)
repeatable_time_info['desc'] = repeatable_time_info['desc'].str.replace('Time','')

repeatable_datetime_info = pd.merge(repeatable_date_info,repeatable_time_info,how='outer',on=['RowID','GUPI','desc']).dropna(how='all',subset=['date','time'])
repeatable_datetime_info = repeatable_datetime_info.sort_values(by=['GUPI','date']).reset_index(drop=True)
repeatable_datetime_info.to_csv('/home/sb2406/rds/hpc-work/timestamps/repeatable_timestamps.csv',index=False)

## Fix missing ICU admission info based on non-repeatable datetime information
# Plan A: replace missing ICU admission timestamp with ED discharge times
missing_ICU_adm_timestamps = CENTER_TBI_ICU_datetime[CENTER_TBI_ICU_datetime.ICUAdmTimeStamp.isna()].reset_index(drop=True)

ED_discharge_datetime = non_repeatable_datetime_info[(non_repeatable_datetime_info.GUPI.isin(missing_ICU_adm_timestamps.GUPI))&(non_repeatable_datetime_info.desc == 'EDDisch')].fillna('1970-01-01').reset_index(drop=True)
ED_discharge_datetime = ED_discharge_datetime.merge(missing_ICU_adm_timestamps[['GUPI','ICUAdmDate']].rename(columns={'ICUAdmDate':'date'}),how='inner',on=['GUPI','date']).reset_index(drop=True)

for curr_GUPI in tqdm.tqdm(ED_discharge_datetime.GUPI,'Imputing missing ICU admission timestamps with matching ED discharge information'):
    curr_EDDisch_time = ED_discharge_datetime.time[ED_discharge_datetime.GUPI == curr_GUPI].values[0]
    missing_ICU_adm_timestamps.ICUAdmTime[missing_ICU_adm_timestamps.GUPI == curr_GUPI] = curr_EDDisch_time
    CENTER_TBI_ICU_datetime.ICUAdmTime[CENTER_TBI_ICU_datetime.GUPI == curr_GUPI] = curr_EDDisch_time

CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUAdmDate', 'ICUAdmTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'][CENTER_TBI_ICU_datetime.ICUAdmDate.isna() | CENTER_TBI_ICU_datetime.ICUAdmTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_ICU_datetime['ICUDurationHours'] = (CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] - CENTER_TBI_ICU_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan B: replace missing ICU admission timestamp with time of arrival to study hospital
missing_ICU_adm_timestamps = CENTER_TBI_ICU_datetime[CENTER_TBI_ICU_datetime.ICUAdmTimeStamp.isna()].reset_index(drop=True)

hosp_arrival_datetime = non_repeatable_datetime_info[(non_repeatable_datetime_info.GUPI.isin(missing_ICU_adm_timestamps.GUPI))&(non_repeatable_datetime_info.desc == 'PresSTHosp')].reset_index(drop=True)
hosp_arrival_datetime = hosp_arrival_datetime.merge(missing_ICU_adm_timestamps[['GUPI','ICUAdmDate']].rename(columns={'ICUAdmDate':'date'}),how='inner',on=['GUPI','date']).reset_index(drop=True)

for curr_GUPI in tqdm.tqdm(hosp_arrival_datetime.GUPI,'Imputing missing ICU admission timestamps with matching hosptial arrival information'):
    curr_hosp_arrival_time = hosp_arrival_datetime.time[hosp_arrival_datetime.GUPI == curr_GUPI].values[0]
    missing_ICU_adm_timestamps.ICUAdmTime[missing_ICU_adm_timestamps.GUPI == curr_GUPI] = curr_hosp_arrival_time
    CENTER_TBI_ICU_datetime.ICUAdmTime[CENTER_TBI_ICU_datetime.GUPI == curr_GUPI] = curr_hosp_arrival_time

CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUAdmDate', 'ICUAdmTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'][CENTER_TBI_ICU_datetime.ICUAdmDate.isna() | CENTER_TBI_ICU_datetime.ICUAdmTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_ICU_datetime['ICUDurationHours'] = (CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] - CENTER_TBI_ICU_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan C: replace missing ICU admission timestamp with time of initial study consent
missing_ICU_adm_timestamps = CENTER_TBI_ICU_datetime[CENTER_TBI_ICU_datetime.ICUAdmTimeStamp.isna()].reset_index(drop=True)

consent_datetime = non_repeatable_datetime_info[(non_repeatable_datetime_info.GUPI.isin(missing_ICU_adm_timestamps.GUPI))&(non_repeatable_datetime_info.desc == 'InfConsInitial')].reset_index(drop=True)
consent_datetime = consent_datetime.merge(missing_ICU_adm_timestamps[['GUPI','ICUAdmDate']].rename(columns={'ICUAdmDate':'date'}),how='inner',on=['GUPI','date']).reset_index(drop=True)

for curr_GUPI in tqdm.tqdm(consent_datetime.GUPI,'Imputing missing ICU admission timestamps with matching study consent information'):
    curr_consent_time = consent_datetime.time[consent_datetime.GUPI == curr_GUPI].values[0]
    missing_ICU_adm_timestamps.ICUAdmTime[missing_ICU_adm_timestamps.GUPI == curr_GUPI] = curr_consent_time
    CENTER_TBI_ICU_datetime.ICUAdmTime[CENTER_TBI_ICU_datetime.GUPI == curr_GUPI] = curr_consent_time

CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUAdmDate', 'ICUAdmTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'][CENTER_TBI_ICU_datetime.ICUAdmDate.isna() | CENTER_TBI_ICU_datetime.ICUAdmTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_ICU_datetime['ICUDurationHours'] = (CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] - CENTER_TBI_ICU_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan D: replace missing ICU admission timestamp with midday (no other information available on that day)
missing_ICU_adm_timestamps = CENTER_TBI_ICU_datetime[CENTER_TBI_ICU_datetime.ICUAdmTimeStamp.isna()].reset_index(drop=True)
CENTER_TBI_ICU_datetime.ICUAdmTime[CENTER_TBI_ICU_datetime.GUPI.isin(missing_ICU_adm_timestamps.GUPI)] = '12:00:00'
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUAdmDate', 'ICUAdmTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'][CENTER_TBI_ICU_datetime.ICUAdmDate.isna() | CENTER_TBI_ICU_datetime.ICUAdmTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_ICU_datetime['ICUDurationHours'] = (CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] - CENTER_TBI_ICU_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

## Fix missing ICU discharge info based on (non)-repeatable datetime information
# Plan A: if patient died in the ICU, replace missing discharge information with death date
missing_ICU_disch_timestamps = CENTER_TBI_ICU_datetime[CENTER_TBI_ICU_datetime.ICUDischTimeStamp.isna()].reset_index(drop=True)

death_datetime = non_repeatable_datetime_info[non_repeatable_datetime_info.GUPI.isin(missing_ICU_disch_timestamps.GUPI)&(non_repeatable_datetime_info.desc == 'Death')].reset_index(drop=True)
death_datetime = death_datetime.merge(missing_ICU_disch_timestamps[['GUPI','ICUDischDate']].rename(columns={'ICUDischDate':'date'}).dropna(subset=['date']),how='inner',on=['GUPI','date']).reset_index(drop=True)

for curr_GUPI in tqdm.tqdm(death_datetime.GUPI,'Imputing missing ICU discharge timestamps with matching time of death information'):
    curr_death_time = death_datetime.time[death_datetime.GUPI == curr_GUPI].values[0]
    missing_ICU_disch_timestamps.ICUDischTime[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_death_time
    CENTER_TBI_ICU_datetime.ICUDischTime[CENTER_TBI_ICU_datetime.GUPI == curr_GUPI] = curr_death_time

CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'][CENTER_TBI_ICU_datetime.ICUDischDate.isna() | CENTER_TBI_ICU_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_ICU_datetime['ICUDurationHours'] = (CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] - CENTER_TBI_ICU_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# A_1: If neither date nor time is available, simply replace after a manual reasonability check
missing_ICU_disch_timestamps = CENTER_TBI_ICU_datetime[CENTER_TBI_ICU_datetime.ICUDischTimeStamp.isna()].reset_index(drop=True)

death_datetime = non_repeatable_datetime_info[non_repeatable_datetime_info.GUPI.isin(missing_ICU_disch_timestamps.GUPI)&(non_repeatable_datetime_info.desc == 'Death')].reset_index(drop=True)
death_datetime = death_datetime[death_datetime.GUPI.isin(missing_ICU_disch_timestamps[missing_ICU_disch_timestamps.ICUDischTime.isna()]['GUPI'])].reset_index(drop=True)

for curr_GUPI in tqdm.tqdm(death_datetime.GUPI,'Imputing missing ICU discharge timestamps with matching time of death information'):
    curr_death_date = death_datetime.date[death_datetime.GUPI == curr_GUPI].values[0]
    curr_death_time = death_datetime.time[death_datetime.GUPI == curr_GUPI].values[0]
    
    missing_ICU_disch_timestamps.ICUDischDate[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_death_date
    missing_ICU_disch_timestamps.ICUDischTime[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_death_time
    
    CENTER_TBI_ICU_datetime.ICUDischDate[CENTER_TBI_ICU_datetime.GUPI == curr_GUPI] = curr_death_date
    CENTER_TBI_ICU_datetime.ICUDischTime[CENTER_TBI_ICU_datetime.GUPI == curr_GUPI] = curr_death_time
    
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'][CENTER_TBI_ICU_datetime.ICUDischDate.isna() | CENTER_TBI_ICU_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_ICU_datetime['ICUDurationHours'] = (CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] - CENTER_TBI_ICU_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan B: Find transfer-out-of-ICU date and time if available
missing_ICU_disch_timestamps = CENTER_TBI_ICU_datetime[CENTER_TBI_ICU_datetime.ICUDischTimeStamp.isna()].reset_index(drop=True)

transfer_datetime = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/TransitionsOfCare/data.csv',na_values = ["NA","NaN"," ", ""])
transfer_datetime = transfer_datetime[transfer_datetime.GUPI.isin(missing_ICU_disch_timestamps.GUPI)&(transfer_datetime.TransFrom == 'ICU')].dropna(subset=['DateEffectiveTransfer'])[['GUPI','DateEffectiveTransfer']].reset_index(drop=True)

for curr_GUPI in tqdm.tqdm(transfer_datetime.GUPI,'Imputing missing ICU discharge timestamps with transfer out of ICU information'):
    curr_transfer_date = transfer_datetime.DateEffectiveTransfer[transfer_datetime.GUPI == curr_GUPI].values[0]
    missing_ICU_disch_timestamps.ICUDischDate[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_transfer_date
    CENTER_TBI_ICU_datetime.ICUDischDate[CENTER_TBI_ICU_datetime.GUPI == curr_GUPI] = curr_transfer_date

CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'][CENTER_TBI_ICU_datetime.ICUDischDate.isna() | CENTER_TBI_ICU_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_ICU_datetime['ICUDurationHours'] = (CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] - CENTER_TBI_ICU_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan C: For non-missing ICU discharge dates, replace time with latest time available among non-repeatable timestamps
missing_ICU_disch_timestamps = CENTER_TBI_ICU_datetime[(CENTER_TBI_ICU_datetime.ICUDischTime.isna())&(~CENTER_TBI_ICU_datetime.ICUDischDate.isna())].reset_index(drop=True)

viable_non_repeatables = non_repeatable_datetime_info.merge(missing_ICU_disch_timestamps.rename(columns={'ICUDischDate':'date'}),how='inner',on=['GUPI','date']).dropna(subset=['time']).reset_index(drop=True)
viable_non_repeatables = viable_non_repeatables.groupby(['GUPI','date'],as_index=False).time.aggregate(max)

for curr_GUPI in tqdm.tqdm(viable_non_repeatables.GUPI,'Imputing missing ICU discharge times with last available timestamp on the day'):
    
    curr_disch_time = viable_non_repeatables.time[viable_non_repeatables.GUPI == curr_GUPI].values[0]
    missing_ICU_disch_timestamps.ICUDischTime[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_disch_time
    CENTER_TBI_ICU_datetime.ICUDischTime[CENTER_TBI_ICU_datetime.GUPI == curr_GUPI] = curr_disch_time

CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'][CENTER_TBI_ICU_datetime.ICUDischDate.isna() | CENTER_TBI_ICU_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_ICU_datetime['ICUDurationHours'] = (CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] - CENTER_TBI_ICU_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan D: For non-missing ICU discharge dates, replace time with latest time available among repeatable timestamps
missing_ICU_disch_timestamps = CENTER_TBI_ICU_datetime[(CENTER_TBI_ICU_datetime.ICUDischTime.isna())&(~CENTER_TBI_ICU_datetime.ICUDischDate.isna())].reset_index(drop=True)

viable_repeatables = repeatable_datetime_info.merge(missing_ICU_disch_timestamps.rename(columns={'ICUDischDate':'date'}),how='inner',on=['GUPI','date']).dropna(subset=['time']).reset_index(drop=True)
viable_repeatables = viable_repeatables.groupby(['GUPI','date'],as_index=False).time.aggregate(max)

for curr_GUPI in tqdm.tqdm(viable_repeatables.GUPI,'Imputing missing ICU discharge times with last available timestamp on the day'):
    curr_disch_time = viable_repeatables.time[viable_repeatables.GUPI == curr_GUPI].values[0]
    missing_ICU_disch_timestamps.ICUDischTime[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_disch_time
    CENTER_TBI_ICU_datetime.ICUDischTime[CENTER_TBI_ICU_datetime.GUPI == curr_GUPI] = curr_disch_time

CENTER_TBI_ICU_datetime.ICUDischTime[CENTER_TBI_ICU_datetime.GUPI.isin(missing_ICU_disch_timestamps.GUPI[missing_ICU_disch_timestamps.ICUDischTime.isna()])] = '23:59:00'
missing_ICU_disch_timestamps.ICUDischTime[missing_ICU_disch_timestamps.ICUDischTime.isna()] = '23:59:00'

CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'][CENTER_TBI_ICU_datetime.ICUDischDate.isna() | CENTER_TBI_ICU_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_ICU_datetime['ICUDurationHours'] = (CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] - CENTER_TBI_ICU_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan E: Replace with the maximum timestamp within 30 days of admission
missing_ICU_disch_timestamps = CENTER_TBI_ICU_datetime[CENTER_TBI_ICU_datetime.ICUDischTimeStamp.isna()].reset_index(drop=True)

viable_non_repeatables = non_repeatable_datetime_info[(non_repeatable_datetime_info.GUPI.isin(missing_ICU_disch_timestamps.GUPI))&(~non_repeatable_datetime_info.date.isna())]
viable_non_repeatables = viable_non_repeatables[viable_non_repeatables.date <= '1970-01-30'].sort_values(by=['GUPI','date','time'],ascending=False).reset_index(drop=True)
viable_non_repeatables = viable_non_repeatables.groupby('GUPI',as_index=False).first()

viable_repeatables = repeatable_datetime_info[(repeatable_datetime_info.GUPI.isin(missing_ICU_disch_timestamps.GUPI))&(~repeatable_datetime_info.date.isna())]
viable_repeatables = viable_repeatables[viable_repeatables.date <= '1970-01-30'].sort_values(by=['GUPI','date','time'],ascending=False).reset_index(drop=True)
viable_repeatables = viable_repeatables.groupby('GUPI',as_index=False).first()

viable_timepoints = pd.concat([viable_non_repeatables,viable_repeatables.drop(columns='RowID')],ignore_index=True)
viable_timepoints.time = viable_timepoints.time.str.replace('"','')

for curr_GUPI in tqdm.tqdm(viable_timepoints.GUPI,'Imputing missing ICU discharge times with last available timestamp on the day'):
    
    if missing_ICU_disch_timestamps[missing_ICU_disch_timestamps.GUPI == curr_GUPI].ICUDischTime.isna().values[0]:
        
        curr_disch_time = viable_timepoints.time[viable_timepoints.GUPI == curr_GUPI].values[0]
        missing_ICU_disch_timestamps.ICUDischTime[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_disch_time
        CENTER_TBI_ICU_datetime.ICUDischTime[CENTER_TBI_ICU_datetime.GUPI == curr_GUPI] = curr_disch_time
        
    if missing_ICU_disch_timestamps[missing_ICU_disch_timestamps.GUPI == curr_GUPI].ICUDischDate.isna().values[0]:
        
        curr_disch_date = viable_timepoints.date[viable_timepoints.GUPI == curr_GUPI].values[0]
        missing_ICU_disch_timestamps.ICUDischDate[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_disch_date
        CENTER_TBI_ICU_datetime.ICUDischDate[CENTER_TBI_ICU_datetime.GUPI == curr_GUPI] = curr_disch_date
                                    
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'][CENTER_TBI_ICU_datetime.ICUDischDate.isna() | CENTER_TBI_ICU_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_ICU_datetime['ICUDurationHours'] = (CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] - CENTER_TBI_ICU_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Plan F: Replace with hospital discharge timestamp if available
missing_ICU_disch_timestamps = CENTER_TBI_ICU_datetime[CENTER_TBI_ICU_datetime.ICUDischTimeStamp.isna()].reset_index(drop=True)

hosp_disch_timestamps = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/DemoInjHospMedHx/data.csv')
hosp_disch_timestamps = hosp_disch_timestamps[['GUPI','ICUDischDate','ICUDischTime','HospDischTime','HospDischDate']]
hosp_disch_timestamps = hosp_disch_timestamps[hosp_disch_timestamps.GUPI.isin(missing_ICU_disch_timestamps.GUPI)].reset_index(drop=True)

for curr_GUPI in tqdm.tqdm(hosp_disch_timestamps.GUPI,'Imputing missing ICU discharge times with hospital discharge'):
    
    if missing_ICU_disch_timestamps[missing_ICU_disch_timestamps.GUPI == curr_GUPI].ICUDischTime.isna().values[0]:
        
        curr_disch_time = hosp_disch_timestamps.HospDischTime[hosp_disch_timestamps.GUPI == curr_GUPI].values[0]
        missing_ICU_disch_timestamps.ICUDischTime[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_disch_time
        CENTER_TBI_ICU_datetime.ICUDischTime[CENTER_TBI_ICU_datetime.GUPI == curr_GUPI] = curr_disch_time
        
    if missing_ICU_disch_timestamps[missing_ICU_disch_timestamps.GUPI == curr_GUPI].ICUDischDate.isna().values[0]:
        
        curr_disch_date = hosp_disch_timestamps.HospDischDate[hosp_disch_timestamps.GUPI == curr_GUPI].values[0]
        missing_ICU_disch_timestamps.ICUDischDate[missing_ICU_disch_timestamps.GUPI == curr_GUPI] = curr_disch_date
        CENTER_TBI_ICU_datetime.ICUDischDate[CENTER_TBI_ICU_datetime.GUPI == curr_GUPI] = curr_disch_date

CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'][CENTER_TBI_ICU_datetime.ICUDischDate.isna() | CENTER_TBI_ICU_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_ICU_datetime['ICUDurationHours'] = (CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] - CENTER_TBI_ICU_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

## Inspect patients with less than 10 hours of ICU stay
inspection_batch = CENTER_TBI_ICU_datetime[CENTER_TBI_ICU_datetime.ICUDurationHours <= 10].reset_index(drop=True)

# Manual corrections based on plausible types
CENTER_TBI_ICU_datetime.ICUAdmDate[CENTER_TBI_ICU_datetime.GUPI == '6URh589'] = '1970-01-01'
CENTER_TBI_ICU_datetime.ICUAdmTime[CENTER_TBI_ICU_datetime.GUPI == '8uBs474'] = '05:19:00'
CENTER_TBI_ICU_datetime.ICUDischDate[CENTER_TBI_ICU_datetime.GUPI == '2CHw582'] = '1970-01-03'
CENTER_TBI_ICU_datetime.ICUDischDate[CENTER_TBI_ICU_datetime.GUPI == '3zmD494'] = '1970-01-02'
CENTER_TBI_ICU_datetime.ICUDischTime[CENTER_TBI_ICU_datetime.GUPI == '3zmD494'] = '13:00:00'

CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUAdmDate', 'ICUAdmTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'][CENTER_TBI_ICU_datetime.ICUAdmDate.isna() | CENTER_TBI_ICU_datetime.ICUAdmTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'][CENTER_TBI_ICU_datetime.ICUDischDate.isna() | CENTER_TBI_ICU_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

CENTER_TBI_ICU_datetime['ICUDurationHours'] = (CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] - CENTER_TBI_ICU_datetime['ICUAdmTimeStamp']).astype('timedelta64[s]')/3600

# Filter patients with at least 24 hours of ICU stay
CENTER_TBI_ICU_datetime = CENTER_TBI_ICU_datetime[CENTER_TBI_ICU_datetime.ICUDurationHours >= 24].sort_values(by=['GUPI']).reset_index(drop=True)

# Save timestamps as CSV
CENTER_TBI_ICU_datetime.to_csv('/home/sb2406/rds/hpc-work/timestamps/ICU_adm_disch_timestamps.csv',index = False)