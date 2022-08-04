#### Master Script 3b: Convert full patient information from ICU stays into tokenised sets ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Tokenise optional time variables
# III. Load and prepare formatted categorical predictors

### I. Initialisation
# Fundamental libraries
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
from tqdm import tqdm
from scipy import stats
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.preprocessing import KBinsDiscretizer

# Custom methods
from functions.token_preparation import categorizer

# Load cross-validation splits of study population
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Load CENTER-TBI ICU admission and discharge timestamps
CENTER_TBI_ICU_datetime = pd.read_csv('/home/sb2406/rds/hpc-work/timestamps/ICU_adm_disch_timestamps.csv')
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# Assign variable for directory for formatted predictors
form_pred_dir = os.path.join('/home/sb2406/rds/hpc-work/CENTER-TBI','FormattedPredictors')

# Create directory for storing tokens for each partition
tokens_dir = '/home/sb2406/rds/hpc-work/tokens'
os.makedirs(tokens_dir,exist_ok=True)

# Define the number of bins for discretising numeric variables
BINS = 20

# Load and format ICU stay windows for study population
study_windows = pd.read_csv('/home/sb2406/rds/hpc-work/timestamps/window_timestamps.csv')
study_windows['TimeStampStart'] = pd.to_datetime(study_windows['TimeStampStart'],format = '%Y-%m-%d %H:%M:%S.%f' )
study_windows['TimeStampEnd'] = pd.to_datetime(study_windows['TimeStampEnd'],format = '%Y-%m-%d %H:%M:%S.%f' )

### II. Tokenise optional time variables
## Add numerical markers of `SecondsSinceMidnight` and `DaysSinceAdm` to `study_windows` dataframe
# Merge admission timestamp to dataframe
study_windows = study_windows.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUAdmTimeStamp']],how='left',on='GUPI')

# Calculate `DaysSinceAdm`
study_windows['DaysSinceAdm'] = (study_windows['TimeStampEnd'] - study_windows['ICUAdmTimeStamp']).astype('timedelta64[s]')/86400

# Calculate `SecondsSinceMidnight` for time of day proxy
study_windows['SecondsSinceMidnight'] = (study_windows['TimeStampEnd'] - study_windows['TimeStampEnd'].dt.normalize()).astype('timedelta64[s]')

## Tokenise `SecondsSinceMidnight` into a `TimeOfDay` marker
# Create a `KBinsDiscretizer` object for discretising `TimeOfDay`
tod_kbd = KBinsDiscretizer(n_bins=BINS, encode='ordinal', strategy='quantile')

# Create a dummy linspace to represent possible `SecondsSinceMidnight`
dummy_secs_from_midnight = np.linspace(0,86400,num=10000)

# Train cuts for discretisation of time of day
tod_kbd.fit(np.expand_dims(dummy_secs_from_midnight,1))

# Discretise `SecondsSinceMidnight` of `study_windows` into bins
study_windows['TimeOfDay'] = ('TimeOfDay_BIN' + categorizer(pd.Series((tod_kbd.transform(np.expand_dims(study_windows.SecondsSinceMidnight.values,1))+1).squeeze()),100)).str.replace(r'\s+', '',regex=True)

## Remove unnecessary added variables
study_windows = study_windows.drop(columns=['ICUAdmTimeStamp','SecondsSinceMidnight'])

### III. Load and prepare formatted predictor sets
## Categorical Baseline Predictors
# Load formatted dataframe
categorical_baseline_predictors = pd.read_pickle(os.path.join(form_pred_dir,'categorical_baseline_predictors.pkl'))
categorical_baseline_predictors['Token'] = categorical_baseline_predictors.Token.str.strip()

# Merge dataframe with study windows to start tokens dataframe
study_tokens_df = study_windows.merge(categorical_baseline_predictors,how='left',on='GUPI').rename(columns={'Token':'TOKENS'})

## Categorical Discharge Predictors
# Load formatted dataframe
categorical_discharge_predictors = pd.read_pickle(os.path.join(form_pred_dir,'categorical_discharge_predictors.pkl')).rename(columns={'Token':'DischargeTokens'})
categorical_discharge_predictors['DischargeTokens'] = categorical_discharge_predictors.DischargeTokens.str.strip()

# Add last window index information to discharge predictor dataframe
categorical_discharge_predictors = categorical_discharge_predictors.merge(study_tokens_df[['GUPI','WindowTotal']].drop_duplicates().rename(columns={'WindowTotal':'WindowIdx'}),how='left',on='GUPI')

# Merge discharge tokens onto study tokens dataframe
study_tokens_df = study_tokens_df.merge(categorical_discharge_predictors,how='left',on=['GUPI','WindowIdx'])
study_tokens_df.TOKENS[~study_tokens_df.DischargeTokens.isna()] = study_tokens_df.TOKENS[~study_tokens_df.DischargeTokens.isna()] + ' ' + study_tokens_df.DischargeTokens[~study_tokens_df.DischargeTokens.isna()]

# Drop `DischargeTokens` column
study_tokens_df = study_tokens_df.drop(columns ='DischargeTokens')

## Categorical Date-Intervalled Predictors
# Load formatted dataframe
categorical_date_interval_predictors = pd.read_pickle(os.path.join(form_pred_dir,'categorical_date_interval_predictors.pkl')).rename(columns={'Token':'DateIntervalTokens'})
categorical_date_interval_predictors['DateIntervalTokens'] = categorical_date_interval_predictors.DateIntervalTokens.str.strip()
categorical_date_interval_predictors['EndToken'] = categorical_date_interval_predictors.EndToken.str.strip()

# Merge window timestamp starts and ends to formatted predictor dataframe
categorical_date_interval_predictors = categorical_date_interval_predictors.merge(study_tokens_df[['GUPI','TimeStampStart','TimeStampEnd','WindowIdx']],how='left',on='GUPI')

# First, isolate events which finish before the date ICU admission and combine end tokens
baseline_categorical_date_interval_predictors = categorical_date_interval_predictors[categorical_date_interval_predictors.WindowIdx == 1]
baseline_categorical_date_interval_predictors = baseline_categorical_date_interval_predictors[baseline_categorical_date_interval_predictors.StopDate.dt.date < baseline_categorical_date_interval_predictors.TimeStampStart.dt.date].reset_index(drop=True)
baseline_categorical_date_interval_predictors.DateIntervalTokens[~baseline_categorical_date_interval_predictors.EndToken.isna()] = baseline_categorical_date_interval_predictors.DateIntervalTokens[~baseline_categorical_date_interval_predictors.EndToken.isna()] + ' ' + baseline_categorical_date_interval_predictors.EndToken[~baseline_categorical_date_interval_predictors.EndToken.isna()]
baseline_categorical_date_interval_predictors = baseline_categorical_date_interval_predictors.drop(columns=['StartDate','StopDate','EndToken','TimeStampStart','TimeStampEnd'])
baseline_categorical_date_interval_predictors = baseline_categorical_date_interval_predictors.groupby(['GUPI','WindowIdx'],as_index=False).DateIntervalTokens.aggregate(lambda x: ' '.join(x))

# Merge event tokens which finish before the date of ICU admission onto study tokens dataframe
study_tokens_df = study_tokens_df.merge(baseline_categorical_date_interval_predictors,how='left',on=['GUPI','WindowIdx'])
study_tokens_df.TOKENS[~study_tokens_df.DateIntervalTokens.isna()] = study_tokens_df.TOKENS[~study_tokens_df.DateIntervalTokens.isna()] + ' ' + study_tokens_df.DateIntervalTokens[~study_tokens_df.DateIntervalTokens.isna()]
study_tokens_df = study_tokens_df.drop(columns ='DateIntervalTokens')

# Then, isolate the events that fit within the given window
categorical_date_interval_predictors = categorical_date_interval_predictors[(categorical_date_interval_predictors.StartDate.dt.date <= categorical_date_interval_predictors.TimeStampEnd.dt.date)&(categorical_date_interval_predictors.StopDate.dt.date >= categorical_date_interval_predictors.TimeStampStart.dt.date)].reset_index(drop=True)

# For each of these isolated events, find the maximum window index, to which the end token will be added
end_token_categorical_date_interval_predictors = categorical_date_interval_predictors.groupby(['GUPI','StartDate','StopDate','DateIntervalTokens','EndToken'],as_index=False).WindowIdx.max().drop(columns=['StartDate','StopDate','DateIntervalTokens']).groupby(['GUPI','WindowIdx'],as_index=False).EndToken.aggregate(lambda x: ' '.join(x))
categorical_date_interval_predictors = categorical_date_interval_predictors.drop(columns='EndToken')

# Merge end-of-interval event tokens onto study tokens dataframe
study_tokens_df = study_tokens_df.merge(end_token_categorical_date_interval_predictors,how='left',on=['GUPI','WindowIdx'])
study_tokens_df.TOKENS[~study_tokens_df.EndToken.isna()] = study_tokens_df.TOKENS[~study_tokens_df.EndToken.isna()] + ' ' + study_tokens_df.EndToken[~study_tokens_df.EndToken.isna()]
study_tokens_df = study_tokens_df.drop(columns ='EndToken')

# Merge date-interval event tokens onto study tokens dataframe
categorical_date_interval_predictors = categorical_date_interval_predictors.groupby(['GUPI','WindowIdx'],as_index=False).DateIntervalTokens.aggregate(lambda x: ' '.join(x))
study_tokens_df = study_tokens_df.merge(categorical_date_interval_predictors,how='left',on=['GUPI','WindowIdx'])
study_tokens_df.TOKENS[~study_tokens_df.DateIntervalTokens.isna()] = study_tokens_df.TOKENS[~study_tokens_df.DateIntervalTokens.isna()] + ' ' + study_tokens_df.DateIntervalTokens[~study_tokens_df.DateIntervalTokens.isna()]
study_tokens_df = study_tokens_df.drop(columns ='DateIntervalTokens')

## Categorical Time-Intervalled Predictors
# Load formatted dataframe
categorical_time_interval_predictors = pd.read_pickle(os.path.join(form_pred_dir,'categorical_time_interval_predictors.pkl')).rename(columns={'Token':'TimeIntervalTokens'})
categorical_time_interval_predictors['TimeIntervalTokens'] = categorical_time_interval_predictors.TimeIntervalTokens.str.strip()
categorical_time_interval_predictors['EndToken'] = categorical_time_interval_predictors.EndToken.str.strip()

# Merge window timestamp starts and ends to formatted predictor dataframe
categorical_time_interval_predictors = categorical_time_interval_predictors.merge(study_tokens_df[['GUPI','TimeStampStart','TimeStampEnd','WindowIdx']],how='left',on='GUPI')

# First, isolate events which finish before the ICU admission timestamp and combine end tokens
baseline_categorical_time_interval_predictors = categorical_time_interval_predictors[categorical_time_interval_predictors.WindowIdx == 1]
baseline_categorical_time_interval_predictors = baseline_categorical_time_interval_predictors[baseline_categorical_time_interval_predictors.EndTimeStamp < baseline_categorical_time_interval_predictors.TimeStampStart].reset_index(drop=True)
baseline_categorical_time_interval_predictors.TimeIntervalTokens[~baseline_categorical_time_interval_predictors.EndToken.isna()] = baseline_categorical_time_interval_predictors.TimeIntervalTokens[~baseline_categorical_time_interval_predictors.EndToken.isna()] + ' ' + baseline_categorical_time_interval_predictors.EndToken[~baseline_categorical_time_interval_predictors.EndToken.isna()]
baseline_categorical_time_interval_predictors = baseline_categorical_time_interval_predictors.drop(columns=['StartTimeStamp','EndTimeStamp','EndToken','TimeStampStart','TimeStampEnd'])
baseline_categorical_time_interval_predictors = baseline_categorical_time_interval_predictors.groupby(['GUPI','WindowIdx'],as_index=False).TimeIntervalTokens.aggregate(lambda x: ' '.join(x))

# Merge event tokens which finish before the ICU admission timestamp onto study tokens dataframe
study_tokens_df = study_tokens_df.merge(baseline_categorical_time_interval_predictors,how='left',on=['GUPI','WindowIdx'])
study_tokens_df.TOKENS[~study_tokens_df.TimeIntervalTokens.isna()] = study_tokens_df.TOKENS[~study_tokens_df.TimeIntervalTokens.isna()] + ' ' + study_tokens_df.TimeIntervalTokens[~study_tokens_df.TimeIntervalTokens.isna()]
study_tokens_df = study_tokens_df.drop(columns ='TimeIntervalTokens')

# Then, isolate the events that fit within the given window
categorical_time_interval_predictors = categorical_time_interval_predictors[(categorical_time_interval_predictors.StartTimeStamp <= categorical_time_interval_predictors.TimeStampEnd)&(categorical_time_interval_predictors.EndTimeStamp >= categorical_time_interval_predictors.TimeStampStart)].reset_index(drop=True)

# For each of these isolated events, find the maximum window index, to which the end token will be added
end_token_categorical_time_interval_predictors = categorical_time_interval_predictors.groupby(['GUPI','StartTimeStamp','EndTimeStamp','TimeIntervalTokens','EndToken'],as_index=False).WindowIdx.max().drop(columns=['StartTimeStamp','EndTimeStamp','TimeIntervalTokens']).groupby(['GUPI','WindowIdx'],as_index=False).EndToken.aggregate(lambda x:' '.join(x))
categorical_time_interval_predictors = categorical_time_interval_predictors.drop(columns='EndToken')

# Merge end-of-interval event tokens onto study tokens dataframe
study_tokens_df = study_tokens_df.merge(end_token_categorical_time_interval_predictors,how='left',on=['GUPI','WindowIdx'])
study_tokens_df.TOKENS[~study_tokens_df.EndToken.isna()] = study_tokens_df.TOKENS[~study_tokens_df.EndToken.isna()] + ' ' + study_tokens_df.EndToken[~study_tokens_df.EndToken.isna()]
study_tokens_df = study_tokens_df.drop(columns ='EndToken')

# Merge time-interval event tokens onto study tokens dataframe
categorical_time_interval_predictors = categorical_time_interval_predictors.groupby(['GUPI','WindowIdx'],as_index=False).TimeIntervalTokens.aggregate(lambda x: ' '.join(x))
study_tokens_df = study_tokens_df.merge(categorical_time_interval_predictors,how='left',on=['GUPI','WindowIdx'])
study_tokens_df.TOKENS[~study_tokens_df.TimeIntervalTokens.isna()] = study_tokens_df.TOKENS[~study_tokens_df.TimeIntervalTokens.isna()] + ' ' + study_tokens_df.TimeIntervalTokens[~study_tokens_df.TimeIntervalTokens.isna()]
study_tokens_df = study_tokens_df.drop(columns ='TimeIntervalTokens')

## Categorical Dated Single-Event Predictors in CENTER-TBI
# Load formatted dataframe
categorical_date_event_predictors = pd.read_pickle(os.path.join(form_pred_dir,'categorical_date_event_predictors.pkl')).rename(columns={'Token':'DateEventTokens'})
categorical_date_event_predictors['DateEventTokens'] = categorical_date_event_predictors.DateEventTokens.str.strip()

# Merge window timestamp starts and ends to formatted predictor dataframe
categorical_date_event_predictors = categorical_date_event_predictors.merge(study_tokens_df[['GUPI','TimeStampStart','TimeStampEnd','WindowIdx']],how='left',on='GUPI')

# First, isolate events which finish before the date ICU admission and combine end tokens
baseline_categorical_date_event_predictors = categorical_date_event_predictors[categorical_date_event_predictors.WindowIdx == 1]
baseline_categorical_date_event_predictors = baseline_categorical_date_event_predictors[baseline_categorical_date_event_predictors.Date.dt.date < baseline_categorical_date_event_predictors.TimeStampStart.dt.date].reset_index(drop=True)
baseline_categorical_date_event_predictors = baseline_categorical_date_event_predictors.drop(columns=['Date','TimeStampStart','TimeStampEnd'])
baseline_categorical_date_event_predictors = baseline_categorical_date_event_predictors.groupby(['GUPI','WindowIdx'],as_index=False).DateEventTokens.aggregate(lambda x: ' '.join(x))

# Merge event tokens which finish before the date of ICU admission onto study tokens dataframe
study_tokens_df = study_tokens_df.merge(baseline_categorical_date_event_predictors,how='left',on=['GUPI','WindowIdx'])
study_tokens_df.TOKENS[~study_tokens_df.DateEventTokens.isna()] = study_tokens_df.TOKENS[~study_tokens_df.DateEventTokens.isna()] + ' ' + study_tokens_df.DateEventTokens[~study_tokens_df.DateEventTokens.isna()]
study_tokens_df = study_tokens_df.drop(columns ='DateEventTokens')


##### HERE SHU

# Then, isolate the events that fit within the given window
categorical_date_event_predictors = categorical_date_event_predictors[(categorical_date_event_predictors.StartDate.dt.date <= categorical_date_event_predictors.TimeStampEnd.dt.date)&(categorical_date_event_predictors.StopDate.dt.date >= categorical_date_event_predictors.TimeStampStart.dt.date)].reset_index(drop=True)

# For each of these isolated events, find the maximum window index, to which the end token will be added
end_token_categorical_date_event_predictors = categorical_date_event_predictors.groupby(['GUPI','StartDate','StopDate','DateEventTokens','EndToken'],as_index=False).WindowIdx.max().drop(columns=['StartDate','StopDate','DateEventTokens']).groupby(['GUPI','WindowIdx'],as_index=False).EndToken.aggregate(lambda x: ' '.join(x))
categorical_date_event_predictors = categorical_date_event_predictors.drop(columns='EndToken')

# Merge end-of-interval event tokens onto study tokens dataframe
study_tokens_df = study_tokens_df.merge(end_token_categorical_date_event_predictors,how='left',on=['GUPI','WindowIdx'])
study_tokens_df.TOKENS[~study_tokens_df.EndToken.isna()] = study_tokens_df.TOKENS[~study_tokens_df.EndToken.isna()] + ' ' + study_tokens_df.EndToken[~study_tokens_df.EndToken.isna()]
study_tokens_df = study_tokens_df.drop(columns ='EndToken')

# Merge date-interval event tokens onto study tokens dataframe
categorical_date_event_predictors = categorical_date_event_predictors.groupby(['GUPI','WindowIdx'],as_index=False).DateIntervalTokens.aggregate(lambda x: ' '.join(x))
study_tokens_df = study_tokens_df.merge(categorical_date_event_predictors,how='left',on=['GUPI','WindowIdx'])
study_tokens_df.TOKENS[~study_tokens_df.DateIntervalTokens.isna()] = study_tokens_df.TOKENS[~study_tokens_df.DateIntervalTokens.isna()] + ' ' + study_tokens_df.DateIntervalTokens[~study_tokens_df.DateIntervalTokens.isna()]
study_tokens_df = study_tokens_df.drop(columns ='DateIntervalTokens')
