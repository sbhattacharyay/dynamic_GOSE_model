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
# IV. Tokenise numeric predictors and place into study windows
# V. Categorize tokens from dictionaries for characterization
# VI. Create full list of tokens for exploration
# VII. Post-hoc: collect and categorise tokens from v6-0

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
import multiprocessing
from scipy import stats
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
warnings.filterwarnings(action="ignore")
from collections import Counter, OrderedDict

# SciKit-Learn methods
from sklearn.preprocessing import KBinsDiscretizer

# PyTorch and PyTorch.Text methods
from torchtext.vocab import vocab, Vocab

# Custom methods
from functions.token_preparation import categorizer, clean_token_rows, get_token_info, count_token_incidences, get_legacy_token_info

# Load cross-validation splits of study population
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Define number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()

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

# Then, isolate the events that fit within the given window
categorical_date_event_predictors = categorical_date_event_predictors[(categorical_date_event_predictors.Date.dt.date <= categorical_date_event_predictors.TimeStampEnd.dt.date)&(categorical_date_event_predictors.Date.dt.date >= categorical_date_event_predictors.TimeStampStart.dt.date)].reset_index(drop=True)

# Merge dated event tokens onto study tokens dataframe
categorical_date_event_predictors = categorical_date_event_predictors.groupby(['GUPI','WindowIdx'],as_index=False).DateEventTokens.aggregate(lambda x: ' '.join(x))
study_tokens_df = study_tokens_df.merge(categorical_date_event_predictors,how='left',on=['GUPI','WindowIdx'])
study_tokens_df.TOKENS[~study_tokens_df.DateEventTokens.isna()] = study_tokens_df.TOKENS[~study_tokens_df.DateEventTokens.isna()] + ' ' + study_tokens_df.DateEventTokens[~study_tokens_df.DateEventTokens.isna()]
study_tokens_df = study_tokens_df.drop(columns ='DateEventTokens')

## Categorical Timestamped Single-Event Predictors in CENTER-TBI
# Load formatted dataframe
categorical_timestamp_event_predictors = pd.read_pickle(os.path.join(form_pred_dir,'categorical_timestamp_event_predictors.pkl')).rename(columns={'Token':'TimeStampEventTokens'})
categorical_timestamp_event_predictors['TimeStampEventTokens'] = categorical_timestamp_event_predictors.TimeStampEventTokens.str.strip()

# Merge window timestamp starts and ends to formatted predictor dataframe
categorical_timestamp_event_predictors = categorical_timestamp_event_predictors.merge(study_tokens_df[['GUPI','TimeStampStart','TimeStampEnd','WindowIdx']],how='left',on='GUPI')

# First, isolate events which finish before the ICU admission timestamp and combine end tokens
baseline_categorical_timestamp_event_predictors = categorical_timestamp_event_predictors[categorical_timestamp_event_predictors.WindowIdx == 1]
baseline_categorical_timestamp_event_predictors = baseline_categorical_timestamp_event_predictors[baseline_categorical_timestamp_event_predictors.TimeStamp < baseline_categorical_timestamp_event_predictors.TimeStampStart].reset_index(drop=True)
baseline_categorical_timestamp_event_predictors = baseline_categorical_timestamp_event_predictors.drop(columns=['TimeStamp','TimeStampStart','TimeStampEnd'])
baseline_categorical_timestamp_event_predictors = baseline_categorical_timestamp_event_predictors.groupby(['GUPI','WindowIdx'],as_index=False).TimeStampEventTokens.aggregate(lambda x: ' '.join(x))

# Merge event tokens which finish before the date of ICU admission onto study tokens dataframe
study_tokens_df = study_tokens_df.merge(baseline_categorical_timestamp_event_predictors,how='left',on=['GUPI','WindowIdx'])
study_tokens_df.TOKENS[~study_tokens_df.TimeStampEventTokens.isna()] = study_tokens_df.TOKENS[~study_tokens_df.TimeStampEventTokens.isna()] + ' ' + study_tokens_df.TimeStampEventTokens[~study_tokens_df.TimeStampEventTokens.isna()]
study_tokens_df = study_tokens_df.drop(columns ='TimeStampEventTokens')

# Then, isolate the events that fit within the given window
categorical_timestamp_event_predictors = categorical_timestamp_event_predictors[(categorical_timestamp_event_predictors.TimeStamp <= categorical_timestamp_event_predictors.TimeStampEnd)&(categorical_timestamp_event_predictors.TimeStamp >= categorical_timestamp_event_predictors.TimeStampStart)].reset_index(drop=True)

# Merge timestamped event tokens onto study tokens dataframe
categorical_timestamp_event_predictors = categorical_timestamp_event_predictors.groupby(['GUPI','WindowIdx'],as_index=False).TimeStampEventTokens.aggregate(lambda x: ' '.join(x))
study_tokens_df = study_tokens_df.merge(categorical_timestamp_event_predictors,how='left',on=['GUPI','WindowIdx'])
study_tokens_df.TOKENS[~study_tokens_df.TimeStampEventTokens.isna()] = study_tokens_df.TOKENS[~study_tokens_df.TimeStampEventTokens.isna()] + ' ' + study_tokens_df.TimeStampEventTokens[~study_tokens_df.TimeStampEventTokens.isna()]
study_tokens_df = study_tokens_df.drop(columns ='TimeStampEventTokens')

## Iterate through and clean categorical predictors and save
# Partition categorical token rows among cores
s = [study_tokens_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
s[:(study_tokens_df.shape[0] - sum(s))] = [over+1 for over in s[:(study_tokens_df.shape[0] - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
windows_per_core = [(study_tokens_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Cleaning categorical predictor dataframe') for idx in range(len(start_idx))]

# Inspect each token row in parallel to ensure unique tokens
with multiprocessing.Pool(NUM_CORES) as pool:
    cleaned_study_tokens_df = pd.concat(pool.starmap(clean_token_rows, windows_per_core),ignore_index=True).sort_values(by=['GUPI','WindowIdx']).reset_index(drop=True)
    
# Save cleaned categorical tokens-windows to formatted predictors directory
cleaned_study_tokens_df.to_pickle(os.path.join(form_pred_dir,'categorical_tokens_in_study_windows.pkl'))

### IV. Tokenise numeric predictors and place into study windows
## Iterate through folds and tokenise numeric predictors per distributions in the training set
for curr_fold in tqdm(cv_splits.FOLD.unique(),'Iterating through folds of cross-validation for numeric predictor tokenisation'):
    
    ## Load cleaned categorical tokens in study windows
    cleaned_study_tokens_df = pd.read_pickle(os.path.join(form_pred_dir,'categorical_tokens_in_study_windows.pkl'))
    cleaned_study_tokens_df['TOKENS'] = cleaned_study_tokens_df.TOKENS.str.strip()

    ## Load formatted numeric predictors
    # Numeric baseline predictors
    numeric_baseline_predictors = pd.read_pickle(os.path.join(form_pred_dir,'numeric_baseline_predictors.pkl')).reset_index(drop=True)
    numeric_baseline_predictors['VARIABLE'] = numeric_baseline_predictors.VARIABLE.str.strip().str.replace('_','')

    # Numeric discharge predictors
    numeric_discharge_predictors = pd.read_pickle(os.path.join(form_pred_dir,'numeric_discharge_predictors.pkl')).reset_index(drop=True)
    numeric_discharge_predictors['VARIABLE'] = numeric_discharge_predictors.VARIABLE.str.strip().str.replace('_','')

    # Numeric dated single-event predictors
    numeric_date_event_predictors = pd.read_pickle(os.path.join(form_pred_dir,'numeric_date_event_predictors.pkl')).reset_index(drop=True)
    numeric_date_event_predictors['VARIABLE'] = numeric_date_event_predictors.VARIABLE.str.strip().str.replace('_','')

    # Numeric timestamped single-event predictors
    numeric_timestamp_event_predictors = pd.read_pickle(os.path.join(form_pred_dir,'numeric_timestamp_event_predictors.pkl')).reset_index(drop=True)
    numeric_timestamp_event_predictors['VARIABLE'] = numeric_timestamp_event_predictors.VARIABLE.str.strip().str.replace('_','')
    
    # Create a subdirectory for the current fold
    fold_dir = os.path.join(tokens_dir,'fold'+str(curr_fold))
    os.makedirs(fold_dir,exist_ok=True)

    ## Extract current training, validation, and testing set GUPIs
    curr_fold_splits = cv_splits[(cv_splits.FOLD==curr_fold)].reset_index(drop=True)
    curr_train_GUPIs = curr_fold_splits[curr_fold_splits.SET=='train'].GUPI.unique()
    curr_val_GUPIs = curr_fold_splits[curr_fold_splits.SET=='val'].GUPI.unique()
    curr_test_GUPIs = curr_fold_splits[curr_fold_splits.SET=='test'].GUPI.unique()
    
    ## First, tokenise `DaysSinceAdm`
    # Create a `KBinsDiscretizer` object for discretising `DaysSinceAdm`
    dsa_kbd = KBinsDiscretizer(n_bins=BINS, encode='ordinal', strategy='quantile')
    
    # Train cuts for discretisation of days since admission
    dsa_kbd.fit(np.expand_dims(cleaned_study_tokens_df[cleaned_study_tokens_df.GUPI.isin(curr_train_GUPIs)].DaysSinceAdm.values,1))

    # Discretise `DaysSinceAdm` of `study_windows` into bins
    cleaned_study_tokens_df['DaysSinceAdm'] = ('DaysSinceAdm_BIN' + categorizer(pd.Series((dsa_kbd.transform(np.expand_dims(cleaned_study_tokens_df.DaysSinceAdm.values,1))+1).squeeze()),100)).str.replace(r'\s+','',regex=True)
    
    ## Numeric baseline predictors
    # Extract unique names of numeric baseline predictors from the training set
    unique_numeric_baseline_predictors = numeric_baseline_predictors[numeric_baseline_predictors.GUPI.isin(curr_train_GUPIs)].VARIABLE.unique()
    
    # Create column for storing bin value
    numeric_baseline_predictors['BIN'] = ''
    
    # For missing values, assign 'NAN' to bin value
    numeric_baseline_predictors.BIN[numeric_baseline_predictors.VALUE.isna()] = '_NAN'
    
    # Iterate through unique numeric baseline predictors and tokenise
    for curr_predictor in tqdm(unique_numeric_baseline_predictors,'Tokenising numeric baseline predictors for fold '+str(curr_fold)):
        
        # Create a `KBinsDiscretizer` object for discretising the current predictor
        curr_nbp_kbd = KBinsDiscretizer(n_bins=BINS, encode='ordinal', strategy='quantile')
        
        # Train cuts for discretisation of the current predictor
        curr_nbp_kbd.fit(np.expand_dims(numeric_baseline_predictors[(numeric_baseline_predictors.VARIABLE==curr_predictor)&(numeric_baseline_predictors.GUPI.isin(curr_train_GUPIs))&(~numeric_baseline_predictors.VALUE.isna())].VALUE.values,1))
        
        # Discretise current predictor into bins
        numeric_baseline_predictors.BIN[(numeric_baseline_predictors.VARIABLE==curr_predictor)&(~numeric_baseline_predictors.VALUE.isna())] = (categorizer(pd.Series((curr_nbp_kbd.transform(np.expand_dims(numeric_baseline_predictors[(numeric_baseline_predictors.VARIABLE==curr_predictor)&(~numeric_baseline_predictors.VALUE.isna())].VALUE.values,1))+1).squeeze()),100)).str.replace(r'\s+','',regex=True).values
        
    # If a predictor has been neglected, replace with value
    numeric_baseline_predictors.BIN[numeric_baseline_predictors.BIN==''] = numeric_baseline_predictors.VALUE[numeric_baseline_predictors.BIN==''].astype(str).str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)
    
    # Create tokens from each variable and bin value
    numeric_baseline_predictors['TOKEN'] = numeric_baseline_predictors.VARIABLE + '_BIN' + numeric_baseline_predictors.BIN
    
    # Concatenate tokens from each GUPI into a combined baseline numeric predictor token set
    numeric_baseline_predictors = numeric_baseline_predictors.drop_duplicates(subset=['GUPI','TOKEN'],ignore_index=True).groupby('GUPI',as_index=False).TOKEN.aggregate(lambda x: ' '.join(x)).rename(columns={'TOKEN':'NumericBaselineTokens'})
    
    # Merge baseline numeric predictors with `cleaned_study_tokens_df`
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(numeric_baseline_predictors,how='left',on=['GUPI'])
    cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericBaselineTokens.isna()] = cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericBaselineTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericBaselineTokens[~cleaned_study_tokens_df.NumericBaselineTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericBaselineTokens')
    
    ## Numeric discharge predictors
    # Extract unique names of numeric discharge predictors from the training set
    unique_numeric_discharge_predictors = numeric_discharge_predictors[numeric_discharge_predictors.GUPI.isin(curr_train_GUPIs)].VARIABLE.unique()
    
    # Create column for storing bin value
    numeric_discharge_predictors['BIN'] = ''
    
    # For missing values, assign 'NAN' to bin value
    numeric_discharge_predictors.BIN[numeric_discharge_predictors.VALUE.isna()] = '_NAN'
    
    # Iterate through unique numeric discharge predictors and tokenise
    for curr_predictor in tqdm(unique_numeric_discharge_predictors,'Tokenising numeric discharge predictors for fold '+str(curr_fold)):
        
        # Create a `KBinsDiscretizer` object for discretising the current predictor
        curr_ndp_kbd = KBinsDiscretizer(n_bins=BINS, encode='ordinal', strategy='quantile')
        
        # Train cuts for discretisation of the current predictor
        curr_ndp_kbd.fit(np.expand_dims(numeric_discharge_predictors[(numeric_discharge_predictors.VARIABLE==curr_predictor)&(numeric_discharge_predictors.GUPI.isin(curr_train_GUPIs))&(~numeric_discharge_predictors.VALUE.isna())].VALUE.values,1))
        
        # Discretise current predictor into bins
        numeric_discharge_predictors.BIN[(numeric_discharge_predictors.VARIABLE==curr_predictor)&(~numeric_discharge_predictors.VALUE.isna())] = (categorizer(pd.Series((curr_ndp_kbd.transform(np.expand_dims(numeric_discharge_predictors[(numeric_discharge_predictors.VARIABLE==curr_predictor)&(~numeric_discharge_predictors.VALUE.isna())].VALUE.values,1))+1).squeeze()),100)).str.replace(r'\s+','',regex=True).values
        
    # If a predictor has been neglected, replace with value
    numeric_discharge_predictors.BIN[numeric_discharge_predictors.BIN==''] = numeric_discharge_predictors.VALUE[numeric_discharge_predictors.BIN==''].astype(str).str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)
    
    # Create tokens from each variable and bin value
    numeric_discharge_predictors['TOKEN'] = numeric_discharge_predictors.VARIABLE + '_BIN' + numeric_discharge_predictors.BIN
    
    # Concatenate tokens from each GUPI into a combined discharge numeric predictor token set
    numeric_discharge_predictors = numeric_discharge_predictors.drop_duplicates(subset=['GUPI','TOKEN'],ignore_index=True).groupby('GUPI',as_index=False).TOKEN.aggregate(lambda x: ' '.join(x)).rename(columns={'TOKEN':'NumericDischargeTokens'})
    
    # Add last window index information to discharge predictor dataframe
    numeric_discharge_predictors = numeric_discharge_predictors.merge(cleaned_study_tokens_df[['GUPI','WindowTotal']].drop_duplicates().rename(columns={'WindowTotal':'WindowIdx'}),how='left',on='GUPI')

    # Merge discharge numeric predictors with `cleaned_study_tokens_df`
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(numeric_discharge_predictors,how='left',on=['GUPI','WindowIdx'])
    cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericDischargeTokens.isna()] = cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericDischargeTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericDischargeTokens[~cleaned_study_tokens_df.NumericDischargeTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericDischargeTokens')
    
    ## Numeric dated single-event predictors
    # Extract unique names of numeric dated single-event predictors from the training set
    unique_numeric_date_event_predictors = numeric_date_event_predictors[numeric_date_event_predictors.GUPI.isin(curr_train_GUPIs)].VARIABLE.unique()
    
    # Create column for storing bin value
    numeric_date_event_predictors['BIN'] = ''
    
    # For missing values, assign 'NAN' to bin value
    numeric_date_event_predictors.BIN[numeric_date_event_predictors.VALUE.isna()] = '_NAN'
    
    # Iterate through unique numeric dated single-event predictors and tokenise
    for curr_predictor in tqdm(unique_numeric_date_event_predictors,'Tokenising numeric dated single-event predictors for fold '+str(curr_fold)):
        
        # Create a `KBinsDiscretizer` object for discretising the current predictor
        curr_ndep_kbd = KBinsDiscretizer(n_bins=BINS, encode='ordinal', strategy='quantile')
        
        # Train cuts for discretisation of the current predictor
        curr_ndep_kbd.fit(np.expand_dims(numeric_date_event_predictors[(numeric_date_event_predictors.VARIABLE==curr_predictor)&(numeric_date_event_predictors.GUPI.isin(curr_train_GUPIs))&(~numeric_date_event_predictors.VALUE.isna())].VALUE.values,1))
        
        # Discretise current predictor into bins
        numeric_date_event_predictors.BIN[(numeric_date_event_predictors.VARIABLE==curr_predictor)&(~numeric_date_event_predictors.VALUE.isna())] = (categorizer(pd.Series((curr_ndep_kbd.transform(np.expand_dims(numeric_date_event_predictors[(numeric_date_event_predictors.VARIABLE==curr_predictor)&(~numeric_date_event_predictors.VALUE.isna())].VALUE.values,1))+1).squeeze()),100)).str.replace(r'\s+','',regex=True).values
        
    # If a predictor has been neglected, replace with value
    numeric_date_event_predictors.BIN[numeric_date_event_predictors.BIN==''] = numeric_date_event_predictors.VALUE[numeric_date_event_predictors.BIN==''].astype(str).str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)
    
    # Create tokens from each variable and bin value
    numeric_date_event_predictors['TOKEN'] = numeric_date_event_predictors.VARIABLE + '_BIN' + numeric_date_event_predictors.BIN
    
    # Concatenate tokens from each GUPI and date into a combined dated single-event numeric predictor token set
    numeric_date_event_predictors = numeric_date_event_predictors.drop_duplicates(subset=['GUPI','Date','TOKEN'],ignore_index=True).groupby(['GUPI','Date'],as_index=False).TOKEN.aggregate(lambda x: ' '.join(x)).rename(columns={'TOKEN':'NumericDateEventTokens'})
    
    # Merge window timestamp starts and ends to formatted predictor dataframe
    numeric_date_event_predictors = numeric_date_event_predictors.merge(cleaned_study_tokens_df[['GUPI','TimeStampStart','TimeStampEnd','WindowIdx']],how='left',on='GUPI')

    # First, isolate events which finish before the date ICU admission and combine end tokens
    baseline_numeric_date_event_predictors = numeric_date_event_predictors[numeric_date_event_predictors.WindowIdx == 1]
    baseline_numeric_date_event_predictors = baseline_numeric_date_event_predictors[baseline_numeric_date_event_predictors.Date.dt.date < baseline_numeric_date_event_predictors.TimeStampStart.dt.date].reset_index(drop=True)
    baseline_numeric_date_event_predictors = baseline_numeric_date_event_predictors.drop(columns=['Date','TimeStampStart','TimeStampEnd'])
    baseline_numeric_date_event_predictors = baseline_numeric_date_event_predictors.groupby(['GUPI','WindowIdx'],as_index=False).NumericDateEventTokens.aggregate(lambda x: ' '.join(x))

    # Merge event tokens which finish before the date of ICU admission onto study tokens dataframe
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(baseline_numeric_date_event_predictors,how='left',on=['GUPI','WindowIdx'])
    cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericDateEventTokens.isna()] = cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericDateEventTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericDateEventTokens[~cleaned_study_tokens_df.NumericDateEventTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericDateEventTokens')

    # Then, isolate the events that fit within the given window
    numeric_date_event_predictors = numeric_date_event_predictors[(numeric_date_event_predictors.Date.dt.date <= numeric_date_event_predictors.TimeStampEnd.dt.date)&(numeric_date_event_predictors.Date.dt.date >= numeric_date_event_predictors.TimeStampStart.dt.date)].reset_index(drop=True)

    # Merge dated event tokens onto study tokens dataframe
    numeric_date_event_predictors = numeric_date_event_predictors.groupby(['GUPI','WindowIdx'],as_index=False).NumericDateEventTokens.aggregate(lambda x: ' '.join(x))
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(numeric_date_event_predictors,how='left',on=['GUPI','WindowIdx'])
    cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericDateEventTokens.isna()] = cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericDateEventTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericDateEventTokens[~cleaned_study_tokens_df.NumericDateEventTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericDateEventTokens')

    ## Numeric timestamped single-event predictors
    # Extract unique names of numeric timestamped single-event predictors from the training set
    unique_numeric_timestamp_event_predictors = numeric_timestamp_event_predictors[numeric_timestamp_event_predictors.GUPI.isin(curr_train_GUPIs)].VARIABLE.unique()
    
    # Create column for storing bin value
    numeric_timestamp_event_predictors['BIN'] = ''
    
    # For missing values, assign 'NAN' to bin value
    numeric_timestamp_event_predictors.BIN[numeric_timestamp_event_predictors.VALUE.isna()] = '_NAN'
    
    # Iterate through unique numeric timestamped single-event predictors and tokenise
    for curr_predictor in tqdm(unique_numeric_timestamp_event_predictors,'Tokenising numeric timestamped single-event predictors for fold '+str(curr_fold)):
        
        # Create a `KBinsDiscretizer` object for discretising the current predictor
        curr_ntep_kbd = KBinsDiscretizer(n_bins=BINS, encode='ordinal', strategy='quantile')
        
        # Train cuts for discretisation of the current predictor
        curr_ntep_kbd.fit(np.expand_dims(numeric_timestamp_event_predictors[(numeric_timestamp_event_predictors.VARIABLE==curr_predictor)&(numeric_timestamp_event_predictors.GUPI.isin(curr_train_GUPIs))&(~numeric_timestamp_event_predictors.VALUE.isna())].VALUE.values,1))
        
        # Discretise current predictor into bins
        numeric_timestamp_event_predictors.BIN[(numeric_timestamp_event_predictors.VARIABLE==curr_predictor)&(~numeric_timestamp_event_predictors.VALUE.isna())] = (categorizer(pd.Series((curr_ntep_kbd.transform(np.expand_dims(numeric_timestamp_event_predictors[(numeric_timestamp_event_predictors.VARIABLE==curr_predictor)&(~numeric_timestamp_event_predictors.VALUE.isna())].VALUE.values,1))+1).squeeze()),100)).str.replace(r'\s+','',regex=True).values
        
    # If a predictor has been neglected, replace with value
    numeric_timestamp_event_predictors.BIN[numeric_timestamp_event_predictors.BIN==''] = numeric_timestamp_event_predictors.VALUE[numeric_timestamp_event_predictors.BIN==''].astype(str).str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)
    
    # Create tokens from each variable and bin value
    numeric_timestamp_event_predictors['TOKEN'] = numeric_timestamp_event_predictors.VARIABLE + '_BIN' + numeric_timestamp_event_predictors.BIN
    
    # Concatenate tokens from each GUPI and timestamp into a combined timestamped single-event numeric predictor token set
    numeric_timestamp_event_predictors = numeric_timestamp_event_predictors.drop_duplicates(subset=['GUPI','TimeStamp','TOKEN'],ignore_index=True).groupby(['GUPI','TimeStamp'],as_index=False).TOKEN.aggregate(lambda x: ' '.join(x)).rename(columns={'TOKEN':'NumericTimeStampEventTokens'})
    
    # Merge window timestamp starts and ends to formatted predictor dataframe
    numeric_timestamp_event_predictors = numeric_timestamp_event_predictors.merge(cleaned_study_tokens_df[['GUPI','TimeStampStart','TimeStampEnd','WindowIdx']],how='left',on='GUPI')

    # First, isolate events which finish before the ICU admission timestamp and combine end tokens
    baseline_numeric_timestamp_event_predictors = numeric_timestamp_event_predictors[numeric_timestamp_event_predictors.WindowIdx == 1]
    baseline_numeric_timestamp_event_predictors = baseline_numeric_timestamp_event_predictors[baseline_numeric_timestamp_event_predictors.TimeStamp < baseline_numeric_timestamp_event_predictors.TimeStampStart].reset_index(drop=True)
    baseline_numeric_timestamp_event_predictors = baseline_numeric_timestamp_event_predictors.drop(columns=['TimeStamp','TimeStampStart','TimeStampEnd'])
    baseline_numeric_timestamp_event_predictors = baseline_numeric_timestamp_event_predictors.groupby(['GUPI','WindowIdx'],as_index=False).NumericTimeStampEventTokens.aggregate(lambda x: ' '.join(x))

    # Merge event tokens which finish before the date of ICU admission onto study tokens dataframe
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(baseline_numeric_timestamp_event_predictors,how='left',on=['GUPI','WindowIdx'])
    cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericTimeStampEventTokens.isna()] = cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericTimeStampEventTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericTimeStampEventTokens[~cleaned_study_tokens_df.NumericTimeStampEventTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericTimeStampEventTokens')

    # Then, isolate the events that fit within the given window
    numeric_timestamp_event_predictors = numeric_timestamp_event_predictors[(numeric_timestamp_event_predictors.TimeStamp <= numeric_timestamp_event_predictors.TimeStampEnd)&(numeric_timestamp_event_predictors.TimeStamp >= numeric_timestamp_event_predictors.TimeStampStart)].reset_index(drop=True)

    # Merge timestamped event tokens onto study tokens dataframe
    numeric_timestamp_event_predictors = numeric_timestamp_event_predictors.groupby(['GUPI','WindowIdx'],as_index=False).NumericTimeStampEventTokens.aggregate(lambda x: ' '.join(x))
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(numeric_timestamp_event_predictors,how='left',on=['GUPI','WindowIdx'])
    cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericTimeStampEventTokens.isna()] = cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericTimeStampEventTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericTimeStampEventTokens[~cleaned_study_tokens_df.NumericTimeStampEventTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericTimeStampEventTokens')
    
    ## Iterate through and tokens
    # Partition categorical token rows among cores
    s = [cleaned_study_tokens_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
    s[:(cleaned_study_tokens_df.shape[0] - sum(s))] = [over+1 for over in s[:(cleaned_study_tokens_df.shape[0] - sum(s))]]    
    end_idx = np.cumsum(s)
    start_idx = np.insert(end_idx[:-1],0,0)
    windows_per_core = [(cleaned_study_tokens_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Cleaning token dataframe for fold '+str(curr_fold)) for idx in range(len(start_idx))]

    # Inspect each token row in parallel to ensure unique tokens
    with multiprocessing.Pool(NUM_CORES) as pool:
        cleaned_study_tokens_df = pd.concat(pool.starmap(clean_token_rows, windows_per_core),ignore_index=True).sort_values(by=['GUPI','WindowIdx']).reset_index(drop=True)

    # Create an ordered dictionary to create a token vocabulary from the training set
    training_token_list = (' '.join(cleaned_study_tokens_df[cleaned_study_tokens_df.GUPI.isin(curr_train_GUPIs)].TOKENS)).split(' ') + cleaned_study_tokens_df[cleaned_study_tokens_df.GUPI.isin(curr_train_GUPIs)].DaysSinceAdm.values.tolist() + cleaned_study_tokens_df[cleaned_study_tokens_df.GUPI.isin(curr_train_GUPIs)].TimeOfDay.values.tolist()
    if ('' in training_token_list):
        training_token_list = list(filter(lambda a: a != '', training_token_list))
    train_token_freqs = OrderedDict(Counter(training_token_list).most_common())
    
    # Build and save vocabulary (PyTorch Text) from training set tokens
    curr_vocab = vocab(train_token_freqs, min_freq=1)
    null_token = ''
    unk_token = '<unk>'
    if null_token not in curr_vocab: curr_vocab.insert_token(null_token, 0)
    if unk_token not in curr_vocab: curr_vocab.insert_token(unk_token, len(curr_vocab))
    curr_vocab.set_default_index(curr_vocab[unk_token])
    cp.dump(curr_vocab, open(os.path.join(fold_dir,'token_dictionary.pkl'), "wb" ))
    
    # Convert token set to indices
    cleaned_study_tokens_df['VocabIndex'] = [curr_vocab.lookup_indices(cleaned_study_tokens_df.TOKENS[curr_row].split(' ')) for curr_row in tqdm(range(cleaned_study_tokens_df.shape[0]),desc='Converting study tokens to indices for fold '+str(curr_fold))]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns='TOKENS')
    cleaned_study_tokens_df['VocabDaysSinceAdmIndex'] = curr_vocab.lookup_indices(cleaned_study_tokens_df.DaysSinceAdm.values.tolist())
    cleaned_study_tokens_df['VocabTimeOfDayIndex'] = curr_vocab.lookup_indices(cleaned_study_tokens_df.TimeOfDay.values.tolist())   
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns=['DaysSinceAdm','TimeOfDay'])
    
    # Split token set into training, validation, and testing sets
    train_tokens = cleaned_study_tokens_df[cleaned_study_tokens_df.GUPI.isin(curr_train_GUPIs)].reset_index(drop=True)
    val_tokens = cleaned_study_tokens_df[cleaned_study_tokens_df.GUPI.isin(curr_val_GUPIs)].reset_index(drop=True)
    test_tokens = cleaned_study_tokens_df[cleaned_study_tokens_df.GUPI.isin(curr_test_GUPIs)].reset_index(drop=True)

    # Save index sets
    train_tokens.to_pickle(os.path.join(fold_dir,'training_indices.pkl'))
    val_tokens.to_pickle(os.path.join(fold_dir,'validation_indices.pkl'))
    test_tokens.to_pickle(os.path.join(fold_dir,'testing_indices.pkl'))
    
### V. Categorize tokens from dictionaries for characterization
## Iterate through folds and tokenise numeric predictors per distributions in the training set
for curr_fold in tqdm(cv_splits.FOLD.unique(),'Iterating through folds of cross-validation for token categorization'):
       
    # Create a subdirectory for the current fold
    fold_dir = os.path.join(tokens_dir,'fold'+str(curr_fold))
    
    ## Extract current training, validation, and testing set GUPIs
    curr_fold_splits = cv_splits[(cv_splits.FOLD==curr_fold)].reset_index(drop=True)
    curr_train_GUPIs = curr_fold_splits[curr_fold_splits.SET=='train'].GUPI.unique()
    curr_val_GUPIs = curr_fold_splits[curr_fold_splits.SET=='val'].GUPI.unique()
    curr_test_GUPIs = curr_fold_splits[curr_fold_splits.SET=='test'].GUPI.unique()
    
    ## Categorize token vocabulary from current fold
    # Load current fold vocabulary
    curr_vocab = cp.load(open(os.path.join(fold_dir,'token_dictionary.pkl'),"rb"))

    # Create dataframe version of vocabulary
    curr_vocab_df = pd.DataFrame({'VocabIndex':list(range(len(curr_vocab))),'Token':curr_vocab.get_itos()})
    
    # Parse out `BaseToken` and `Value` from `Token`
    curr_vocab_df['BaseToken'] = curr_vocab_df.Token.str.split('_').str[0]
    curr_vocab_df['Value'] = curr_vocab_df.Token.str.split('_',n=1).str[1].fillna('')
    
    # Determine wheter tokens represent missing values
    curr_vocab_df['Missing'] = curr_vocab_df.Token.str.endswith('_NAN')
    
    # Determine whether tokens are numeric
    curr_vocab_df['Numeric'] = curr_vocab_df.Token.str.contains('_BIN')
    
    # Determine whether tokens are baseline or discharge
    curr_vocab_df['Baseline'] = curr_vocab_df.Token.str.startswith('Baseline')
    curr_vocab_df['Discharge'] = curr_vocab_df.Token.str.startswith('Discharge')
    
    # For baseline and discharge tokens, remove prefix from `BaseToken` entry
    curr_vocab_df.BaseToken[curr_vocab_df.Baseline] = curr_vocab_df.BaseToken[curr_vocab_df.Baseline].str.replace('Baseline','',1,regex=False)
    curr_vocab_df.BaseToken[curr_vocab_df.Discharge] = curr_vocab_df.BaseToken[curr_vocab_df.Discharge].str.replace('Discharge','',1,regex=False)

    # Load manually corrected `BaseToken` categorization key
    base_token_key = pd.read_excel('/home/sb2406/rds/hpc-work/tokens/base_token_keys.xlsx')
    base_token_key['BaseToken'] = base_token_key['BaseToken'].fillna('')
    
    # Merge base token key information to dataframe version of vocabulary
    curr_vocab_df = curr_vocab_df.merge(base_token_key,how='left',on=['BaseToken','Numeric','Baseline','Discharge'])
    
    # Load index sets for current fold
    train_inidices = pd.read_pickle(os.path.join(fold_dir,'training_indices.pkl'))
    val_inidices = pd.read_pickle(os.path.join(fold_dir,'validation_indices.pkl'))
    test_inidices = pd.read_pickle(os.path.join(fold_dir,'testing_indices.pkl'))
    
    # Add set indicator and combine index sets for current fold
    train_inidices['Set'] = 'train'
    val_inidices['Set'] = 'val'
    test_inidices['Set'] = 'test'
    indices_df = pd.concat([train_inidices,val_inidices,test_inidices],ignore_index=True)
    
    # Partition training indices among cores and calculate token info in parallel
    s = [indices_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
    s[:(indices_df.shape[0] - sum(s))] = [over+1 for over in s[:(indices_df.shape[0] - sum(s))]]
    end_idx = np.cumsum(s)
    start_idx = np.insert(end_idx[:-1],0,0)
    index_splits = [(indices_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),curr_vocab_df,False,True,'Characterising tokens in study windows for fold '+str(curr_fold)) for idx in range(len(start_idx))]
    with multiprocessing.Pool(NUM_CORES) as pool:
        study_window_token_info = pd.concat(pool.starmap(get_token_info, index_splits),ignore_index=True)
    
    # Save calculated token information into current fold directory
    study_window_token_info.to_pickle(os.path.join(fold_dir,'token_type_counts.pkl'))

    # Partition training indices among cores and calculate token incidence info in parallel
    s = [len(indices_df.GUPI.unique()) // NUM_CORES for _ in range(NUM_CORES)]
    s[:(len(indices_df.GUPI.unique()) - sum(s))] = [over+1 for over in s[:(len(indices_df.GUPI.unique()) - sum(s))]]
    end_idx = np.cumsum(s)
    start_idx = np.insert(end_idx[:-1],0,0)
    index_splits = [(indices_df[indices_df.GUPI.isin(indices_df.GUPI.unique()[start_idx[idx]:end_idx[idx]])].reset_index(drop=True),curr_vocab,curr_vocab_df,False,True,'Counting the incidences of tokens for fold '+str(curr_fold)) for idx in range(len(start_idx))]
    with multiprocessing.Pool(NUM_CORES) as pool:
        token_patient_incidences = pd.concat(pool.starmap(count_token_incidences, index_splits),ignore_index=True)
    
    # Save token incidence information into current fold directory
    token_patient_incidences['Fold'] = curr_fold
    token_patient_incidences.to_pickle(os.path.join(fold_dir,'token_incidences_per_patient.pkl'))

#     # Calculate number of unique patients per non-missing token
#     patients_per_token = token_patient_incidences.groupby('Token',as_index=False).GUPI.count().sort_values(by=['GUPI','Token'],ascending=[False,True]).reset_index(drop=True).rename(columns={'GUPI':'PatientCount'})
    
#     # Calculate number of unique non-missing tokens per patient
#     unique_tokens_per_patient = token_patient_incidences.groupby('GUPI',as_index=False).Token.count().sort_values(by=['Token','GUPI'],ascending=[False,True]).reset_index(drop=True).rename(columns={'Token':'UniqueTokenCount'})
    
#     # Calculate total number of instances per token
#     instances_per_token = token_patient_incidences.groupby('Token',as_index=False).Count.sum().sort_values(by=['Count','Token'],ascending=[False,True]).reset_index(drop=True).rename(columns={'Count':'TotalCount'})

### VI. Create full list of tokens for exploration
## Iterate through folds to load token dictionaries per fold
# Initialize empty list for storing tokens
compiled_tokens_list = []

# Iterate through folds
for curr_fold in tqdm(cv_splits.FOLD.unique(),'Iterating through folds of cross-validation for vocabulary collection'):
           
    # Create a subdirectory for the current fold
    fold_dir = os.path.join(tokens_dir,'fold'+str(curr_fold))
    
    # Load current fold vocabulary
    curr_vocab = cp.load(open(os.path.join(fold_dir,'token_dictionary.pkl'),"rb"))
    
    # Append tokens from current vocabulary to running list
    compiled_tokens_list.append(curr_vocab.get_itos())

# Flatten list of token lists
compiled_tokens_list = np.unique(list(itertools.chain.from_iterable(compiled_tokens_list)))

## Create characterised dataframe of all possible tokens
# Initialise dataframe
full_token_keys = pd.DataFrame({'Token':compiled_tokens_list})

# Parse out `BaseToken` and `Value` from `Token`
full_token_keys['BaseToken'] = full_token_keys.Token.str.split('_').str[0]
full_token_keys['Value'] = full_token_keys.Token.str.split('_',n=1).str[1].fillna('')

# Determine wheter tokens represent missing values
full_token_keys['Missing'] = full_token_keys.Token.str.endswith('_NAN')

# Determine whether tokens are numeric
full_token_keys['Numeric'] = full_token_keys.Token.str.contains('_BIN')

# Determine whether tokens are baseline or discharge
full_token_keys['Baseline'] = full_token_keys.Token.str.startswith('Baseline')
full_token_keys['Discharge'] = full_token_keys.Token.str.startswith('Discharge')

# For baseline and discharge tokens, remove prefix from `BaseToken` entry
full_token_keys.BaseToken[full_token_keys.Baseline] = full_token_keys.BaseToken[full_token_keys.Baseline].str.replace('Baseline','',1,regex=False)
full_token_keys.BaseToken[full_token_keys.Discharge] = full_token_keys.BaseToken[full_token_keys.Discharge].str.replace('Discharge','',1,regex=False)

# Load manually corrected `BaseToken` categorization key
base_token_key = pd.read_excel('/home/sb2406/rds/hpc-work/tokens/base_token_keys.xlsx')
base_token_key['BaseToken'] = base_token_key['BaseToken'].fillna('')

# Merge base token key information to dataframe version of vocabulary
full_token_keys = full_token_keys.merge(base_token_key,how='left',on=['BaseToken','Numeric','Baseline','Discharge'])

## Manually add 'Missing' status for other unknown tokens
# Convert all code "88" (unknowns) to missing status except for features with feasible 88 values
full_token_keys.Missing[(full_token_keys.Value=='088')&(~full_token_keys.BaseToken.isin(['DrgSubIllctUseDur','GOATTotScr']))] = True

# Convert all tokens with value containing "UNK" to missing status
full_token_keys.Missing[full_token_keys.Value.str.contains('UNK')&(full_token_keys.Ordered|full_token_keys.Binary)] = True

# Convert all code "77" (unknowns) binary variables and `TILPhysicianSatICP` to missing status
full_token_keys.Missing[(full_token_keys.Value=='077')&(full_token_keys.Binary|(full_token_keys.BaseToken=='TILPhysicianSatICP'))] = True

# Convert code "2" (unknowns) for `CTLesionDetected` to missing status
full_token_keys.Missing[(full_token_keys.Value=='002')&((full_token_keys.BaseToken=='CTLesionDetected')|(full_token_keys.BaseToken=='ERCTLesionDetected'))] = True

# If binary, convert "uninterpretable" imaging codes to Missing
full_token_keys.Missing[(full_token_keys.Value.isin(['UNINTERPRETABLE','INDETEMINATE','NOTINTERPRETED']))&(full_token_keys.Binary)] = True

## Add ordering index to Binary and Ordered variables
# If Binary or Ordered variables have less than 2 nonmissing options in dataset, remove Binary or Ordered label
CountPerBaseToken = full_token_keys.groupby(['BaseToken','Ordered','Binary'],as_index=False).Missing.aggregate({'Missings':'sum','ValueOptions':'count'})
CountPerBaseToken['NonMissings'] = CountPerBaseToken.ValueOptions - CountPerBaseToken.Missings
full_token_keys.Binary[full_token_keys.BaseToken.isin(CountPerBaseToken[CountPerBaseToken.Binary&(CountPerBaseToken.NonMissings != 2)].BaseToken.unique())] = False
full_token_keys.Ordered[full_token_keys.BaseToken.isin(CountPerBaseToken[CountPerBaseToken.Ordered&(CountPerBaseToken.NonMissings == 1)].BaseToken.unique())] = False

# Initialise column for storing ordering index for inary or Ordered variables
full_token_keys.insert(3, 'OrderIdx',np.nan)

# Create list inary or Ordered variables
binary_or_ordered_vars = full_token_keys[full_token_keys.Binary|full_token_keys.Ordered].BaseToken.unique()

# Sort full token dataframe alphabetically prior to iteration
full_token_keys = full_token_keys.sort_values(by=['BaseToken','Token'],ignore_index=True)

# Iterate through Binary or Ordered variables and order values alphabetically
for curr_var in tqdm(binary_or_ordered_vars,'Iterating through Binary or Ordered variables for ordering'):    
    full_token_keys.OrderIdx[(full_token_keys.BaseToken==curr_var)&~full_token_keys.Missing] = np.arange(full_token_keys[(full_token_keys.BaseToken==curr_var)&~full_token_keys.Missing].shape[0])

# Fix ordering index for exception-case variables
exception_vars = ['InjViolenceVictimAlcohol','InjViolenceVictimDrugs','LOCLossOfConsciousness','EDCompEventHypothermia','EDComplEventHypoxia','EDComplEventHypotension','InjViolenceOtherPartyDrugs','InjViolenceOtherPartyAlcohol']
full_token_keys.OrderIdx[(full_token_keys.BaseToken.isin(exception_vars))&(full_token_keys.OrderIdx==1)] = 3
full_token_keys.OrderIdx[(full_token_keys.BaseToken.isin(exception_vars))&(full_token_keys.OrderIdx==2)] = 4
full_token_keys.OrderIdx[(full_token_keys.BaseToken.isin(exception_vars))&(full_token_keys.OrderIdx==3)] = 2
full_token_keys.OrderIdx[(full_token_keys.BaseToken.isin(exception_vars))&(full_token_keys.OrderIdx==4)] = 1

## Save full token list dataframe if it manually edited version does not yet exist
if not os.path.exists('/home/sb2406/rds/hpc-work/tokens/full_token_keys.xlsx'):
    full_token_keys.to_excel('/home/sb2406/rds/hpc-work/tokens/full_token_keys.xlsx',index=False)

### VII. Post-hoc: collect and categorise tokens from v6-0
## Initialisation
# Set version code
VERSION = 'v6-0'

# Define model output directory
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

## Determine and save v6-0 cross-validation partitions
# Create vector of possible GOSE labels
gose_labels = ['1', '2_or_3', '4', '5', '6', '7', '8']

# Load testing set compiled predictions
test_predictions_df = pd.read_csv(os.path.join(model_dir,'compiled_test_predictions.csv'))

# Isolate unique combinations of GUPI-REPEAT-FOLD
study_test_partitions = test_predictions_df[['GUPI','TrueLabel','REPEAT','FOLD']].drop_duplicates(ignore_index=True)
study_test_partitions['SET'] = 'test'

# Convert `TrueLabel` index to GOSE label
study_test_partitions['GOSE'] = study_test_partitions.TrueLabel.apply(lambda x: gose_labels[int(x)])

# Drop and reorder columns to match `cv_splits` format
study_test_partitions = study_test_partitions[['REPEAT', 'FOLD', 'SET', 'GUPI', 'GOSE']]

# Load validation set compiled predictions
val_predictions_df = pd.read_csv(os.path.join(model_dir,'compiled_val_predictions.csv'))

# Isolate unique combinations of GUPI-REPEAT-FOLD
study_val_partitions = val_predictions_df[['GUPI','TrueLabel','REPEAT','FOLD']].drop_duplicates(ignore_index=True)
study_val_partitions['SET'] = 'val'

# Convert `TrueLabel` index to GOSE label
study_val_partitions['GOSE'] = study_val_partitions.TrueLabel.apply(lambda x: gose_labels[int(x)])

# Drop and reorder columns to match `cv_splits` format
study_val_partitions = study_val_partitions[['REPEAT', 'FOLD', 'SET', 'GUPI', 'GOSE']]

# Concatenate the testing and validation set partitions
study_test_val_partitions = pd.concat([study_test_partitions,study_val_partitions],ignore_index=True)

# Create list of unique patients in the study
uniq_GUPIs = study_test_val_partitions.GUPI.unique()

# Iterate through CV partitions and determine patients who were in the training set
uniq_CV_partitions = study_test_val_partitions[['REPEAT','FOLD']].drop_duplicates(ignore_index=True)
study_train_partitions = []
for curr_partition in tqdm(range(uniq_CV_partitions.shape[0]),'Iterating through CV partitions to determine training set population'):

    # Extract current CV partition parameters
    curr_repeat = uniq_CV_partitions.REPEAT[curr_partition]
    curr_fold = uniq_CV_partitions.FOLD[curr_partition]

    # Determine GUPIs that are in the testing or validation set of this partition
    test_val_GUPIs = study_test_val_partitions[(study_test_val_partitions.REPEAT == curr_repeat)&(study_test_val_partitions.FOLD == curr_fold)].GUPI.tolist()

    # Extract remaining GUPIs and their GOSE labels
    curr_train_partitions = study_test_val_partitions[~study_test_val_partitions.GUPI.isin(test_val_GUPIs)][['GUPI','GOSE']].drop_duplicates(ignore_index=True)

    # Add current repeat, fold, and set information
    curr_train_partitions['REPEAT'] = curr_repeat
    curr_train_partitions['FOLD'] = curr_fold
    curr_train_partitions['SET'] = 'train'

    # Rectify order of columns and append to running list
    curr_train_partitions = curr_train_partitions[['REPEAT', 'FOLD', 'SET', 'GUPI', 'GOSE']]
    study_train_partitions.append(curr_train_partitions)

study_train_partitions = pd.concat(study_train_partitions,ignore_index=True)

# Concatenate testing/validation set partitions with the training set partitions to form legacy CV splits dataframe
legacy_cv_splits = pd.concat([study_test_val_partitions,study_train_partitions],ignore_index=True).sort_values(by=['REPEAT','FOLD','SET','GUPI'],ignore_index=True)

# Save legacy CV splits dataframe
legacy_cv_splits.to_csv('../legacy_cross_validation_splits.csv',index=False)

## Create a dictionary of all available tokens in version v6-0
# Identify all token dictionary files in the tokens directory
legacy_vocab_files = []
for path in Path('/home/sb2406/rds/hpc-work/tokens').rglob('from_adm_strategy_abs_token_dictionary.pkl'):
    legacy_vocab_files.append(str(path.resolve()))

# Load and concatenate all legacy tokens
legacy_tokens = [cp.load(open(f,"rb")).get_itos() for f in tqdm(legacy_vocab_files,'Loading tokens from all vocab files in v6-0')]

# Flatten list of token lists
legacy_tokens = np.unique(list(itertools.chain.from_iterable(legacy_tokens)))

# Initialise dataframe
legacy_full_token_keys = pd.DataFrame({'Token':legacy_tokens})

# Determine whether tokens are baseline
legacy_full_token_keys['Baseline'] = legacy_full_token_keys['Token'].str.startswith('Baseline')

# Determine whether tokens are numeric
legacy_full_token_keys['Numeric'] = legacy_full_token_keys['Token'].str.contains('_BIN')

# Determine wheter tokens represent missing values
legacy_full_token_keys['Missing'] = ((legacy_full_token_keys.Numeric)&(legacy_full_token_keys['Token'].str.endswith('_BIN_missing')))|((~legacy_full_token_keys.Numeric)&(legacy_full_token_keys['Token'].str.endswith('_NA')))

# Create empty column for predictor base token
legacy_full_token_keys['BaseToken'] = ''

# For numeric tokens, extract the portion of the string before '_BIN' as the BaseToken
legacy_full_token_keys.BaseToken[legacy_full_token_keys.Numeric] = legacy_full_token_keys.Token[legacy_full_token_keys.Numeric].str.replace('\\_BIN.*','',1,regex=True)

# For non-numeric tokens, extract everything before the final underscore, if one exists, as the BaseToken
legacy_full_token_keys.BaseToken[~legacy_full_token_keys.Numeric] = legacy_full_token_keys.Token[~legacy_full_token_keys.Numeric].str.replace('_[^_]*$','',1,regex=True)

# For baseline tokens, remove the "Baseline" prefix in the BaseToken
legacy_full_token_keys.BaseToken[legacy_full_token_keys.Baseline] = legacy_full_token_keys.BaseToken[legacy_full_token_keys.Baseline].str.replace('Baseline','',1,regex=False)

# Remove underscores from `BaseToken` values if they stil exist
legacy_full_token_keys.BaseToken = legacy_full_token_keys.BaseToken.str.replace('_','')

## Compare v6-0 dictionary `BaseTokens` with previously categorised `BaseTokens`
# Load old study corrected `BaseToken` categorization key
old_token_dictionary = pd.read_excel('/home/sb2406/rds/hpc-work/tokens/copy_old_token_dictionary.xlsx')
old_token_dictionary.BaseToken = old_token_dictionary.BaseToken.str.replace('_','')
old_token_dictionary['BaseToken'] = old_token_dictionary['BaseToken'].fillna('')

# Extract `BaseToken` characteristics from old token dictionary
old_base_token_keys = old_token_dictionary[['BaseToken','ICUIntervention','ClinicianInput','Type']].drop_duplicates(ignore_index=True)

# Merge old base token key information to dataframe version of vocabulary
legacy_full_token_keys = legacy_full_token_keys.merge(old_base_token_keys,how='left')

# Fix instances of variables not in original set
legacy_full_token_keys['ICUIntervention'] = legacy_full_token_keys['ICUIntervention'].fillna(value=False)
legacy_full_token_keys['ClinicianInput'] = legacy_full_token_keys['ClinicianInput'].fillna(value=False)
legacy_full_token_keys['Type'] = legacy_full_token_keys['Type'].fillna('Miscellaneous')

# Load new `BaseToken` dictionary to add `Ordered` and `Binary` columns
base_token_key = pd.read_excel('/home/sb2406/rds/hpc-work/tokens/base_token_keys.xlsx')
base_token_key['BaseToken'] = base_token_key['BaseToken'].fillna('')
base_token_key = base_token_key[['BaseToken','Ordered','Binary']]
base_token_key.BaseToken[base_token_key.BaseToken.str.startswith('Medication')] = base_token_key.BaseToken[base_token_key.BaseToken.str.startswith('Medication')].str.replace('Medication','')

# Merge new base token key information to dataframe version of vocabulary to get ordered and binary columns
legacy_full_token_keys = legacy_full_token_keys.merge(base_token_key,how='left')

# If variable is numeric and ordered indicator is missing, set to true
legacy_full_token_keys.Ordered[(legacy_full_token_keys.Ordered.isna())&(legacy_full_token_keys.Numeric)] = True
legacy_full_token_keys.Baseline[(legacy_full_token_keys.Baseline.isna())&(legacy_full_token_keys.Numeric)] = False

# Save legacy `BaseToken` dictionary
legacy_full_token_keys = legacy_full_token_keys[['Token','BaseToken','Missing','Numeric','Baseline','Ordered','Binary','ICUIntervention','ClinicianInput','Type']]
legacy_full_token_keys.to_excel('/home/sb2406/rds/hpc-work/tokens/pre_check_legacy_full_token_keys.xlsx',index=False)

# Load manually corrected legacy `BaseToken` dictionary
legacy_full_token_keys = pd.read_excel('/home/sb2406/rds/hpc-work/tokens/legacy_full_token_keys.xlsx')

## Categorize tokens from v6-0 dictionaries for characterization
# Iterate through unique CV partitions
legacy_cv_splits = pd.read_csv('../legacy_cross_validation_splits.csv')
uniq_CV_partitions = legacy_cv_splits[['REPEAT','FOLD']].drop_duplicates(ignore_index=True)
for curr_partition in tqdm(range(uniq_CV_partitions.shape[0]),'Iterating through CV partitions for token categorization'):

    # Extract current CV partition parameters
    curr_repeat = uniq_CV_partitions.REPEAT[curr_partition]
    curr_fold = uniq_CV_partitions.FOLD[curr_partition]

    # Create a subdirectory for the current fold
    fold_dir = os.path.join(tokens_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold))

    ## Extract current training, validation, and testing set GUPIs
    curr_fold_splits = legacy_cv_splits[(legacy_cv_splits.FOLD==curr_fold)&(legacy_cv_splits.REPEAT==curr_repeat)].reset_index(drop=True)
    curr_train_GUPIs = curr_fold_splits[curr_fold_splits.SET=='train'].GUPI.unique()
    curr_val_GUPIs = curr_fold_splits[curr_fold_splits.SET=='val'].GUPI.unique()
    curr_test_GUPIs = curr_fold_splits[curr_fold_splits.SET=='test'].GUPI.unique()

    ## Categorize token vocabulary from current fold
    # Load current fold vocabulary
    curr_vocab = cp.load(open(os.path.join(fold_dir,'from_adm_strategy_abs_token_dictionary.pkl'),"rb"))

    # Create dataframe version of vocabulary
    curr_vocab_df = pd.DataFrame({'VocabIndex':list(range(len(curr_vocab))),'Token':curr_vocab.get_itos()})

    # Determine whether tokens are baseline
    curr_vocab_df['Baseline'] = curr_vocab_df['Token'].str.startswith('Baseline')

    # Determine whether tokens are numeric
    curr_vocab_df['Numeric'] = curr_vocab_df['Token'].str.contains('_BIN')

    # Determine wheter tokens represent missing values
    curr_vocab_df['Missing'] = ((curr_vocab_df.Numeric)&(curr_vocab_df['Token'].str.endswith('_BIN_missing')))|((~curr_vocab_df.Numeric)&(curr_vocab_df['Token'].str.endswith('_NA')))

    # Create empty column for predictor base token
    curr_vocab_df['BaseToken'] = ''

    # For numeric tokens, extract the portion of the string before '_BIN' as the BaseToken
    curr_vocab_df.BaseToken[curr_vocab_df.Numeric] = curr_vocab_df.Token[curr_vocab_df.Numeric].str.replace('\\_BIN.*','',1,regex=True)

    # For non-numeric tokens, extract everything before the final underscore, if one exists, as the BaseToken
    curr_vocab_df.BaseToken[~curr_vocab_df.Numeric] = curr_vocab_df.Token[~curr_vocab_df.Numeric].str.replace('_[^_]*$','',1,regex=True)

    # For baseline tokens, remove the "Baseline" prefix in the BaseToken
    curr_vocab_df.BaseToken[curr_vocab_df.Baseline] = curr_vocab_df.BaseToken[curr_vocab_df.Baseline].str.replace('Baseline','',1,regex=False)

    # Remove underscores from `BaseToken` values if they stil exist
    curr_vocab_df.BaseToken = curr_vocab_df.BaseToken.str.replace('_','')

    # Load manually corrected legacy `Token` categorization key
    legacy_full_token_keys = pd.read_excel('/home/sb2406/rds/hpc-work/tokens/legacy_full_token_keys.xlsx')
    legacy_full_token_keys['BaseToken'] = legacy_full_token_keys['BaseToken'].fillna('')

    # Merge base token key information to dataframe version of vocabulary
    curr_vocab_df = curr_vocab_df.merge(legacy_full_token_keys[['BaseToken','Type','Ordered','Binary','ICUIntervention','ClinicianInput']].drop_duplicates(ignore_index=True),how='left')

    # Load index sets for current fold
    train_inidices = pd.read_pickle(os.path.join(fold_dir,'from_adm_strategy_abs_training_indices.pkl'))
    val_inidices = pd.read_pickle(os.path.join(fold_dir,'from_adm_strategy_abs_validation_indices.pkl'))
    test_inidices = pd.read_pickle(os.path.join(fold_dir,'from_adm_strategy_abs_testing_indices.pkl'))

    # Add set indicator and combine index sets for current fold
    train_inidices['Set'] = 'train'
    val_inidices['Set'] = 'val'
    test_inidices['Set'] = 'test'
    indices_df = pd.concat([train_inidices,val_inidices,test_inidices],ignore_index=True)

    # Partition training indices among cores and calculate token info in parallel
    s = [indices_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
    s[:(indices_df.shape[0] - sum(s))] = [over+1 for over in s[:(indices_df.shape[0] - sum(s))]]
    end_idx = np.cumsum(s)
    start_idx = np.insert(end_idx[:-1],0,0)
    index_splits = [(indices_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),curr_vocab_df,False,True,'Characterising tokens in study windows for repeat '+str(curr_repeat)+' and fold '+str(curr_fold)) for idx in range(len(start_idx))]
    with multiprocessing.Pool(NUM_CORES) as pool:
        study_window_token_info = pd.concat(pool.starmap(get_legacy_token_info, index_splits),ignore_index=True)

    # Save calculated token information into current fold directory
    study_window_token_info.to_pickle(os.path.join(fold_dir,'token_type_counts.pkl'))

    # Partition training indices among cores and calculate token incidence info in parallel
    s = [len(indices_df.GUPI.unique()) // NUM_CORES for _ in range(NUM_CORES)]
    s[:(len(indices_df.GUPI.unique()) - sum(s))] = [over+1 for over in s[:(len(indices_df.GUPI.unique()) - sum(s))]]
    end_idx = np.cumsum(s)
    start_idx = np.insert(end_idx[:-1],0,0)
    index_splits = [(indices_df[indices_df.GUPI.isin(indices_df.GUPI.unique()[start_idx[idx]:end_idx[idx]])].reset_index(drop=True),curr_vocab,curr_vocab_df,False,True,'Counting the incidences of tokens for repeat '+str(curr_repeat)+' and fold '+str(curr_fold)) for idx in range(len(start_idx))]
    with multiprocessing.Pool(NUM_CORES) as pool:
        token_patient_incidences = pd.concat(pool.starmap(count_token_incidences, index_splits),ignore_index=True)

    # Save token incidence information into current fold directory
    token_patient_incidences['Repeat'] = curr_repeat
    token_patient_incidences['Fold'] = curr_fold
    token_patient_incidences.to_pickle(os.path.join(fold_dir,'token_incidences_per_patient.pkl'))

#     # Calculate number of unique patients per non-missing token
#     patients_per_token = token_patient_incidences.groupby('Token',as_index=False).GUPI.count().sort_values(by=['GUPI','Token'],ascending=[False,True]).reset_index(drop=True).rename(columns={'GUPI':'PatientCount'})
    
#     # Calculate number of unique non-missing tokens per patient
#     unique_tokens_per_patient = token_patient_incidences.groupby('GUPI',as_index=False).Token.count().sort_values(by=['Token','GUPI'],ascending=[False,True]).reset_index(drop=True).rename(columns={'Token':'UniqueTokenCount'})
    
#     # Calculate total number of instances per token
#     instances_per_token = token_patient_incidences.groupby('Token',as_index=False).Count.sum().sort_values(by=['Count','Token'],ascending=[False,True]).reset_index(drop=True).rename(columns={'Count':'TotalCount'})