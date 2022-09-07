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
from functions.token_preparation import categorizer, clean_token_rows, get_token_info, count_token_incidences

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