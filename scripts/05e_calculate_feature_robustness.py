#### Master Script 05e: Calculate feature robustness check values in parallel ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Perform robustness checks for ordered and unordered variables based on provided resample row index
# III. Calculate Kendall's Tau values between ordered token values and corresponding TimeSHAP values 
# IV. Calculate median TimeSHAP values for unordered tokens

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
from sklearn.utils import resample

# Custom methods
from functions.analysis import calc_tau

# Set version code
VERSION = 'v7-0'

# Load cross-validation splits of study population
cv_splits = pd.read_csv('../cross_validation_splits.csv')
partitions = cv_splits[['FOLD']].drop_duplicates().reset_index(drop=True)
study_GUPI_GOSE = cv_splits[['GUPI','GOSE']].drop_duplicates()

# Define number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Define a directory for the storage of model interpretation values
interp_dir = '/home/sb2406/rds/hpc-work/model_interpretations/'+VERSION

# Define a directory for the storage of TimeSHAP values
shap_dir = os.path.join(interp_dir,'timeSHAP')

# Define the directory in which tokens are stored
token_dir='/home/sb2406/rds/hpc-work/tokens'

# Load keys associated with full token set
full_token_keys = pd.read_excel(os.path.join(token_dir,'full_token_keys.xlsx'))

# Load compiled TimeSHAP values
compiled_timeSHAP_values = pd.read_pickle(os.path.join(shap_dir,'compiled_timeSHAP_values.pkl'))

# Load bootstrapping resample dataframe for robustness calculations
bs_resamples = pd.read_pickle(os.path.join(shap_dir,'robustness_bs_resamples.pkl'))

# Create directory to store resampled robustness check calculations
robust_bs_dir = os.path.join(shap_dir,'robustness_bootstrapping')
os.makedirs(robust_bs_dir,exist_ok=True)

### II. Perform robustness checks for ordered and unordered variables based on provided resample row index
# Argument-induced bootstrapping functions
def main(array_task_id):
    
    # Extract current bootstrapping resample parameters    
    curr_rs_idx = bs_resamples.RESAMPLE_IDX[array_task_id]
    curr_GUPIs = bs_resamples.GUPIs[array_task_id]
    
    # Restrict TimeSHAP dataframe to GUPIs in current resample
    curr_bs_timeSHAP = compiled_timeSHAP_values[compiled_timeSHAP_values.GUPI.isin(curr_GUPIs)].reset_index(drop=True)
    
    # Merge full token key information to current bootstrap timeSHAP values
    curr_bs_timeSHAP = curr_bs_timeSHAP.merge(full_token_keys,how='left')

    # For unknown value tokens, fill in missing characterisation information
    curr_bs_timeSHAP.Type[curr_bs_timeSHAP.Token.str.startswith('<unk>_')] = 'Unknown'
    
    ### III. Calculate Kendall's Tau values between ordered token values and corresponding TimeSHAP values 
    ## Determine the calculation configurations for the current resample
    # Find, for each `BaseToken`, the number of nonmissing tokens per possible `OrderIdx`
    order_idx_counts = curr_bs_timeSHAP[['TUNE_IDX','Threshold','BaseToken','OrderIdx']].groupby(['TUNE_IDX','Threshold','BaseToken'],as_index=False).value_counts().sort_values(by=['TUNE_IDX','Threshold','BaseToken','OrderIdx'])

    # Find, for each `BaseToken`, the number of available distinct tokens and total datapoints
    summarised_order_idx_values = order_idx_counts.groupby(['TUNE_IDX','Threshold','BaseToken'],as_index=False)['count'].aggregate({'UniqueTokens':'count','TotalPoints':'sum'})
    order_idx_counts = order_idx_counts.merge(summarised_order_idx_values,how='left')

    # Remove all `BaseToken`s for which there is not more than one unique token
    order_idx_counts = order_idx_counts[order_idx_counts.UniqueTokens>1].reset_index(drop=True)

    # Create dataframe of unique combinations of Kendall Tau analysis configurations
    uniq_tau_configs = order_idx_counts[['TUNE_IDX','Threshold','BaseToken']].drop_duplicates(ignore_index=True)
    
    ## Calculate and save Kendall's Tau values
    # Filter current TimeSHAP dataframe to fit feasible Kendall Tau configurations
    filt_tau_timeSHAP = curr_bs_timeSHAP.merge(uniq_tau_configs,how='inner').dropna(subset='OrderIdx').reset_index(drop=True)

    # Remove unnecessary columns to expedite processing
    filt_tau_timeSHAP = filt_tau_timeSHAP[['TUNE_IDX','Threshold','BaseToken','Token','OrderIdx','SHAP']]
    
    # Calculate Tau values by group
    curr_tau_values = filt_tau_timeSHAP.groupby(['TUNE_IDX','Threshold','BaseToken'],as_index=False).apply(calc_tau).rename(columns={None:'Tau'}).fillna(value={'Tau':0})
    
    # Add resample index to Tau values dataframe before saving
    curr_tau_values['RESAMPLE_IDX'] = curr_rs_idx
    
    # Save Kendall's Tau values
    curr_tau_values.to_pickle(os.path.join(robust_bs_dir,'ordered_tau_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))
    
    ### IV. Calculate median TimeSHAP values for unordered tokens
    ## Isolate the unordered or missing value tokens in current resample
    # Merge unique combinations of Kendall Tau analysis configurations to current resample TimeSHAP set
    curr_bs_timeSHAP = curr_bs_timeSHAP.merge(uniq_tau_configs,how='left',indicator=True)
    
    # Remove all tokens included in the Kendall's Tau analysis, if the Ordered Idx is nonmissing
    filt_median_timeSHAP = curr_bs_timeSHAP[(curr_bs_timeSHAP._merge!='both')|(curr_bs_timeSHAP.OrderIdx.isna())].reset_index(drop=True)
    
    # Remove tokens with "Unknown" type
    filt_median_timeSHAP = filt_median_timeSHAP[filt_median_timeSHAP.Type!='Unknown'].reset_index(drop=True)
    
    ## Calculate and save median TimeSHAP values per token
    # Calculate median SHAP values by group
    curr_median_values = filt_median_timeSHAP.groupby(['TUNE_IDX','Threshold','BaseToken','Token','Missing'],as_index=False).SHAP.aggregate({'medianSHAP':'median','Instances':'count'})
    
    # Add resample index to median values dataframe before saving
    curr_median_values['RESAMPLE_IDX'] = curr_rs_idx
    
    # Save median SHAP values
    curr_median_values.to_pickle(os.path.join(robust_bs_dir,'unordered_median_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))    
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])
    main(array_task_id)