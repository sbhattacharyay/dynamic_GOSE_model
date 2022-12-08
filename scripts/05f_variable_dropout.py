#### Master Script 05f: Compile robustness check results and token incidences to dropout extraneous variables ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile and save bootstrapped robustness check dataframes
# III. Calculate confidence intervals on robustness check metrics
# IV. Filter out variables with low patient incidence in the dataset
# V. Among remaining ordered variables, filter out ones with insignificant Kendall's Tau
# V. Among remaining unordered variables, filter out ones with insignificant median SHAP values

### I. Initialisation
# Fundamental libraries
import os
import re
import sys
import time
import glob
import random
import shutil
import datetime
import warnings
import itertools
import numpy as np
import pandas as pd
import pickle as cp
from tqdm import tqdm
import seaborn as sns
import multiprocessing
from scipy import stats
from pathlib import Path
from shutil import rmtree
from ast import literal_eval
import matplotlib.pyplot as plt
from scipy.special import logit
from argparse import ArgumentParser
from collections import Counter, OrderedDict
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# Custom methods
from functions.model_building import load_robustness_checks

# Set version code
VERSION = 'v7-0'

# Define a directory for the storage of model interpretation values
interp_dir = '/home/sb2406/rds/hpc-work/model_interpretations/'+VERSION

# Define a directory for the storage of TimeSHAP values
shap_dir = os.path.join(interp_dir,'timeSHAP')

# Define a directory for the storage of resampled robustness check calculations
robust_bs_dir = os.path.join(shap_dir,'robustness_bootstrapping')

# Define the directory in which tokens are stored
token_dir='/home/sb2406/rds/hpc-work/tokens'

# Load keys associated with full token set
full_token_keys = pd.read_excel(os.path.join(token_dir,'full_token_keys.xlsx'))

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Define model performance directory based on version code
model_perf_dir = '/home/sb2406/rds/hpc-work/model_performance/'+VERSION

# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
partitions = cv_splits[['FOLD']].drop_duplicates().reset_index(drop=True)
study_GUPI_GOSE = cv_splits[['GUPI','GOSE']].drop_duplicates()

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

### II. Compile and save bootstrapped robustness check dataframes
## Compile and save ordered variable Kendall's Tau values
# Search for all Kendall's Tau value files
tau_files = []
for path in Path(robust_bs_dir).rglob('ordered_tau_rs_*'):
    tau_files.append(str(path.resolve()))
    
# Characterise the Kendall's Tau files found
tau_file_info_df = pd.DataFrame({'FILE':tau_files,
                                 'VERSION':[re.search('_interpretations/(.*)/timeSHAP', curr_file).group(1) for curr_file in tau_files],
                                 'RESAMPLE_IDX':[int(re.search('_rs_(.*).pkl', curr_file).group(1)) for curr_file in tau_files],
                                }).sort_values(by=['RESAMPLE_IDX']).reset_index(drop=True)

# Partition Kendall's Tau files across available cores
s = [tau_file_info_df.RESAMPLE_IDX.max() // NUM_CORES for _ in range(NUM_CORES)]
s[:(tau_file_info_df.RESAMPLE_IDX.max() - sum(s))] = [over+1 for over in s[:(tau_file_info_df.RESAMPLE_IDX.max() - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
tau_files_per_core = [(tau_file_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling TimeSHAP Kendall Tau values for ordered predictors') for idx in range(len(start_idx))]

# Load TimeSHAP Kendall's Tau dataframes in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_tau_values = pd.concat(pool.starmap(load_robustness_checks, tau_files_per_core),ignore_index=True)

# Save compiled TimeSHAP Kendall's Tau dataframe
compiled_tau_values.to_pickle(os.path.join(shap_dir,'compiled_tau_values.pkl'))

## Compile and save unordered variable median TimeSHAP values
# Search for all median TimeSHAP files
median_files = []
for path in Path(robust_bs_dir).rglob('unordered_median_rs_*'):
    median_files.append(str(path.resolve()))
    
# Characterise the median TimeSHAP files found
median_file_info_df = pd.DataFrame({'FILE':median_files,
                                    'VERSION':[re.search('_interpretations/(.*)/timeSHAP', curr_file).group(1) for curr_file in median_files],
                                    'RESAMPLE_IDX':[int(re.search('_rs_(.*).pkl', curr_file).group(1)) for curr_file in median_files],
                                   }).sort_values(by=['RESAMPLE_IDX']).reset_index(drop=True)

# Partition median TimeSHAP files across available cores
s = [median_file_info_df.RESAMPLE_IDX.max() // NUM_CORES for _ in range(NUM_CORES)]
s[:(median_file_info_df.RESAMPLE_IDX.max() - sum(s))] = [over+1 for over in s[:(median_file_info_df.RESAMPLE_IDX.max() - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
median_files_per_core = [(median_file_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling median TimeSHAP values for unordered predictors') for idx in range(len(start_idx))]

# Load median TimeSHAP dataframes in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_median_values = pd.concat(pool.starmap(load_robustness_checks, median_files_per_core),ignore_index=True)

# Save compiled median TimeSHAP dataframe
compiled_median_values.to_pickle(os.path.join(shap_dir,'compiled_median_values.pkl'))

## Delete robustness check directory once all files have been accounted for
shutil.rmtree(robust_bs_dir)

### III. Calculate confidence intervals on robustness check metrics
## Kendall's Tau values for ordered variables
# Load compiled Kendall's Tau values
compiled_tau_values = pd.read_pickle(os.path.join(shap_dir,'compiled_tau_values.pkl'))

# Calculate confidence intervals at different percentiles
CI_tau_values = compiled_tau_values.groupby(['TUNE_IDX','Threshold','BaseToken'],as_index=False)['Tau'].aggregate({'Q005':lambda x: np.quantile(x,.005),'Q025':lambda x: np.quantile(x,.025),'Q05':lambda x: np.quantile(x,.05),'median':np.median,'Q95':lambda x: np.quantile(x,.95),'Q975':lambda x: np.quantile(x,.975),'Q995':lambda x: np.quantile(x,.995),'mean':np.mean,'std':np.std,'resamples':'count'}).reset_index(drop=True)

# Add significance indicators for variables at different confidence intervals
CI_tau_values['Significant01%'] = ~((CI_tau_values['Q005']<=0)&(CI_tau_values['Q995']>=0))
CI_tau_values['Significant05%'] = ~((CI_tau_values['Q025']<=0)&(CI_tau_values['Q975']>=0))
CI_tau_values['Significant10%'] = ~((CI_tau_values['Q05']<=0)&(CI_tau_values['Q95']>=0))

# Save Kendall's Tau confidence intervals
CI_tau_values.to_pickle(os.path.join(shap_dir,'tau_values_CI.pkl'))

## Median TimeSHAP values for unordered variables
# Load compiled median TimeSHAP values
compiled_median_values = pd.read_pickle(os.path.join(shap_dir,'compiled_median_values.pkl'))

# Calculate confidence intervals at different percentiles
CI_median_values = compiled_median_values.groupby(['TUNE_IDX','Threshold','BaseToken','Token','Missing'],as_index=False)['medianSHAP'].aggregate({'Q005':lambda x: np.quantile(x,.005),'Q025':lambda x: np.quantile(x,.025),'Q05':lambda x: np.quantile(x,.05),'median':np.median,'Q95':lambda x: np.quantile(x,.95),'Q975':lambda x: np.quantile(x,.975),'Q995':lambda x: np.quantile(x,.995),'mean':np.mean,'std':np.std,'resamples':'count'}).reset_index(drop=True)

# Add significance indicators for variables at different confidence intervals
CI_median_values['Significant01%'] = ~((CI_median_values['Q005']<=0)&(CI_median_values['Q995']>=0))
CI_median_values['Significant05%'] = ~((CI_median_values['Q025']<=0)&(CI_median_values['Q975']>=0))
CI_median_values['Significant10%'] = ~((CI_median_values['Q05']<=0)&(CI_median_values['Q95']>=0))

# Save median TimeSHAP confidence intervals
CI_median_values.to_pickle(os.path.join(shap_dir,'median_values_CI.pkl'))

### IV. Filter out variables with low patient incidence in the dataset
## Load token-patient incidences in the first fold of cross-validation
token_patient_incidences = pd.read_pickle(os.path.join(token_dir,'fold1','token_incidences_per_patient.pkl'))

## Characterise number of patients per variable
# Merge full token set keys to token-patient incidence dataframe
token_patient_incidences = token_patient_incidences.merge(full_token_keys[['Token','BaseToken','Value','Missing']],how='left')

# Remove tokens that represent missing or unknown values
token_patient_incidences = token_patient_incidences[~token_patient_incidences.Missing].reset_index(drop=True)

# Count the number of unique patients per BaseToken
patients_per_base_token = token_patient_incidences.groupby('BaseToken',as_index=False).GUPI.aggregate({'UniquePatientCount':lambda x: len(np.unique(x))})

## Determine which variables to filter out based on set proportion of population
# Define minimum proportion of population necessary to maintain
MINIMUM_PROPORTION = 0.1

# Find variables with at least minimum number of patients to remain in the dataset
base_tokens_to_keep = patients_per_base_token[patients_per_base_token.UniquePatientCount>=(study_GUPI_GOSE.shape[0]*MINIMUM_PROPORTION)].reset_index(drop=True)

### V. Among remaining ordered variables, filter out ones with insignificant Kendall's Tau
## Prepare Kendall's Tau confidence intervals
# Load Kendall's Tau confidence intervals
CI_tau_values = pd.read_pickle(os.path.join(shap_dir,'tau_values_CI.pkl'))

# Keep only variables that passed incidence filtering and had a significant Tau value at a=1%
CI_tau_values_to_keep = CI_tau_values[(CI_tau_values.BaseToken.isin(base_tokens_to_keep.BaseToken))&(CI_tau_values['Significant01%'])].reset_index(drop=True)

# Add column to designate directionality of variable
CI_tau_values_to_keep['Direction'] = np.nan
CI_tau_values_to_keep.Direction[CI_tau_values_to_keep['median']>0] = 'Positive'
CI_tau_values_to_keep.Direction[CI_tau_values_to_keep['median']<0] = 'Negative'

# Save dataframe of variables that survive Kendall's Tau filtering
CI_tau_values_to_keep.to_csv(os.path.join(shap_dir,'sig_direction_ordered_variables.csv'),index=False)

## Isolate variables by directional effect to determine infeasible relationships
# Calculation proportion of variables across thresholds
direction_proportions = CI_tau_values_to_keep.groupby(['TUNE_IDX','BaseToken'],as_index=False).Direction.value_counts(normalize=True)

# First, find all variables with no difference in effect across thresholds
no_difference_across_thresholds = direction_proportions[direction_proportions['proportion'] == 1].reset_index(drop=True)

# Focus on the effect of variables with no difference across thresholds
no_difference_across_thresholds = no_difference_across_thresholds[['BaseToken','Direction']].drop_duplicates(ignore_index=True).sort_values(by=['BaseToken','Direction'],ignore_index=True)

# Save as excel file for manual directionality assessment
no_difference_across_thresholds.to_excel(os.path.join(shap_dir,'directionality_assessment_1.xlsx'),index=False)

# Second, find all variables with difference in effect across thresholds
difference_across_thresholds = direction_proportions[direction_proportions['proportion'] != 1].reset_index(drop=True)

# Just extract tuning configuration index and `BaseToken`
difference_across_thresholds = difference_across_thresholds[['TUNE_IDX','BaseToken']].drop_duplicates(ignore_index=True)

# Add threshold information
difference_across_thresholds = CI_tau_values_to_keep.merge(difference_across_thresholds,how='inner')[['TUNE_IDX','Threshold','BaseToken','Direction']]

# Save as excel file for manual mixed directionality assessment
difference_across_thresholds.to_excel(os.path.join(shap_dir,'directionality_assessment_2.xlsx'),index=False)

### VI. Among remaining unordered variables, filter out ones with insignificant median SHAP values
## Prepare median TimeSHAP values
# Load median TimeSHAP confidence intervals
CI_median_values = pd.read_pickle(os.path.join(shap_dir,'median_values_CI.pkl'))

# Keep only variables that passed incidence filtering
CI_median_values_to_keep = CI_median_values[CI_median_values.BaseToken.isin(base_tokens_to_keep.BaseToken)].reset_index(drop=True)

## Isolate unordered variables for which there are at least one significant token value
# Calculate number of significant nonmissing tokens per `BaseToken`
sig_tokens_per_variable = CI_median_values_to_keep[CI_median_values_to_keep.Missing==False].groupby(['TUNE_IDX','Threshold','BaseToken'],as_index=False)['Significant01%'].aggregate({'TokenCount':'count','SignificantCount':'sum'})

# Keep `BaseTokens` with at least one significant token value
sig_tokens_per_variable = sig_tokens_per_variable[sig_tokens_per_variable['SignificantCount']!=0].reset_index(drop=True)

# Remove unordered variables with at over 100 different token values
sig_tokens_per_variable = sig_tokens_per_variable[sig_tokens_per_variable.TokenCount <= 150].reset_index(drop=True)

## Filter confidence intervals of TimeSHAP median values of significant unordered variables for manual inspection
filt_CI_median_values_to_keep = CI_median_values_to_keep.merge(sig_tokens_per_variable,how='inner')

filt_CI_median_values_to_keep = filt_CI_median_values_to_keep[filt_CI_median_values_to_keep['Significant01%']&(filt_CI_median_values_to_keep.Missing==False)].reset_index(drop=True)

# Add column to designate directionality of variable
filt_CI_median_values_to_keep['Direction'] = np.nan
filt_CI_median_values_to_keep.Direction[filt_CI_median_values_to_keep['median']>0] = 'Positive'
filt_CI_median_values_to_keep.Direction[filt_CI_median_values_to_keep['median']<0] = 'Negative'

filt_CI_median_values_to_keep['Value'] = filt_CI_median_values_to_keep.Token.str.split('_',n=1).str[1].fillna('')
    

filt_CI_median_values_to_keep = filt_CI_median_values_to_keep[['TUNE_IDX', 'Threshold', 'BaseToken', 'Value', 'Direction']]

filt_CI_median_values_to_keep.to_csv(os.path.join(shap_dir,'sig_direction_unordered_variables.csv'),index=False)