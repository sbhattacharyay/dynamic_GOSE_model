#### Master Script 05d: Compile TimeSHAP values calculated in parallel ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile TimeSHAP values and clean directory
# III. Partition missed significant transitions for second-pass parallel TimeSHAP calculation
# IV. Compile second-pass TimeSHAP values and clean directory
# V. Create bootstrapping resamples for calculating feature robustness
# VI. Prepare TimeSHAP values for plotting

### I. Initialisation
# Fundamental libraries
import os
import re
import sys
import time
import glob
import copy
import shutil
import random
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
from ast import literal_eval
import matplotlib.pyplot as plt
from collections import Counter
from argparse import ArgumentParser
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# PyTorch, PyTorch.Text, and Lightning-PyTorch methods
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Import TimeSHAP methods
import timeshap.explainer as tsx
import timeshap.plot as tsp
from timeshap.wrappers import TorchModelWrapper
from timeshap.utils import get_avg_score_with_avg_event

# Custom methods
from classes.datasets import DYN_ALL_PREDICTOR_SET
from models.dynamic_APM import GOSE_model, timeshap_GOSE_model
from functions.model_building import collate_batch, format_shap, format_tokens, format_time_tokens, df_to_multihot_matrix, load_timeSHAP

# Set version code
VERSION = 'v7-0'

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Define directory in which tokens are stored
tokens_dir = '/home/sb2406/rds/hpc-work/tokens'

# Load the current version tuning grid
post_tuning_grid = pd.read_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'))

# Load cross-validation split information to extract testing resamples
cv_splits = pd.read_csv('../cross_validation_splits.csv')
study_GUPIs = cv_splits[['GUPI','GOSE']].drop_duplicates()
test_splits = cv_splits[cv_splits.SET == 'test'].reset_index(drop=True)
uniq_GUPIs = test_splits.GUPI.unique()

# Define a directory for the storage of model interpretation values
interp_dir = '/home/sb2406/rds/hpc-work/model_interpretations/'+VERSION

# Define a directory for the storage of TimeSHAP values
shap_dir = os.path.join(interp_dir,'timeSHAP')

# Define a subdirectory for the storage of TimeSHAP values
sub_shap_dir = os.path.join(shap_dir,'parallel_results')

# Define a subdirectory for the storage of second-pass TimeSHAP values
second_sub_shap_dir = os.path.join(shap_dir,'second_pass_parallel_results')

# Define a subdirectory for the storage of missed TimeSHAP transitions
missed_transition_dir = os.path.join(shap_dir,'missed_transitions')

# Define a subdirectory for the storage of second-pass missed TimeSHAP transitions
second_missed_transition_dir = os.path.join(shap_dir,'second_pass_missed_transitions')

# Load and concatenate partitioned significant clinical transitions for allocated TimeSHAP calculation
timeshap_partitions = cp.load(open(os.path.join(shap_dir,'timeSHAP_partitions.pkl'),"rb"))
timeshap_partitions = [tp.assign(PARTITION_IDX=ix) for ix,tp in enumerate(timeshap_partitions)]
timeshap_partitions = pd.concat(timeshap_partitions,ignore_index=True)

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

### II. Compile TimeSHAP values and clean directory
## Find completed TimeSHAP configurations and log remaining configurations, if any
# Identify TimeSHAP dataframe files in parallel storage directory
tsx_files = []
for path in Path(os.path.join(sub_shap_dir)).rglob('timeSHAP_values_partition_idx_*'):
    tsx_files.append(str(path.resolve()))

# Characterise found TimeSHAP dataframe files
tsx_info_df = pd.DataFrame({'FILE':tsx_files,
                            'PARTITION_IDX':[int(re.search('partition_idx_(.*).pkl', curr_file).group(1)) for curr_file in tsx_files]
                           }).sort_values(by=['PARTITION_IDX']).reset_index(drop=True)

# Identify TimeSHAP significant transitions that were missed based on stored files
missed_transition_files = []
for path in Path(os.path.join(missed_transition_dir)).rglob('missing_transitions_partition_idx_*'):
    missed_transition_files.append(str(path.resolve()))

# Characterise found missing transition dataframe files
missed_info_df = pd.DataFrame({'FILE':missed_transition_files,
                               'PARTITION_IDX':[int(re.search('partition_idx_(.*).pkl', curr_file).group(1)) for curr_file in missed_transition_files]
                              }).sort_values(by=['PARTITION_IDX']).reset_index(drop=True)

# Determine partition indices that have not yet been accounted for
full_range = list(range(10000))
remaining_partition_indices = np.sort(list(set(full_range)-set(tsx_info_df.PARTITION_IDX)-set(missed_info_df.PARTITION_IDX))).tolist()

# Create partitions for TimeSHAP configurations that are unaccounted for
original_partition_list = cp.load(open(os.path.join(shap_dir,'timeSHAP_partitions.pkl'),"rb"))
remaining_timeshap_partitions = [original_partition_list[ix] for ix in tqdm(remaining_partition_indices)]

# Save remaining partitions
cp.dump(remaining_timeshap_partitions, open(os.path.join(shap_dir,'remaining_timeSHAP_partitions.pkl'), "wb" ))
cp.dump(remaining_partition_indices, open(os.path.join(shap_dir,'remaining_timeSHAP_partition_indices.pkl'), "wb" ))

## In parallel, load, compile, and save missed significant transitions
# Partition missed transition files across available cores
s = [missed_info_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
s[:(missed_info_df.shape[0] - sum(s))] = [over+1 for over in s[:(missed_info_df.shape[0] - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
missed_files_per_core = [(missed_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling missed significant transitions') for idx in range(len(start_idx))]

# Load missed signficant transition dataframes in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_missed_transitions = pd.concat(pool.starmap(load_timeSHAP, missed_files_per_core),ignore_index=True)

# Save compiled missed transitions dataframe into TimeSHAP directory
compiled_missed_transitions.to_pickle(os.path.join(shap_dir,'first_pass_missed_transitions.pkl'))

## In parallel, load, compile, and save TimeSHAP values
# Partition completed TimeSHAP files across available cores
s = [tsx_info_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
s[:(tsx_info_df.shape[0] - sum(s))] = [over+1 for over in s[:(tsx_info_df.shape[0] - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
tsx_files_per_core = [(tsx_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling TimeSHAP values') for idx in range(len(start_idx))]

# Load completed TimeSHAP dataframes in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_timeSHAP_values = pd.concat(pool.starmap(load_timeSHAP, tsx_files_per_core),ignore_index=True)
    
# Save compiled TimeSHAP values dataframe into TimeSHAP directory
compiled_timeSHAP_values.to_pickle(os.path.join(shap_dir,'first_pass_timeSHAP_values.pkl'))

## After compiling and saving values, delete individual files
# Delete missed transition files
shutil.rmtree(missed_transition_dir)

# Delete TimeSHAP value files
shutil.rmtree(sub_shap_dir)

### III. Partition missed significant transitions for second-pass parallel TimeSHAP calculation
## Partition evenly for parallel calculation
# Load missed significant points of prognostic transition
compiled_missed_transitions = pd.read_pickle(os.path.join(shap_dir,'first_pass_missed_transitions.pkl'))

# Partition evenly along number of available array tasks
max_array_tasks = 10000
s = [compiled_missed_transitions.shape[0] // max_array_tasks for _ in range(max_array_tasks)]
s[:(compiled_missed_transitions.shape[0] - sum(s))] = [over+1 for over in s[:(compiled_missed_transitions.shape[0] - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
timeshap_partitions = pd.concat([compiled_missed_transitions.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True).assign(PARTITION_IDX=idx) for idx in range(len(start_idx))],ignore_index=True)

# Save derived missed transition partitions
cp.dump(timeshap_partitions, open(os.path.join(shap_dir,'second_pass_timeSHAP_partitions.pkl'), "wb" ))

### IV. Compile second-pass TimeSHAP values and clean directory
## Find completed second-pass TimeSHAP configurations and log remaining configurations, if any
# Identify TimeSHAP dataframe files in parallel storage directory
second_tsx_files = []
for path in Path(os.path.join(second_sub_shap_dir)).rglob('timeSHAP_values_partition_idx_*'):
    second_tsx_files.append(str(path.resolve()))

# Characterise found second-pass TimeSHAP dataframe files
second_tsx_info_df = pd.DataFrame({'FILE':second_tsx_files,
                                   'PARTITION_IDX':[int(re.search('partition_idx_(.*).pkl', curr_file).group(1)) for curr_file in second_tsx_files]
                                  }).sort_values(by=['PARTITION_IDX']).reset_index(drop=True)

# Identify second-pass TimeSHAP significant transitions that were missed based on stored files
second_missed_transition_files = []
for path in Path(os.path.join(second_missed_transition_dir)).rglob('timeSHAP_values_partition_idx_*'):
    second_missed_transition_files.append(str(path.resolve()))

# Characterise found second-pass missing transition dataframe files
second_missed_info_df = pd.DataFrame({'FILE':second_missed_transition_files,
                                      'PARTITION_IDX':[int(re.search('partition_idx_(.*).pkl', curr_file).group(1)) for curr_file in second_missed_transition_files]
                                     }).sort_values(by=['PARTITION_IDX']).reset_index(drop=True)

# Determine partition indices that have not yet been accounted for
full_range = list(range(10000))
remaining_partition_indices = np.sort(list(set(full_range)-set(second_tsx_info_df.PARTITION_IDX)-set(second_missed_info_df.PARTITION_IDX))).tolist()

# Create second-pass partitions for TimeSHAP configurations that are unaccounted for
second_partition_list = pd.read_pickle(os.path.join(shap_dir,'second_pass_timeSHAP_partitions.pkl'))
remaining_timeshap_partitions = second_partition_list[second_partition_list.PARTITION_IDX.isin(remaining_partition_indices)].reset_index(drop=True)

# Save remaining partitions
remaining_timeshap_partitions.to_pickle(os.path.join(shap_dir,'second_pass_remaining_timeSHAP_partitions.pkl'))

## In parallel, load, compile, and save second-pass missed significant transitions
# Partition missed transition files across available cores
s = [second_missed_info_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
s[:(second_missed_info_df.shape[0] - sum(s))] = [over+1 for over in s[:(second_missed_info_df.shape[0] - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
missed_files_per_core = [(second_missed_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling second-pass missed significant transitions') for idx in range(len(start_idx))]

# Load second-pass missed signficant transition dataframes in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    second_pass_compiled_missed_transitions = pd.concat(pool.starmap(load_timeSHAP, missed_files_per_core),ignore_index=True)

# Save compiled second-pass missed transitions dataframe into TimeSHAP directory
second_pass_compiled_missed_transitions.to_pickle(os.path.join(shap_dir,'second_pass_missed_transitions.pkl'))

## In parallel, load, compile, and save second-pass TimeSHAP values
# Partition completed second-pass TimeSHAP files across available cores
s = [second_tsx_info_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
s[:(second_tsx_info_df.shape[0] - sum(s))] = [over+1 for over in s[:(second_tsx_info_df.shape[0] - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
tsx_files_per_core = [(second_tsx_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling second-pass TimeSHAP values') for idx in range(len(start_idx))]

# Load completed second-pass TimeSHAP dataframes in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    second_compiled_timeSHAP_values = pd.concat(pool.starmap(load_timeSHAP, tsx_files_per_core),ignore_index=True)

# Load compiled first-pass TimeSHAP values dataframe from TimeSHAP directory
first_compiled_timeSHAP_values = pd.read_pickle(os.path.join(shap_dir,'first_pass_timeSHAP_values.pkl')).rename(columns={'Feature':'Token'})

# Clean first-pass TimeSHAP values dataframe before compilation
first_compiled_timeSHAP_values = first_compiled_timeSHAP_values.drop(columns=['Random seed','NSamples'])
first_compiled_timeSHAP_values['PASS'] = 'First'
                                        
# Clean second-pass TimeSHAP values dataframe before compilation
second_compiled_timeSHAP_values = second_compiled_timeSHAP_values.rename(columns={'Feature':'Token'}).drop(columns=['Random seed','NSamples'])
second_compiled_timeSHAP_values['PASS'] = 'Second'

# Compile and save TimeSHAP values dataframe into TimeSHAP directory
compiled_timeSHAP_values = pd.concat([first_compiled_timeSHAP_values,second_compiled_timeSHAP_values],ignore_index=True).sort_values(by=['TUNE_IDX','FOLD','Threshold','GUPI','WindowIdx','Token'],ignore_index=True)
                                        
# Save compiled TimeSHAP values dataframe into TimeSHAP directory
compiled_timeSHAP_values.to_pickle(os.path.join(shap_dir,'compiled_timeSHAP_values.pkl'))

# ## After compiling and saving values, delete individual files
# # Delete second-pass missed transition files
# shutil.rmtree(second_missed_transition_dir)

# # Delete second-pass TimeSHAP value files
# shutil.rmtree(second_sub_shap_dir)

### V. Create bootstrapping resamples for calculating feature robustness
## Load compiled TimeSHAP values dataframe from TimeSHAP directory
compiled_timeSHAP_values = pd.read_pickle(os.path.join(shap_dir,'compiled_timeSHAP_values.pkl'))






### VI. Prepare TimeSHAP values for plotting
## Prepare TimeSHAP value dataframe
# Load compiled TimeSHAP values dataframe from TimeSHAP directory
compiled_timeSHAP_values = pd.read_pickle(os.path.join(shap_dir,'compiled_timeSHAP_values.pkl'))

# Average SHAP values per GUPI
summarised_timeSHAP_values = compiled_timeSHAP_values.groupby(['TUNE_IDX','GUPI','Token','Threshold'],as_index=False)['SHAP'].mean()

# Parse out `BaseToken` and `Value` from `Token`
summarised_timeSHAP_values['BaseToken'] = summarised_timeSHAP_values.Token.str.split('_').str[0]
summarised_timeSHAP_values['Value'] = summarised_timeSHAP_values.Token.str.split('_',n=1).str[1].fillna('')

# Determine wheter tokens represent missing values
summarised_timeSHAP_values['Missing'] = summarised_timeSHAP_values.Token.str.endswith('_NAN')

# Determine whether tokens are numeric
summarised_timeSHAP_values['Numeric'] = summarised_timeSHAP_values.Token.str.contains('_BIN')

# Determine whether tokens are baseline or discharge
summarised_timeSHAP_values['Baseline'] = summarised_timeSHAP_values.Token.str.startswith('Baseline')
summarised_timeSHAP_values['Discharge'] = summarised_timeSHAP_values.Token.str.startswith('Discharge')

# For baseline and discharge tokens, remove prefix from `BaseToken` entry
summarised_timeSHAP_values.BaseToken[summarised_timeSHAP_values.Baseline] = summarised_timeSHAP_values.BaseToken[summarised_timeSHAP_values.Baseline].str.replace('Baseline','',1,regex=False)
summarised_timeSHAP_values.BaseToken[summarised_timeSHAP_values.Discharge] = summarised_timeSHAP_values.BaseToken[summarised_timeSHAP_values.Discharge].str.replace('Discharge','',1,regex=False)

# Load manually corrected `BaseToken` categorization key
base_token_key = pd.read_excel('/home/sb2406/rds/hpc-work/tokens/base_token_keys.xlsx')
base_token_key['BaseToken'] = base_token_key['BaseToken'].fillna('')

# Merge base token key information to dataframe version of vocabulary
summarised_timeSHAP_values = summarised_timeSHAP_values.merge(base_token_key,how='left',on=['BaseToken','Numeric','Baseline','Discharge'])

# For initial purpose, filter out any missing values
summarised_timeSHAP_values = summarised_timeSHAP_values[~summarised_timeSHAP_values.Missing].reset_index(drop=True)

## Determine "most important" features for visualisation
# Calculate summary statistics of SHAP per Token
token_level_timeSHAP_summaries = summarised_timeSHAP_values.groupby(['TUNE_IDX','Threshold','BaseToken','Token'],as_index=False)['SHAP'].aggregate({'median':np.median,'mean':np.mean,'std':np.std,'instances':'count'}).sort_values('median').reset_index(drop=True)

# Calculate overall summary statistics of SHAP per BaseToken
basetoken_timeSHAP_summaries = token_level_timeSHAP_summaries.groupby(['TUNE_IDX','Threshold','BaseToken'],as_index=False)['median'].aggregate({'std':np.std,'min':np.min,'q1':lambda x: np.quantile(x,.25),'median':np.median,'q3':lambda x: np.quantile(x,.75),'max':np.max,'mean':np.mean, 'variable_values':'count'}).sort_values('std',ascending=False).reset_index(drop=True)

# Calculate total instances per `BaseToken` and merge information to dataframe
basetoken_timeSHAP_summaries = basetoken_timeSHAP_summaries.merge(summarised_timeSHAP_values.groupby(['TUNE_IDX','Threshold','BaseToken'],as_index=False)['GUPI'].aggregate({'total_instances':lambda x: len(np.unique(x))}),how='left',on=['TUNE_IDX','Threshold','BaseToken'])

# Filter out `BaseTokens` with less than 2 unique patients
basetoken_timeSHAP_summaries = basetoken_timeSHAP_summaries[basetoken_timeSHAP_summaries.total_instances >= 2].reset_index(drop=True)

# For each TUNE_IDX-Threshold combination, select the top 20 `BaseTokens` based on variance across values
top_variance_timeSHAP_basetokens = basetoken_timeSHAP_summaries.loc[basetoken_timeSHAP_summaries.groupby(['TUNE_IDX','Threshold'])['std'].head(20).index].reset_index(drop=True)
top_variance_timeSHAP_basetokens['RankIdx'] = top_variance_timeSHAP_basetokens.groupby(['TUNE_IDX','Threshold'])['std'].rank('dense', ascending=False)
filtered_top_variance_timeSHAP_values = summarised_timeSHAP_values.merge(top_variance_timeSHAP_basetokens[['TUNE_IDX', 'Threshold', 'BaseToken','RankIdx']],how='inner',on=['TUNE_IDX', 'Threshold', 'BaseToken']).reset_index(drop=True)

# For each TUNE_IDX-Threshold combination, select the bottom 10 `BaseTokens` based on min median token SHAP values
basetoken_timeSHAP_summaries = basetoken_timeSHAP_summaries.sort_values('min').reset_index(drop=True)
top_min_timeSHAP_basetokens = basetoken_timeSHAP_summaries.loc[basetoken_timeSHAP_summaries.groupby(['TUNE_IDX','Threshold'])['min'].head(10).index].reset_index(drop=True)
top_min_timeSHAP_basetokens['RankIdx'] = top_min_timeSHAP_basetokens.groupby(['TUNE_IDX','Threshold'])['min'].rank('dense', ascending=False) + 10

# For each TUNE_IDX-Threshold combination, select the top 10 `BaseTokens` based on max median token SHAP values that are not in bottom 10
filt_set = basetoken_timeSHAP_summaries.merge(top_min_timeSHAP_basetokens[['TUNE_IDX','Threshold','BaseToken']], on=['TUNE_IDX','Threshold','BaseToken'],how='left', indicator=True)
filt_set = filt_set[filt_set['_merge'] == 'left_only'].sort_values('max',ascending=False).drop(columns='_merge').reset_index(drop=True)
top_max_timeSHAP_basetokens = filt_set.loc[filt_set.groupby(['TUNE_IDX','Threshold'])['max'].head(10).index].reset_index(drop=True)
top_max_timeSHAP_basetokens['RankIdx'] = top_max_timeSHAP_basetokens.groupby(['TUNE_IDX','Threshold'])['max'].rank('dense', ascending=False)

# Combine and filter
min_max_timeSHAP_basetokens = pd.concat([top_max_timeSHAP_basetokens,top_min_timeSHAP_basetokens],ignore_index=True)
filtered_min_max_timeSHAP_values = summarised_timeSHAP_values.merge(min_max_timeSHAP_basetokens[['TUNE_IDX', 'Threshold', 'BaseToken','RankIdx']],how='inner',on=['TUNE_IDX', 'Threshold', 'BaseToken']).reset_index(drop=True)
unique_values_per_base_token = filtered_min_max_timeSHAP_values.groupby('BaseToken',as_index=False).Value.aggregate({'unique_values':lambda x: len(np.unique(x))})
filtered_min_max_timeSHAP_values = filtered_min_max_timeSHAP_values.merge(unique_values_per_base_token,how='left')
filtered_min_max_timeSHAP_values['TokenRankIdx'] = filtered_min_max_timeSHAP_values.groupby(['BaseToken'])['Token'].rank('dense', ascending=True)

filtered_min_max_timeSHAP_values.to_csv(os.path.join(shap_dir,'filtered_plotting_timeSHAP_values.csv'),index=False)




### SAVE: tokens that transitioned over threshold lines
trial_df = compiled_timeSHAP_values[['TUNE_IDX','GUPI','WindowIdx','Token','Threshold','SHAP']].sort_values(by=['TUNE_IDX','GUPI','WindowIdx','Token','Threshold'],ignore_index=True)
trial_df['SHAPDirection'] = 'Neutral'
trial_df.SHAPDirection[trial_df.SHAP>0] = 'Positive'
trial_df.SHAPDirection[trial_df.SHAP<0] = 'Negative'


floofi = trial_df[['TUNE_IDX','GUPI','WindowIdx','Token','SHAPDirection']].groupby(['TUNE_IDX','GUPI','WindowIdx','Token'],as_index=False).value_counts(normalize=True)
troofi = floofi[(floofi.proportion!=1)&(~floofi.Token.str.startswith('<unk>'))&(~floofi.Token.str.endswith('_NAN'))].reset_index(drop=True)
troofi = pd.pivot_table(troofi, values = 'proportion', index=['TUNE_IDX','GUPI','WindowIdx','Token'], columns = 'SHAPDirection').reset_index()
scoofi = troofi[(~troofi.Negative.isna())&(~troofi.Positive.isna())].reset_index(drop=True)