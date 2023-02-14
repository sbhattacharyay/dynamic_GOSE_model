#### Master Script 06a: Prepare for sensitivity analysis to account for differences in patient stay ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Construct a list of predictions to recalculate after removing dynamic variable tokens
# III. Compile and clean static-only testing set predictions (run after running script 6b)
# IV. Prepare bootstrapping resamples for sensitivity analysis

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
from torchtext.vocab import Vocab
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# SciKit-Learn methods
from sklearn.utils import resample

# Custom methods
from classes.datasets import DYN_ALL_PREDICTOR_SET
from classes.calibration import TemperatureScaling, VectorScaling
from functions.model_building import format_time_tokens, collate_batch, load_static_only_predictions
from models.dynamic_APM import GOSE_model

# Set version code
VERSION = 'v6-0'

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Define directory in which tokens are stored
tokens_dir = '/home/sb2406/rds/hpc-work/tokens'

# Define model performance directory based on version code
model_perf_dir = '/home/sb2406/rds/hpc-work/model_performance/'+VERSION

# Load the current version tuning grid
# post_tuning_grid = pd.read_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'))
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))
post_tuning_grid = tuning_grid[tuning_grid.TUNE_IDX==135].reset_index(drop=True)

# Load legacy cross-validation split information to extract testing resamples
legacy_cv_splits = pd.read_csv('../legacy_cross_validation_splits.csv')
study_GUPIs = legacy_cv_splits[['GUPI','GOSE']].drop_duplicates()

# Load and filter checkpoint file dataframe based on provided model version
ckpt_info = pd.read_pickle(os.path.join('/home/sb2406/rds/hpc-work/model_interpretations/',VERSION,'timeSHAP','ckpt_info.pkl'))
ckpt_info = ckpt_info[ckpt_info.TUNE_IDX==135].reset_index(drop=True)

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

### II. Construct a list of predictions to recalculate after removing dynamic variable tokens
## Find and prepare list of all testing set tokens to convert
# Identify files which match token index pattern in tokens directory
token_idx_files = []
for path in Path(os.path.join(tokens_dir)).rglob('from_adm_strategy_abs_testing_indices.pkl'):
    token_idx_files.append(str(path.resolve()))

# Identify files which match token dictionary pattern in tokens directory
token_dict_files = []
for path in Path(os.path.join(tokens_dir)).rglob('from_adm_strategy_abs_token_dictionary.pkl'):
    token_dict_files.append(str(path.resolve()))

# Characterise the token index file list
token_idx_info_df = pd.DataFrame({'IDX_FILE':token_idx_files,
                                  'REPEAT':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in token_idx_files],
                                  'FOLD':[int(re.search('/fold(.*)/from_', curr_file).group(1)) for curr_file in token_idx_files]
                                  }).sort_values(by=['REPEAT','FOLD']).reset_index(drop=True)

# Characterise the token dictionary file list
token_dict_info_df = pd.DataFrame({'DICT_FILE':token_dict_files,
                                   'REPEAT':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in token_dict_files],
                                   'FOLD':[int(re.search('/fold(.*)/from_', curr_file).group(1)) for curr_file in token_dict_files]
                                   }).sort_values(by=['REPEAT','FOLD']).reset_index(drop=True)

# Merge lists of token indices and token dictionaries
token_info_df = token_idx_info_df.merge(token_dict_info_df,how='left')[['REPEAT','FOLD','DICT_FILE','IDX_FILE']]

# Merge model checkpoint information as well
token_info_df = token_info_df.merge(ckpt_info[['REPEAT','FOLD','TUNE_IDX','file']].rename(columns={'file':'CKPT_FILE'}),how='left')

## Save partition-token-checkpoint information file for sensitivity analysis
token_info_df.to_pickle(os.path.join(model_dir,'sensitivity_analysis_prediction_grid.pkl'))

### III. Compile and clean static-only testing set predictions (run after running script 6b)
## Prepare and save a dataframe of all static-only predictions
# Search for all static-only prediction files
static_only_pred_files = []
for path in Path(model_dir).rglob('calibrated_static_only_test_predictions.pkl'):
    static_only_pred_files.append(str(path.resolve()))

# Characterise the prediction files found
pred_file_info_df = pd.DataFrame({'FILE':static_only_pred_files,
                                  'REPEAT':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in static_only_pred_files],
                                  'FOLD':[int(re.search('/fold(.*)/tune', curr_file).group(1)) for curr_file in static_only_pred_files],
                                  'TUNE_IDX':[int(re.search('/tune(.*)/', curr_file).group(1)) for curr_file in static_only_pred_files],
                                  'VERSION':[re.search('_outputs/(.*)/repeat', curr_file).group(1) for curr_file in static_only_pred_files]
                                 }).sort_values(by=['REPEAT','FOLD','TUNE_IDX']).reset_index(drop=True)

# Partition predictions across available cores
s = [pred_file_info_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
s[:(pred_file_info_df.shape[0] - sum(s))] = [over+1 for over in s[:(pred_file_info_df.shape[0] - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
curr_files_per_core = [(pred_file_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling static-only predictions') for idx in range(len(start_idx))]
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_static_predictions_df = pd.concat(pool.starmap(load_static_only_predictions, curr_files_per_core),ignore_index=True)

# Save compiled static predictions appropriately
compiled_static_predictions_df.to_pickle(os.path.join(model_dir,'compiled_static_only_test_predictions.pkl'))

## Clean static-only prediction files from model output folders
for curr_file in tqdm(pred_file_info_df.FILE,'Deleting individual static-only prediction files'):
    os.remove(curr_file)

### IV. Prepare bootstrapping resamples for sensitivity analysis
## Identifty the number of windows per patient
# Load compiled testing set predictions
compiled_test_preds_df = pd.read_csv(os.path.join(model_dir,'compiled_test_predictions.csv'))

# Calculate the maximum window index per patient
study_window_totals = compiled_test_preds_df.groupby(['GUPI','TrueLabel'],as_index=False)['WindowIdx'].max().rename(columns={'WindowIdx':'WindowTotal'})

## Iterate through window indices of analysis and produce bootstrapping resamples
# Define the number of bootstrapping resamples
NUM_RESAMP = 1000

# Create empty list to store window index-specific resamples
wi_bs_resample_list = []

# Iterate through window indices
for curr_wi in tqdm(range(1,86),'Producing bootstrapping resamples'):
    
    # For each window index of analysis, find the patients that remain at that point
    curr_wi_remaining = study_window_totals[study_window_totals.WindowTotal>=curr_wi].reset_index(drop=True)
    
    # Produce list of boostrapping resamples for current window index
    bs_rs_GUPIs = [resample(curr_wi_remaining.GUPI.values,replace=True,n_samples=curr_wi_remaining.shape[0],stratify=curr_wi_remaining.TrueLabel.values) for _ in range(NUM_RESAMP)]
    bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in bs_rs_GUPIs]

    # Create a mirroring list representing resampling indices
    bs_rs_indices = [np.repeat(idx+1, len(gupi_list)) for idx,gupi_list in enumerate(bs_rs_GUPIs)]

    # Flatten both lists and use to create to a dataframe
    curr_wi_bs_resample = pd.DataFrame({'WINDOW_IDX':curr_wi,'RESAMPLE_IDX':np.concatenate(bs_rs_indices),'GUPIs':np.concatenate(bs_rs_GUPIs)})

    # Append bootstrapping resamples of current window index to running empty list
    wi_bs_resample_list.append(curr_wi_bs_resample)

# Concatenate list to create dataframe of window index-specific resamples
wi_bs_resamples = pd.concat(wi_bs_resample_list,ignore_index=True)

# Save bootstrapping resamples
wi_bs_resamples.to_pickle(os.path.join(model_perf_dir,'sensitivity_bs_resamples.pkl'))

## Create resamples for cut-off analysis
# Create empty list for storing cut-off analysis resamples
cutoff_bs_resample_list = []

# Iterate through window index cut-offs
for curr_wi in tqdm(range(12,86),'Producing bootstrapping resamples'):

    # For each window index of analysis, find the patients that remain at that point and the patients that have dropped out
    curr_wi_remaining = study_window_totals[study_window_totals.WindowTotal>curr_wi].reset_index(drop=True)
    curr_wi_dropout = study_window_totals[study_window_totals.WindowTotal<=curr_wi].reset_index(drop=True)

    # Produce list of boostrapping resamples for current window index
    remaining_bs_rs_GUPIs = [resample(curr_wi_remaining.GUPI.values,replace=True,n_samples=curr_wi_remaining.shape[0],stratify=curr_wi_remaining.TrueLabel.values) for _ in range(NUM_RESAMP)]
    remaining_bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in remaining_bs_rs_GUPIs]

    # Produce list of boostrapping resamples for current window index
    dropout_bs_rs_GUPIs = [resample(curr_wi_dropout.GUPI.values,replace=True,n_samples=curr_wi_dropout.shape[0],stratify=curr_wi_dropout.TrueLabel.values) for _ in range(NUM_RESAMP)]
    dropout_bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in dropout_bs_rs_GUPIs]

    # Create a mirroring list representing resampling indices
    remaining_bs_rs_indices = [np.repeat(idx+1, len(gupi_list)) for idx,gupi_list in enumerate(remaining_bs_rs_GUPIs)]
    dropout_bs_rs_indices = [np.repeat(idx+1, len(gupi_list)) for idx,gupi_list in enumerate(dropout_bs_rs_GUPIs)]

    # Flatten both lists and use to create to a dataframe
    remaining_curr_wi_bs_resample = pd.DataFrame({'WINDOW_IDX':curr_wi,'RESAMPLE_IDX':np.concatenate(remaining_bs_rs_indices),'GUPIs':np.concatenate(remaining_bs_rs_GUPIs)})
    dropout_curr_wi_bs_resample = pd.DataFrame({'WINDOW_IDX':curr_wi,'RESAMPLE_IDX':np.concatenate(dropout_bs_rs_indices),'GUPIs':np.concatenate(dropout_bs_rs_GUPIs)})

    # Create new column indicating remaining or dropped out sample
    remaining_curr_wi_bs_resample['SAMPLE'] = 'Remaining'
    dropout_curr_wi_bs_resample['SAMPLE'] = 'Dropout'

    # Append bootstrapping resamples of current window index to running empty list
    cutoff_bs_resample_list.append(remaining_curr_wi_bs_resample)
    cutoff_bs_resample_list.append(dropout_curr_wi_bs_resample)

# Concatenate list to create dataframe of window index-specific resamples
cutoff_bs_resamples = pd.concat(cutoff_bs_resample_list,ignore_index=True)

# Save bootstrapping resamples
cutoff_bs_resamples.to_pickle(os.path.join(model_perf_dir,'cutoff_analysis_bs_resamples.pkl'))