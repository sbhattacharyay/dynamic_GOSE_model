#### Master Script 06a: Prepare for sensitivity analysis to account for differences in patient stay ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Construct a list of predictions to recalculate after removing dynamic variable tokens

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

# Custom methods
from classes.datasets import DYN_ALL_PREDICTOR_SET
from classes.calibration import TemperatureScaling, VectorScaling
from functions.model_building import format_time_tokens, collate_batch
from models.dynamic_APM import GOSE_model

# Set version code
VERSION = 'v6-0'

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Define directory in which tokens are stored
tokens_dir = '/home/sb2406/rds/hpc-work/tokens'

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

# ## Partition list of all testing set tokens among available cores
# # Identify number of token sets per core
# s = [token_info_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
# s[:(token_info_df.shape[0] - sum(s))] = [over+1 for over in s[:(token_info_df.shape[0] - sum(s))]]

# # Convert counts per core into indices of dataframe
# end_idx = np.cumsum(s)
# start_idx = np.insert(end_idx[:-1],0,0)

# # Split dataframe into array components based on core partitions
# token_files_per_core = [(token_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),post_tuning_grid,study_GUPIs,True,'Calculating testing set predictions after removing all dynamic variable tokens for sensitivity analysis') for idx in range(len(start_idx))]

# ## Calculate testing set predictions with dynamic tokens removed
# # Calculate predictions in parallel
# with multiprocessing.Pool(NUM_CORES) as pool:
#     compiled_static_only_predictions = pd.concat(pool.starmap(static_only_predictions, token_files_per_core),ignore_index=True)

# # Save compiled static-only predictions
# compiled_static_only_predictions.to_csv(os.path.join(model_dir,'compiled_static_only_predictions.csv'),index=False)