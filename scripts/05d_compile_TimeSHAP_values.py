#### Master Script 05d: Compile TimeSHAP values calculated in parallel ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile TimeSHAP values and clean directory

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

# Define a subdirectory for the storage of missed TimeSHAP transitions
missed_transition_dir = os.path.join(shap_dir,'missed_transitions')

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

