#### Master Script 3d: Characterise token set to determine ideal end times for model training ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Create and classify token dictionary for chosen partition
# III. Characterise tokens of chosen partition

### I. Initialisation
# Import necessary libraries
import os
import re
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
from collections import Counter, OrderedDict

# PyTorch and PyTorch.Text methods
from torchtext.vocab import vocab, Vocab

# Custom token preparation functions
from functions.token_preparation import get_token_info

# Set directory from which to load the tokens
token_dir = '/home/sb2406/rds/hpc-work/tokens'

# Load cross-validation splits
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Set the number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Set and repeat and fold to focus on for analysis
REPEAT = 1
FOLD = 1
fold_dir = os.path.join(token_dir,'repeat'+str(REPEAT).zfill(2),'fold'+str(FOLD).zfill(1))

### II. Create and classify token dictionary for chosen partition
# Load token dictionaries of chosen partition
curr_adm_vocab = cp.load(open(os.path.join(fold_dir,'from_adm_token_dictionary.pkl'),"rb"))
curr_disch_vocab = cp.load(open(os.path.join(fold_dir,'from_disch_token_dictionary.pkl'),"rb"))

# Create dataframe version of dictionaries
curr_adm_vocab_df = pd.DataFrame({'VocabIndex':list(range(len(curr_adm_vocab))),'Token':curr_adm_vocab.get_itos()})
curr_disch_vocab_df = pd.DataFrame({'VocabIndex':list(range(len(curr_disch_vocab))),'Token':curr_disch_vocab.get_itos()})

# Determine whether tokens are baseline
curr_adm_vocab_df['Baseline'] = curr_adm_vocab_df['Token'].str.startswith('Baseline')
curr_disch_vocab_df['Baseline'] = curr_disch_vocab_df['Token'].str.startswith('Baseline')

# Determine whether tokens are numeric
curr_adm_vocab_df['Numeric'] = curr_adm_vocab_df['Token'].str.contains('_BIN')
curr_disch_vocab_df['Numeric'] = curr_disch_vocab_df['Token'].str.contains('_BIN')

# Determine wheter tokens represent missing values
curr_adm_vocab_df['Missing'] = ((curr_adm_vocab_df.Numeric)&(curr_adm_vocab_df['Token'].str.endswith('_BIN_missing')))|((~curr_adm_vocab_df.Numeric)&(curr_adm_vocab_df['Token'].str.endswith('_NA')))
curr_disch_vocab_df['Missing'] = ((curr_disch_vocab_df.Numeric)&(curr_disch_vocab_df['Token'].str.endswith('_BIN_missing')))|((~curr_disch_vocab_df.Numeric)&(curr_disch_vocab_df['Token'].str.endswith('_NA')))

# Create empty column for predictor base token
curr_adm_vocab_df['BaseToken'] = ''
curr_disch_vocab_df['BaseToken'] = ''

# For numeric tokens, extract the portion of the string before '_BIN' as the BaseToken
curr_adm_vocab_df.BaseToken[curr_adm_vocab_df.Numeric] = curr_adm_vocab_df.Token[curr_adm_vocab_df.Numeric].str.replace('\\_BIN.*','',1,regex=True)
curr_disch_vocab_df.BaseToken[curr_disch_vocab_df.Numeric] = curr_disch_vocab_df.Token[curr_disch_vocab_df.Numeric].str.replace('\\_BIN.*','',1,regex=True)

# For non-numeric tokens, extract everything before the final underscore, if one exists, as the BaseToken
curr_adm_vocab_df.BaseToken[~curr_adm_vocab_df.Numeric] = curr_adm_vocab_df.Token[~curr_adm_vocab_df.Numeric].str.replace('_[^_]*$','',1,regex=True)
curr_disch_vocab_df.BaseToken[~curr_disch_vocab_df.Numeric] = curr_disch_vocab_df.Token[~curr_disch_vocab_df.Numeric].str.replace('_[^_]*$','',1,regex=True)

# For baseline tokens, remove the "Baseline" prefix in the BaseToken
curr_adm_vocab_df.BaseToken[curr_adm_vocab_df.Baseline] = curr_adm_vocab_df.BaseToken[curr_adm_vocab_df.Baseline].str.replace('Baseline','',1,regex=False)
curr_disch_vocab_df.BaseToken[curr_disch_vocab_df.Baseline] = curr_disch_vocab_df.BaseToken[curr_disch_vocab_df.Baseline].str.replace('Baseline','',1,regex=False)

# Load complete token dictionary from prior study
old_token_dictionary = pd.read_csv('/home/sb2406/rds/hpc-work/tokens/old_token_dictionary.csv').drop(columns=['Token','count','UnderscoreCount','Baseline','Numeric']).drop_duplicates(ignore_index=True)

# Merge old token dictionary with vocab dataframes
curr_adm_vocab_df = curr_adm_vocab_df.merge(old_token_dictionary,how='left',on=['BaseToken'])
curr_disch_vocab_df = curr_disch_vocab_df.merge(old_token_dictionary,how='left',on=['BaseToken'])

# Add 'TimeOfDay' token information to vocab dataframes
curr_adm_vocab_df.ICUIntervention[curr_adm_vocab_df.BaseToken=='TimeOfDay'] = False
curr_adm_vocab_df.ClinicianInput[curr_adm_vocab_df.BaseToken=='TimeOfDay'] = False
curr_adm_vocab_df.Type[curr_adm_vocab_df.BaseToken=='TimeOfDay'] = 'Time of day'

curr_disch_vocab_df.ICUIntervention[curr_disch_vocab_df.BaseToken=='TimeOfDay'] = False
curr_disch_vocab_df.ClinicianInput[curr_disch_vocab_df.BaseToken=='TimeOfDay'] = False
curr_disch_vocab_df.Type[curr_disch_vocab_df.BaseToken=='TimeOfDay'] = 'Time of day'

### III. Characterise tokens of chosen partition
# Load token indices of chosen partition
adm_training_indices = pd.read_pickle(os.path.join(fold_dir,'from_adm_training_indices.pkl'))
disch_training_indices = pd.read_pickle(os.path.join(fold_dir,'from_disch_training_indices.pkl'))
adm_testing_indices = pd.read_pickle(os.path.join(fold_dir,'from_adm_testing_indices.pkl'))
disch_testing_indices = pd.read_pickle(os.path.join(fold_dir,'from_disch_testing_indices.pkl'))

# Partition training indices among cores and calculate token info in parallel
train_s = [adm_training_indices.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
train_s[:(adm_training_indices.shape[0] - sum(train_s))] = [over+1 for over in train_s[:(adm_training_indices.shape[0] - sum(train_s))]]
end_idx = np.cumsum(train_s)
start_idx = np.insert(end_idx[:-1],0,0)

adm_train_idx_splits = [(adm_training_indices.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),curr_adm_vocab_df,False,True,'Characterising training tokens from admission') for idx in range(len(start_idx))]
with multiprocessing.Pool(NUM_CORES) as pool:
    adm_train_token_info = pd.concat(pool.starmap(get_token_info, adm_train_idx_splits),ignore_index=True)
    
disch_train_idx_splits = [(disch_training_indices.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),curr_disch_vocab_df,False,True,'Characterising training tokens from discharge') for idx in range(len(start_idx))]
with multiprocessing.Pool(NUM_CORES) as pool:
    disch_train_token_info = pd.concat(pool.starmap(get_token_info, disch_train_idx_splits),ignore_index=True)

# Partition testing indices among cores and calculate token info in parallel
test_s = [adm_testing_indices.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
test_s[:(adm_testing_indices.shape[0] - sum(test_s))] = [over+1 for over in test_s[:(adm_testing_indices.shape[0] - sum(test_s))]]
end_idx = np.cumsum(test_s)
start_idx = np.insert(end_idx[:-1],0,0)

adm_test_idx_splits = [(adm_testing_indices.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),curr_adm_vocab_df,False,True,'Characterising testing tokens from admission') for idx in range(len(start_idx))]
with multiprocessing.Pool(NUM_CORES) as pool:
    adm_test_token_info = pd.concat(pool.starmap(get_token_info, adm_test_idx_splits),ignore_index=True)
    
disch_test_idx_splits = [(disch_testing_indices.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),curr_disch_vocab_df,False,True,'Characterising testing tokens from discharge') for idx in range(len(start_idx))]
with multiprocessing.Pool(NUM_CORES) as pool:
    disch_test_token_info = pd.concat(pool.starmap(get_token_info, disch_test_idx_splits),ignore_index=True)

# Compile training and testing set into one:
adm_token_info = pd.concat([adm_train_token_info,adm_test_token_info],ignore_index=True)
disch_token_info = pd.concat([disch_train_token_info,disch_test_token_info],ignore_index=True)

# Save token characteristic dataframes into appropriate directory
adm_token_info.to_csv(os.path.join(fold_dir,'from_adm_token_characteristics.csv'),index=False)
disch_token_info.to_csv(os.path.join(fold_dir,'from_disch_token_characteristics.csv'),index=False)
