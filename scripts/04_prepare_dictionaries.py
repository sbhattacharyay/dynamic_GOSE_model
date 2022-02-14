#### Master Script 4: Train dictionaries and convert tokens to embedding layer indices ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Train dictionaries per repeated cross-validation partition and convert tokens to indices

### I. Initialisation
# Fundamental libraries
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

# Scikit-Learn methods
from sklearn.preprocessing import KBinsDiscretizer

# Custom token preparation functions
from functions.token_preparation import no_digit_strings_to_na, categorizer, tokenize_categoricals, get_ts_event_tokens, get_date_event_tokens, load_tokens

# Set directory from which to load the tokens
token_dir = '/home/sb2406/rds/hpc-work/tokens'

# Load cross-validation splits
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Set the number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()

### II. Train dictionaries per repeated cross-validation partition and convert tokens to indices
# Iterate through cross-validation splits
for curr_repeat in cv_splits.repeat.unique():
    for curr_fold in cv_splits.fold.unique():
        
        # Define token dictionary of current split
        fold_dir = os.path.join(token_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold).zfill(1))
        
        # Extract current training and testing GUPIs
        train_GUPIs = cv_splits.GUPI[(cv_splits.repeat == curr_repeat) & (cv_splits.fold == curr_fold) & (cv_splits.test_or_train == 'train')].values
        test_GUPIs = cv_splits.GUPI[(cv_splits.repeat == curr_repeat) & (cv_splits.fold == curr_fold) & (cv_splits.test_or_train == 'test')].values
        
        # Create partition resamples among cores
        train_s = [train_GUPIs.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
        train_s[:(train_GUPIs.shape[0] - sum(train_s))] = [val+1 for val in train_s[:(train_GUPIs.shape[0] - sum(train_s))]]    
        train_end_indices = np.cumsum(train_s)
        train_start_indices = np.insert(train_end_indices[:-1],0,0)
        train_files_per_core = [(train_GUPIs[train_start_indices[idx]:train_end_indices[idx]],fold_dir,True,'Collecting training tokens') for idx in range(len(train_start_indices))]
        
        test_s = [test_GUPIs.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
        test_s[:(test_GUPIs.shape[0] - sum(test_s))] = [val+1 for val in test_s[:(test_GUPIs.shape[0] - sum(test_s))]]    
        test_end_indices = np.cumsum(test_s)
        test_start_indices = np.insert(test_end_indices[:-1],0,0)
        test_files_per_core = [(test_GUPIs[test_start_indices[idx]:test_end_indices[idx]],fold_dir,True,'Collecting testing tokens') for idx in range(len(test_start_indices))]
        
        # Compile training set tokens
        with multiprocessing.Pool(NUM_CORES) as pool:
            train_tokens = pd.concat(pool.starmap(load_tokens, train_files_per_core),ignore_index=True)
        
        # Compile testing set tokens
        with multiprocessing.Pool(NUM_CORES) as pool:
            test_tokens = pd.concat(pool.starmap(load_tokens, test_files_per_core),ignore_index=True)        
        
        # Create an ordered dictionary to create a token vocabulary
        train_token_freqs = train_tokens.Tokens.value_counts().to_dict(into=OrderedDict)
        
        # Build vocabulary (PyTorch Text)
        curr_vocab = vocab(train_token_freqs, min_freq=1)
        unk_token = '<unk>'
        if unk_token not in curr_vocab: curr_vocab.insert_token(unk_token, 0)
        curr_vocab.set_default_index(curr_vocab[unk_token])
        
        # Convert training and testing tokens to indices
        train_tokens['Index'] = [curr_vocab[t] for t in train_tokens.Tokens]
        test_tokens['Index'] = [curr_vocab[t] for t in test_tokens.Tokens]
        
        # Group indices by GUPI
        train_indices = train_tokens.groupby('GUPI',as_index=False)['Index'].aggregate(lambda col: col.tolist()).reset_index(drop=True)
        test_indices = test_tokens.groupby('GUPI',as_index=False)['Index'].aggregate(lambda col: col.tolist()).reset_index(drop=True)
        
        # Store training and testing indices into selected directory
        train_indices.to_pickle(os.path.join(fold_dir,'training_indices.pkl'))
        test_indices.to_pickle(os.path.join(fold_dir,'testing_indices.pkl'))
        
        # Store torch Vocab object
        cp.dump(curr_vocab, open(os.path.join(fold_dir,'token_dictionary.pkl'), "wb" ))
