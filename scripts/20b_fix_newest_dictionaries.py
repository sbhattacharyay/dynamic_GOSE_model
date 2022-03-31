#### Master Script 20b: Train dictionaries and convert tokens to embedding layer indices ####
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
from functions.token_preparation import load_tokens, convert_tokens, del_files

# Set directory from which to load the tokens
token_dir = '/home/sb2406/rds/hpc-work/tokens'

# Load cross-validation splits
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Set the number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Create combinations of repeats, folds, and admission/discharge
uniq_partitions = cv_splits[['repeat','fold']].drop_duplicates(ignore_index=True)
uniq_partitions['key'] = 1
strategy_df = pd.DataFrame({'strategy':['abs','diff'],'key':1})
indexing_combos = uniq_partitions.merge(strategy_df,how='outer').drop(columns='key').reset_index(drop=True)

def main(array_task_id):
    
    # Extract parameters of current indexing combination 
    curr_repeat = indexing_combos.repeat[array_task_id]
    curr_fold = indexing_combos.fold[array_task_id]
    curr_strategy = indexing_combos.strategy[array_task_id]
    
    # Define token dictionary of current split
    fold_dir = os.path.join(token_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold).zfill(1))

    # Extract current training and testing GUPIs
    train_GUPIs = cv_splits.GUPI[(cv_splits.repeat == curr_repeat) & (cv_splits.fold == curr_fold) & (cv_splits.set == 'train')].values
    val_GUPIs = cv_splits.GUPI[(cv_splits.repeat == curr_repeat) & (cv_splits.fold == curr_fold) & (cv_splits.set == 'val')].values
    test_GUPIs = cv_splits.GUPI[(cv_splits.repeat == curr_repeat) & (cv_splits.fold == curr_fold) & (cv_splits.set == 'test')].values

    # Load and compile training set tokens
    train_tokens = pd.concat([pd.read_csv(os.path.join(fold_dir,'from_adm_GUPI_'+g+'_strategy_'+curr_strategy+'.csv')) for g in tqdm(train_GUPIs,desc='Collecting training tokens')],ignore_index=True).fillna('')
    train_time_tokens = pd.concat([pd.read_csv(os.path.join(fold_dir,'time_tokens_GUPI_'+g+'_strategy_'+curr_strategy+'.csv')) for g in tqdm(train_GUPIs,desc='Collecting training time tokens')],ignore_index=True).fillna('')

    # Load and compile validation set tokens
    val_tokens = pd.concat([pd.read_csv(os.path.join(fold_dir,'from_adm_GUPI_'+g+'_strategy_'+curr_strategy+'.csv')) for g in tqdm(val_GUPIs,desc='Collecting validation tokens')],ignore_index=True).fillna('')
    val_time_tokens = pd.concat([pd.read_csv(os.path.join(fold_dir,'time_tokens_GUPI_'+g+'_strategy_'+curr_strategy+'.csv')) for g in tqdm(val_GUPIs,desc='Collecting validation time tokens')],ignore_index=True).fillna('')

    # Load and compile testing set tokens
    test_tokens = pd.concat([pd.read_csv(os.path.join(fold_dir,'from_adm_GUPI_'+g+'_strategy_'+curr_strategy+'.csv')) for g in tqdm(test_GUPIs,desc='Collecting testing tokens')],ignore_index=True).fillna('')
    test_time_tokens = pd.concat([pd.read_csv(os.path.join(fold_dir,'time_tokens_GUPI_'+g+'_strategy_'+curr_strategy+'.csv')) for g in tqdm(test_GUPIs,desc='Collecting testing time tokens')],ignore_index=True).fillna('')

    # Create an ordered dictionary to create a token vocabulary from admission
    training_token_list = (' '.join(train_tokens.Token)).split(' ')
    time_from_adm_list = (' '.join(test_time_tokens.TimeFromAdm)).split(' ')
    time_of_day_list = (' '.join(test_time_tokens.TimeOfDay)).split(' ')
    training_token_list = training_token_list + time_from_adm_list + time_of_day_list
    if ('' in training_token_list):
        training_token_list = list(filter(lambda a: a != '', training_token_list))
    train_token_freqs = OrderedDict(Counter(training_token_list).most_common())

    # Build and save vocabulary (PyTorch Text) from admission
    curr_vocab = vocab(train_token_freqs, min_freq=1)
    null_token = ''
    unk_token = '<unk>'
    if null_token not in curr_vocab: curr_vocab.insert_token(null_token, 0)
    if unk_token not in curr_vocab: curr_vocab.insert_token(unk_token, len(curr_vocab))
    curr_vocab.set_default_index(curr_vocab[unk_token])
    cp.dump(curr_vocab, open(os.path.join(fold_dir,'from_adm_strategy_'+curr_strategy+'_token_dictionary.pkl'), "wb" ))
    
    # Convert training set tokens to indices
    train_tokens['VocabIndex'] = [curr_vocab.lookup_indices(train_tokens.Token[curr_row].split(' ')) for curr_row in tqdm(range(train_tokens.shape[0]),desc='Converting training tokens to indices for strategy: '+curr_strategy)]
    train_tokens = train_tokens.drop(columns='Token')
    train_time_tokens['VocabTimeFromAdmIndex'] = [curr_vocab.lookup_indices(train_time_tokens.TimeFromAdm[curr_row].split(' ')) for curr_row in tqdm(range(train_time_tokens.shape[0]),desc='Converting training time from admission tokens to indices for strategy: '+curr_strategy)]
    train_time_tokens['VocabTimeOfDayIndex'] = [curr_vocab.lookup_indices(train_time_tokens.TimeOfDay[curr_row].split(' ')) for curr_row in tqdm(range(train_time_tokens.shape[0]),desc='Converting training time of day tokens to indices for strategy: '+curr_strategy)]
    train_time_tokens = train_time_tokens.drop(columns=['TimeFromAdm','TimeOfDay'])
    train_tokens = train_tokens.merge(train_time_tokens,on=['GUPI','TimeStampStart','TimeStampEnd','WindowIdx'],how='left').reset_index(drop=True)
    
    # Convert validation set tokens to indices
    val_tokens['VocabIndex'] = [curr_vocab.lookup_indices(val_tokens.Token[curr_row].split(' ')) for curr_row in tqdm(range(val_tokens.shape[0]),desc='Converting validation tokens to indices for strategy: '+curr_strategy)]
    val_tokens = val_tokens.drop(columns='Token')
    val_time_tokens['VocabTimeFromAdmIndex'] = [curr_vocab.lookup_indices(val_time_tokens.TimeFromAdm[curr_row].split(' ')) for curr_row in tqdm(range(val_time_tokens.shape[0]),desc='Converting validation time from admission tokens to indices for strategy: '+curr_strategy)]
    val_time_tokens['VocabTimeOfDayIndex'] = [curr_vocab.lookup_indices(val_time_tokens.TimeOfDay[curr_row].split(' ')) for curr_row in tqdm(range(val_time_tokens.shape[0]),desc='Converting validation time of day tokens to indices for strategy: '+curr_strategy)]
    val_time_tokens = val_time_tokens.drop(columns=['TimeFromAdm','TimeOfDay'])
    val_tokens = val_tokens.merge(val_time_tokens,on=['GUPI','TimeStampStart','TimeStampEnd','WindowIdx'],how='left').reset_index(drop=True)

    # Convert testing set tokens to indices
    test_tokens['VocabIndex'] = [curr_vocab.lookup_indices(test_tokens.Token[curr_row].split(' ')) for curr_row in tqdm(range(test_tokens.shape[0]),desc='Converting testing tokens to indices for strategy: '+curr_strategy)]
    test_tokens = test_tokens.drop(columns='Token')
    test_time_tokens['VocabTimeFromAdmIndex'] = [curr_vocab.lookup_indices(test_time_tokens.TimeFromAdm[curr_row].split(' ')) for curr_row in tqdm(range(test_time_tokens.shape[0]),desc='Converting testing time from admission tokens to indices for strategy: '+curr_strategy)]
    test_time_tokens['VocabTimeOfDayIndex'] = [curr_vocab.lookup_indices(test_time_tokens.TimeOfDay[curr_row].split(' ')) for curr_row in tqdm(range(test_time_tokens.shape[0]),desc='Converting testing time of day tokens to indices for strategy: '+curr_strategy)]
    test_time_tokens = test_time_tokens.drop(columns=['TimeFromAdm','TimeOfDay'])
    test_tokens = test_tokens.merge(test_time_tokens,on=['GUPI','TimeStampStart','TimeStampEnd','WindowIdx'],how='left').reset_index(drop=True)
    
    # Delete all used token files once index files are stored to save disk space
    token_file_list = [(os.path.join(fold_dir,'from_adm_GUPI_'+g+'_strategy_'+curr_strategy+'.csv')) for g in cv_splits.GUPI.unique()]
    time_token_file_list = [(os.path.join(fold_dir,'time_tokens_GUPI_'+g+'_strategy_'+curr_strategy+'.csv')) for g in cv_splits.GUPI.unique()]
    token_file_list = token_file_list + time_token_file_list
    token_file_list = [f for f in token_file_list if os.path.exists(f)]
    [os.remove(f) for f in tqdm(token_file_list,desc='Deleting current partition token files for strategy: '+curr_strategy)]
    
    # Store training and testing indices into selected directory
    train_tokens.to_pickle(os.path.join(fold_dir,'from_adm_strategy_'+curr_strategy+'_training_indices.pkl'))
    val_tokens.to_pickle(os.path.join(fold_dir,'from_adm_strategy_'+curr_strategy+'_validation_indices.pkl'))
    test_tokens.to_pickle(os.path.join(fold_dir,'from_adm_strategy_'+curr_strategy+'_testing_indices.pkl'))
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)