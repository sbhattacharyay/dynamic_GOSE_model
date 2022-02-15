#### Master Script 3c: Train dictionaries and convert tokens to embedding layer indices ####
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
adm_disch_df = pd.DataFrame({'adm_or_disch':['adm','disch'],'key':1})
indexing_combos = uniq_partitions.merge(adm_disch_df,how='outer').reset_index(drop=True)
# indexing_combos = pd.read_pickle('indexing_combos.pkl')

def main(array_task_id):
    
    curr_repeat = indexing_combos.repeat[array_task_id]
    curr_fold = indexing_combos.fold[array_task_id]
    curr_adm_or_disch = indexing_combos.adm_or_disch[array_task_id]
    
    # Define token dictionary of current split
    fold_dir = os.path.join(token_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold).zfill(1))

    # Extract current training and testing GUPIs
    train_GUPIs = cv_splits.GUPI[(cv_splits.repeat == curr_repeat) & (cv_splits.fold == curr_fold) & (cv_splits.test_or_train == 'train')].values
    test_GUPIs = cv_splits.GUPI[(cv_splits.repeat == curr_repeat) & (cv_splits.fold == curr_fold) & (cv_splits.test_or_train == 'test')].values

    # Load and compile training set tokens
    train_tokens = pd.concat([pd.read_csv(os.path.join(fold_dir,'from_'+curr_adm_or_disch+'_'+g+'.csv')) for g in tqdm(train_GUPIs,desc='Collecting '+curr_adm_or_disch+' training tokens')],ignore_index=True)
    
    # Load and compile testing set tokens
    test_tokens = pd.concat([pd.read_csv(os.path.join(fold_dir,'from_'+curr_adm_or_disch+'_'+g+'.csv')) for g in tqdm(test_GUPIs,desc='Collecting '+curr_adm_or_disch+' testing tokens')],ignore_index=True)
    
    # Create an ordered dictionary to create a token vocabulary from admission
    training_token_list = (' '.join(train_tokens.Token)).split(' ')
    train_token_freqs = OrderedDict(Counter(training_token_list).most_common())

    # Build and save vocabulary (PyTorch Text) from admission
    curr_vocab = vocab(train_token_freqs, min_freq=1)
    unk_token = '<unk>'
    if unk_token not in curr_vocab: curr_vocab.insert_token(unk_token, 0)
    curr_vocab.set_default_index(curr_vocab[unk_token])
    cp.dump(curr_vocab, open(os.path.join(fold_dir,'from_'+curr_adm_or_disch+'_token_dictionary.pkl'), "wb" ))
    
    # Convert training set tokens to indices
    train_tokens['VocabIndex'] = [curr_vocab.lookup_indices(train_tokens.Token[curr_row].split(' ')) for curr_row in tqdm(range(train_tokens.shape[0]),desc='Converting '+curr_adm_or_disch+' training tokens to indices')]
    train_tokens = train_tokens.drop(columns='Token')
    
    # Convert testing set tokens to indices
    test_tokens['VocabIndex'] = [curr_vocab.lookup_indices(test_tokens.Token[curr_row].split(' ')) for curr_row in tqdm(range(test_tokens.shape[0]),desc='Converting '+curr_adm_or_disch+' testing tokens to indices')]
    test_tokens = test_tokens.drop(columns='Token')
    
    # Store training and testing indices into selected directory
    train_tokens.to_pickle(os.path.join(fold_dir,'from_'+curr_adm_or_disch+'_training_indices.pkl'))
    test_tokens.to_pickle(os.path.join(fold_dir,'from_'+curr_adm_or_disch+'_testing_indices.pkl'))
    
    # Delete all used token files once index files are stored to save disk space
    token_file_list = [(os.path.join(fold_dir,'from_'+curr_adm_or_disch+'_'+g+'.csv')) for g in cv_splits.GUPI.unique()]
    [os.remove(f) for f in tqdm(token_file_list,desc='Deleting current fold '+curr_adm_or_disch+' token files')]
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)