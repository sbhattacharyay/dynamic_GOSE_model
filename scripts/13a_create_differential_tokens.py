#### Master Script 13a: Create differential tokens for new training approach ####
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

index_files = []
for path in Path(token_dir).rglob('*_indices.pkl'):
    index_files.append(str(path.resolve()))
    
# Characterise tokenized index files
index_file_info = pd.DataFrame({'file':index_files,
                               'repeat':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in index_files],
                               'fold':[int(re.search('/fold(.*)/from_', curr_file).group(1)) for curr_file in index_files],
                               'adm_or_disch':[re.search('/from_(.*)_t', curr_file).group(1) for curr_file in index_files],
                               'train_or_test':[re.search('_(.*)ing_indices.pkl', curr_file).group(1) for curr_file in index_files]
                              }).sort_values(by=['repeat','fold','adm_or_disch','train_or_test']).reset_index(drop=True)
index_file_info['train_or_test'] = index_file_info['train_or_test'].str.rsplit(pat='_', n=1).apply(lambda x: x[1])

curr_file = index_file_info.file[0]
curr_repeat = index_file_info.repeat[0]
curr_fold = index_file_info.fold[0]
curr_adm_or_disch = index_file_info.adm_or_disch[0]
curr_train_or_test = index_file_info.train_or_test[0]

fold_dir = os.path.join(token_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold).zfill(1))

curr_vocab = cp.load(open(os.path.join(fold_dir,'from_'+curr_adm_or_disch+'_token_dictionary.pkl'),"rb"))

curr_indices = pd.read_pickle(curr_file)

curr_GUPI = curr_indices.GUPI.unique()[0]

curr_GUPI_indices = curr_indices[curr_indices.GUPI == curr_GUPI].reset_index(drop=True)

