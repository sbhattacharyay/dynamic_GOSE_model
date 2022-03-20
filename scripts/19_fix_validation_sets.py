#### Master Script 19: Add partitions for validation sets within the training sets ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation

### I. Initialisation
# Fundamental libraries
import os
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
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

### II. Split training set into fixed validation sets
# Load currently set repeated cross-validation splits
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Extract unique repeat-fold combinations
repeated_cv_partitions = cv_splits[['repeat','fold']].drop_duplicates().reset_index(drop=True)

# Random seed iterator
iterator = 0

# Iterate through unique partitions and split training set into training and validation sets
for curr_row_idx in tqdm(range(repeated_cv_partitions.shape[0])):
    
    # Change iterator
    iterator += 1
    
    # Extract current repeat and fold of partition
    curr_repeat = repeated_cv_partitions.repeat[curr_row_idx]
    curr_fold = repeated_cv_partitions.fold[curr_row_idx]
    
    # Extract current training and testing set dataframes
    curr_training_set = cv_splits[(cv_splits.repeat == curr_repeat)&(cv_splits.fold == curr_fold)&(cv_splits.test_or_train == 'train')].reset_index(drop=True)
    
    # Stratified split of training set into validation set
    full_training_GOSE = curr_training_set[['GUPI','GOSE']].drop_duplicates(ignore_index=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=iterator)
    for train_index, val_index in sss.split(full_training_GOSE.drop(columns='GOSE'),full_training_GOSE.GOSE):
        train_GUPIs, val_GUPIs = full_training_GOSE.GUPI[train_index], full_training_GOSE.GUPI[val_index]
    
    # Assign chosen training set GUPIs to 'val'
    cv_splits.test_or_train[(cv_splits.repeat == curr_repeat)&(cv_splits.fold == curr_fold)&(cv_splits.test_or_train == 'train')&(cv_splits.GUPI.isin(val_GUPIs))] = 'val'
    
# Rename `test_or_train` column to 'set'
cv_splits = cv_splits.rename(columns={'test_or_train':'set'}).sort_values(by=['repeat','fold','set','GUPI'],ignore_index=True)
cv_splits.repeat = cv_splits.repeat.astype('int')
cv_splits.fold = cv_splits.fold.astype('int')

# Overwrite existing partitions
cv_splits.to_csv('../cross_validation_splits.csv',index=False)