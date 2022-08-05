#### Master Script 2: Partition CENTER-TBI for stratified k-fold cross-validation ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Resampling of dataset for training/testing splits
# III. Resampling of training sets to set assign validation set
# IV. Save partitions
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
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# Load CENTER-TBI dataset to access 6-mo outcomes
CENTER_TBI_demo_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN"," ", ""])

# Load ICU admission and discharge information to extract study GUPIs
CENTER_TBI_ICU_datetime = pd.read_csv('/home/sb2406/rds/hpc-work/timestamps/ICU_adm_disch_timestamps.csv',na_values = ["NA","NaN"," ", ""])

# Filter patients included in the study
CENTER_TBI_demo_info = CENTER_TBI_demo_info[CENTER_TBI_demo_info.GUPI.isin(CENTER_TBI_ICU_datetime.GUPI)].reset_index(drop=True)

### II. Resampling of dataset for validation
# Establish number of folds
FOLDS = 5

# Initialize stratified k-fold cross-validator with fixed random seed
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=2022)

# Initialize empty dataframe to store repeat and fold information
cv_splits = pd.DataFrame(np.empty((0,4)),columns = ['FOLD','SET','GUPI','GOSE'])

# Store vectors of study GUPIs and GOSEs
study_GUPIs = CENTER_TBI_demo_info.GUPI.values
study_GOSEs = CENTER_TBI_demo_info.GOSE6monthEndpointDerived.values

# Iterate through cross-validator and store splits in the dataframe
iter_no = 0
for train_index, test_index in skf.split(study_GUPIs, study_GOSEs):
    iter_no += 1
    
    GUPI_train, GUPI_test = study_GUPIs[train_index], study_GUPIs[test_index]
    GOSE_train, GOSE_test = study_GOSEs[train_index], study_GOSEs[test_index]

    train_df = pd.DataFrame({'FOLD':int(iter_no),'SET':'train','GUPI':GUPI_train,'GOSE':GOSE_train})
    test_df = pd.DataFrame({'FOLD':int(iter_no),'SET':'test','GUPI':GUPI_test,'GOSE':GOSE_test})
    
    cv_splits = cv_splits.append(train_df, ignore_index = True)
    cv_splits = cv_splits.append(test_df, ignore_index = True)

### III. Resampling of training sets to set assign validation set
# Iterate through folds to access individual training sets
for curr_fold in cv_splits.FOLD.unique():
    
    # Extract current training set
    curr_training_set = cv_splits[(cv_splits.SET == 'train')&(cv_splits.FOLD == curr_fold)].reset_index(drop=True)
    
    # Initialize stratified splitter with fixed random seed
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(curr_fold))
    
    # Extract indices from split
    for train_index, val_index in sss.split(curr_training_set.drop(columns='GOSE'),curr_training_set.GOSE):
        train_GUPIs, val_GUPIs = curr_training_set.GUPI[train_index], curr_training_set.GUPI[val_index]
    
    # Assign chosen training set GUPIs to 'val'
    cv_splits.SET[(cv_splits.FOLD == curr_fold)&(cv_splits.SET == 'train')&(cv_splits.GUPI.isin(val_GUPIs))] = 'val'
    
# Sort cross-validation splits and force datatypes
cv_splits = cv_splits.sort_values(by=['FOLD','SET','GUPI'],ignore_index=True)
cv_splits.FOLD = cv_splits.FOLD.astype('int')
    
### IV. Save partitions
cv_splits.to_csv('../cross_validation_splits.csv',index=False)