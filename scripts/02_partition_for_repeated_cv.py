#### Master Script 2: Partition CENTER-TBI for stratified, repeated k-fold cross-validation ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Resampling of dataset for validation
# III. Save partitions

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
from sklearn.model_selection import RepeatedStratifiedKFold

# Load CENTER-TBI dataset to access 6-mo outcomes
CENTER_TBI_demo_info = pd.read_csv('../CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN"," ", ""])

# Load ICU admission and discharge information to extract study GUPIs
CENTER_TBI_ICU_datetime = pd.read_csv('../ICU_adm_disch_timestamps.csv',na_values = ["NA","NaN"," ", ""])

# Filter patients included in the study
CENTER_TBI_demo_info = CENTER_TBI_demo_info[CENTER_TBI_demo_info.GUPI.isin(CENTER_TBI_ICU_datetime.GUPI)].reset_index(drop=True)

### II. Resampling of dataset for validation
# Establish number of repeats and folds
REPEATS = 20
FOLDS = 5

# Initialize repeated stratified k-fold cross-validator with fixed random seed
rskf = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats = REPEATS, random_state = 2021)

# Initialize empty dataframe to store repeat and fold information
cv_splits = pd.DataFrame(np.empty((0,5)),columns = ['repeat','fold','test_or_train','GUPI','GOSE'])

# Store vectors of study GUPIs and GOSEs
study_GUPIs = CENTER_TBI_demo_info.GUPI.values
study_GOSEs = CENTER_TBI_demo_info.GOSE6monthEndpointDerived.values

# Iterate through cross-validator and store splits in the dataframe
iter_no = 0
for train_index, test_index in rskf.split(study_GUPIs, study_GOSEs):
    
    fold_no = (iter_no % FOLDS) + 1
    repeat_no = np.floor(iter_no/FOLDS) + 1
    
    GUPI_train, GUPI_test = study_GUPIs[train_index], study_GUPIs[test_index]
    GOSE_train, GOSE_test = study_GOSEs[train_index], study_GOSEs[test_index]

    train_df = pd.DataFrame({'repeat':int(repeat_no),'fold':int(fold_no),'test_or_train':'train','GUPI':GUPI_train,'GOSE':GOSE_train})
    test_df = pd.DataFrame({'repeat':int(repeat_no),'fold':int(fold_no),'test_or_train':'test','GUPI':GUPI_test,'GOSE':GOSE_test})
        
    cv_splits = cv_splits.append(train_df, ignore_index = True)
    cv_splits = cv_splits.append(test_df, ignore_index = True)
    
    iter_no += 1

### III. Save partitions
# Convert repeat and fold to integer type and save CV folds
cv_splits.repeat = cv_splits.repeat.astype('int')
cv_splits.fold = cv_splits.fold.astype('int')
cv_splits.to_csv('../cross_validation_splits.csv',index=False)