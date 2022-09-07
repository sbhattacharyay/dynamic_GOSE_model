#### Master Script 04f: Compile testing set performance results to calculate confidence intervals of dynamic all-predictor-based models ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile and save bootstrapped testing set performance dataframes
# III. Dropout configurations based on testing set calibration and discrimination information
# IV. Delete folders of underperforming configurations
# V. Create bootstrapping resamples for calculating testing set performance

### I. Initialisation
# Fundamental libraries
import os
import re
import sys
import time
import glob
import random
import shutil
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
from shutil import rmtree
from ast import literal_eval
import matplotlib.pyplot as plt
from scipy.special import logit
from argparse import ArgumentParser
from collections import Counter, OrderedDict
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

# StatsModel methods
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
from statsmodels.nonparametric.smoothers_lowess import lowess

# Custom methods
from functions.model_building import load_val_performance

# Set version code
VERSION = 'v7-0'

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Define model performance directory based on version code
model_perf_dir = '/home/sb2406/rds/hpc-work/model_performance/'+VERSION

# Define subdirectory to store testing set bootstrapping results
test_bs_dir = os.path.join(model_perf_dir,'testing_set_bootstrapping')

# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
partitions = cv_splits[['FOLD']].drop_duplicates().reset_index(drop=True)
study_GUPI_GOSE = cv_splits[['GUPI','GOSE']].drop_duplicates()

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Set number of resamples for bootstrapping-based testing set performance
NUM_RESAMP = 1000

# Load the post-dropout tuning grid
filt_tuning_grid = pd.read_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'))

### II. Compile and save bootstrapped testing set performance dataframes
# Search for all performance files
perf_files = []
for path in Path(test_bs_dir).rglob('test_*'):
    perf_files.append(str(path.resolve()))

# Characterise the performance files found
perf_file_info_df = pd.DataFrame({'FILE':perf_files,
                                  'VERSION':[re.search('_performance/(.*)/testing_set_', curr_file).group(1) for curr_file in perf_files],
                                  'METRIC':[re.search('/test_(.*)_rs_', curr_file).group(1) for curr_file in perf_files],
                                  'RESAMPLE_IDX':[int(re.search('_rs_(.*).pkl', curr_file).group(1)) for curr_file in perf_files],
                                 }).sort_values(by=['METRIC','RESAMPLE_IDX']).reset_index(drop=True)

# Separate discrimination and calibration file dataframes
discrimination_file_info_df = perf_file_info_df[perf_file_info_df.METRIC == 'discrimination'].reset_index(drop=True)
calibration_file_info_df = perf_file_info_df[perf_file_info_df.METRIC == 'calibration'].reset_index(drop=True)

# Partition performance files across available cores
s = [discrimination_file_info_df.RESAMPLE_IDX.max() // NUM_CORES for _ in range(NUM_CORES)]
s[:(discrimination_file_info_df.RESAMPLE_IDX.max() - sum(s))] = [over+1 for over in s[:(discrimination_file_info_df.RESAMPLE_IDX.max() - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
discrimination_files_per_core = [(discrimination_file_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling testing set discrimination performance') for idx in range(len(start_idx))]
calibration_files_per_core = [(calibration_file_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling testing set calibration performance') for idx in range(len(start_idx))]

# Load testing set discrimination and calibration performance dataframes in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_test_discrimination = pd.concat(pool.starmap(load_test_performance, discrimination_files_per_core),ignore_index=True)
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_test_calibration = pd.concat(pool.starmap(load_test_performance, calibration_files_per_core),ignore_index=True)

### III.Calculate 95% confidence intervals on test set performance metrics
## Discrimination metrics
test_CI_discrimination = compiled_test_discrimination.groupby(['TUNE_IDX','METRIC','WINDOW_IDX'],as_index=False)['VALUE'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'resamples':'count'}).reset_index(drop=True)

## Calibration metrics
test_CI_calibration = compiled_test_calibration.groupby(['TUNE_IDX','METRIC','WINDOW_IDX','THRESHOLD'],as_index=False)['VALUE'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'resamples':'count'}).reset_index(drop=True)

## Save confidence intervals of both calibration and discrimination metrics
test_CI_discrimination.to_csv(os.path.join(model_perf_dir,'test_set_discrimination_CI.csv'),index=False)
test_CI_calibration.to_csv(os.path.join(model_perf_dir,'test_set_calibration_CI.csv'),index=False)