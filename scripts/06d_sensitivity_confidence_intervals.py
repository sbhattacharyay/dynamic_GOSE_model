#### Master Script 06d: Compile performance results from sensitivity analysis to calculate confidence intervals of dynamic all-predictor-based models ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile and save bootstrapped sensitivity testing set difference performance dataframes
# III. Calculate 95% confidence intervals on sensitivity difference performance metrics
# IV. Calculate 95% confidence intervals for sensitivity cutoff analysis

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
from functions.model_building import load_test_performance

# Set version code
VERSION = 'v6-0'

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Define model performance directory based on version code
model_perf_dir = '/home/sb2406/rds/hpc-work/model_performance/'+VERSION

# Define subdirectory to store testing set bootstrapping results
test_bs_dir = os.path.join(model_perf_dir,'sensitivity_bootstrapping')

# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../legacy_cross_validation_splits.csv')
partitions = cv_splits[['REPEAT','FOLD']].drop_duplicates().reset_index(drop=True)
study_GUPI_GOSE = cv_splits[['GUPI','GOSE']].drop_duplicates()

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Set number of resamples for bootstrapping-based testing set performance
NUM_RESAMP = 1000

# Load the post-dropout tuning grid
# filt_tuning_grid = pd.read_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'))
filt_tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))
filt_tuning_grid = filt_tuning_grid[filt_tuning_grid.TUNE_IDX == 135].reset_index(drop=True)

### II. Compile and save bootstrapped sensitivity testing set difference performance dataframes
# Search for all performance files
diff_perf_files = []
for path in Path(test_bs_dir).rglob('diff_*'):
    diff_perf_files.append(str(path.resolve()))

# Characterise the performance files found
perf_file_info_df = pd.DataFrame({'FILE':diff_perf_files,
                                  'VERSION':[re.search('_performance/(.*)/sensitivity_bootstrapping', curr_file).group(1) for curr_file in diff_perf_files],
                                  'METRIC':[re.search('/diff_(.*)_rs_', curr_file).group(1) for curr_file in diff_perf_files],
                                  'RESAMPLE_IDX':[int(re.search('_rs_(.*).pkl', curr_file).group(1)) for curr_file in diff_perf_files],
                                 }).sort_values(by=['METRIC','RESAMPLE_IDX']).reset_index(drop=True)

# Separate discrimination and calibration file dataframes
discrimination_file_info_df = perf_file_info_df[perf_file_info_df.METRIC == 'discrimination'].reset_index(drop=True)
calibration_file_info_df = perf_file_info_df[perf_file_info_df.METRIC == 'calibration'].reset_index(drop=True)

# Partition performance files across available cores
s = [discrimination_file_info_df.RESAMPLE_IDX.max() // NUM_CORES for _ in range(NUM_CORES)]
s[:(discrimination_file_info_df.RESAMPLE_IDX.max() - sum(s))] = [over+1 for over in s[:(discrimination_file_info_df.RESAMPLE_IDX.max() - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
discrimination_files_per_core = [(discrimination_file_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling sensitivity discrimination performance') for idx in range(len(start_idx))]
calibration_files_per_core = [(calibration_file_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling sensitivity calibration performance') for idx in range(len(start_idx))]

# Load testing set discrimination and calibration performance dataframes in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_sens_discrimination = pd.concat(pool.starmap(load_test_performance, discrimination_files_per_core),ignore_index=True)
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_sens_calibration = pd.concat(pool.starmap(load_test_performance, calibration_files_per_core),ignore_index=True)

### III. Calculate 95% confidence intervals on sensitivity difference performance metrics
## Discrimination metrics
compiled_sens_discrimination['STATIC_DIFFERENCE'] = compiled_sens_discrimination['VALUE'] - compiled_sens_discrimination['STATIC_VALUE']
compiled_sens_discrimination['FIRST_WINDOW_DIFFERENCE'] = compiled_sens_discrimination['VALUE'] - compiled_sens_discrimination['FIRST_WINDOW_VALUE']
compiled_sens_discrimination = compiled_sens_discrimination.melt(id_vars=['TUNE_IDX','WINDOW_IDX','METRIC','RESAMPLE_IDX'])
sens_CI_static_discrimination = compiled_sens_discrimination.groupby(['TUNE_IDX','METRIC','WINDOW_IDX','variable'],as_index=False)['STATIC_DIFFERENCE'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'resamples':'count'}).reset_index(drop=True)

sens_CI_static_discrimination.insert(3,'BASELINE_TYPE',['Static' for idx in range(sens_CI_static_discrimination.shape[0])])
sens_CI_first_window_discrimination = compiled_sens_discrimination.groupby(['TUNE_IDX','METRIC','WINDOW_IDX'],as_index=False)['FIRST_WINDOW_DIFFERENCE'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'resamples':'count'}).reset_index(drop=True)
sens_CI_first_window_discrimination.insert(3,'BASELINE_TYPE',['FirstWindow' for idx in range(sens_CI_first_window_discrimination.shape[0])])

## Calibration metrics
compiled_sens_calibration['STATIC_DIFFERENCE'] = compiled_sens_calibration['STATIC_VALUE'] - compiled_sens_calibration['VALUE']
compiled_sens_calibration.STATIC_DIFFERENCE[compiled_sens_calibration.METRIC=='CALIB_SLOPE'] = (compiled_sens_calibration.STATIC_VALUE[compiled_sens_calibration.METRIC=='CALIB_SLOPE'] - 1).abs() - (compiled_sens_calibration.VALUE[compiled_sens_calibration.METRIC=='CALIB_SLOPE'] - 1).abs()
compiled_sens_calibration['FIRST_WINDOW_DIFFERENCE'] = compiled_sens_calibration['FIRST_WINDOW_VALUE'] - compiled_sens_calibration['VALUE']
compiled_sens_calibration.FIRST_WINDOW_DIFFERENCE[compiled_sens_calibration.METRIC=='CALIB_SLOPE'] = (compiled_sens_calibration.FIRST_WINDOW_VALUE[compiled_sens_calibration.METRIC=='CALIB_SLOPE'] - 1).abs() - (compiled_sens_calibration.VALUE[compiled_sens_calibration.METRIC=='CALIB_SLOPE'] - 1).abs()
sens_CI_static_calibration = compiled_sens_calibration.groupby(['TUNE_IDX','METRIC','WINDOW_IDX','THRESHOLD'],as_index=False)['STATIC_DIFFERENCE'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'resamples':'count'}).reset_index(drop=True)
sens_CI_static_calibration.insert(3,'BASELINE_TYPE',['Static' for idx in range(sens_CI_static_calibration.shape[0])])
sens_CI_first_window_calibration = compiled_sens_calibration.groupby(['TUNE_IDX','METRIC','WINDOW_IDX','THRESHOLD'],as_index=False)['FIRST_WINDOW_DIFFERENCE'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'resamples':'count'}).reset_index(drop=True)
sens_CI_first_window_calibration.insert(3,'BASELINE_TYPE',['FirstWindow' for idx in range(sens_CI_first_window_calibration.shape[0])])

## Compile and save confidence intervals of both calibration and discrimination metrics
sens_CI_discrimination = pd.concat([sens_CI_static_discrimination,sens_CI_first_window_discrimination],ignore_index=True)
sens_CI_discrimination.to_csv(os.path.join(model_perf_dir,'sensitivity_diff_discrimination_CI.csv'),index=False)
sens_CI_calibration = pd.concat([sens_CI_static_calibration,sens_CI_first_window_calibration],ignore_index=True)
sens_CI_calibration.to_csv(os.path.join(model_perf_dir,'sensitivity_diff_calibration_CI.csv'),index=False)

### IV. Calculate 95% confidence intervals for sensitivity cutoff analysis
## Compile v6-0 predictions for each bootstrap resample
# Search for all performance files
cutoff_files = []
for path in Path(test_bs_dir).rglob('cutoff_discrimination*'):
    cutoff_files.append(str(path.resolve()))

# Characterise the performance files found
cutoff_file_info_df = pd.DataFrame({'FILE':cutoff_files,
                                    'VERSION':[re.search('_performance/(.*)/sensitivity_bootstrapping', curr_file).group(1) for curr_file in cutoff_files],
                                    'METRIC':[re.search('/cutoff_(.*)_rs_', curr_file).group(1) for curr_file in cutoff_files],
                                    'RESAMPLE_IDX':[int(re.search('_rs_(.*).pkl', curr_file).group(1)) for curr_file in cutoff_files]
                                    }).sort_values(by=['METRIC','RESAMPLE_IDX']).reset_index(drop=True)

# Partition performance files across available cores
s = [cutoff_file_info_df.RESAMPLE_IDX.max() // NUM_CORES for _ in range(NUM_CORES)]
s[:(cutoff_file_info_df.RESAMPLE_IDX.max() - sum(s))] = [over+1 for over in s[:(cutoff_file_info_df.RESAMPLE_IDX.max() - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
cutoff_files_per_core = [(cutoff_file_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling sensitivity cutoff performance') for idx in range(len(start_idx))]

# Load testing set discrimination and calibration performance dataframes in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_cutoff_discrimination = pd.concat(pool.starmap(load_test_performance, cutoff_files_per_core),ignore_index=True)

# Calculate difference between discrimination value of remaining population and that of dropout population
compiled_cutoff_discrimination['CUTOFF_DIFFERENCE'] = compiled_cutoff_discrimination['DROPOUT_VALUE'] - compiled_cutoff_discrimination['REMAINING_VALUE']

# Convert dataframe to long form
compiled_cutoff_discrimination = compiled_cutoff_discrimination.melt(id_vars=['TUNE_IDX','WINDOW_IDX','CUTOFF_IDX','METRIC','RESAMPLE_IDX'])

# Calculate confidence intervals
cutoff_CI_discrimination = compiled_cutoff_discrimination.groupby(['TUNE_IDX','METRIC','CUTOFF_IDX','WINDOW_IDX','variable'],as_index=False)['value'].aggregate({'lo':lambda x: np.nanquantile(x,.025),'median':np.nanmedian,'hi':lambda x: np.nanquantile(x,.975),'mean':np.nanmean,'std':np.nanstd,'resamples':lambda x: x.count()}).reset_index(drop=True)

# Save confidence intervals
cutoff_CI_discrimination.to_csv(os.path.join(model_perf_dir,'sensitivity_cutoff_discrimination_CI.csv'),index=False)

# Calculate mean cutoff difference among first 11 windows
baseline_mean_cutoff_difference = compiled_cutoff_discrimination[(compiled_cutoff_discrimination.variable == 'CUTOFF_DIFFERENCE')&(compiled_cutoff_discrimination.WINDOW_IDX!=compiled_cutoff_discrimination.CUTOFF_IDX)].groupby(['TUNE_IDX','METRIC','CUTOFF_IDX','RESAMPLE_IDX'],as_index=False)['value'].mean()

# Calculate confidence intervals
baseline_CI_mean_cutoff_difference = baseline_mean_cutoff_difference.groupby(['TUNE_IDX','METRIC','CUTOFF_IDX'],as_index=False)['value'].aggregate({'lo':lambda x: np.nanquantile(x,.025),'median':np.nanmedian,'hi':lambda x: np.nanquantile(x,.975),'mean':np.nanmean,'std':np.nanstd,'resamples':lambda x: x.count()}).reset_index(drop=True)

# Save confidence intervals
baseline_CI_mean_cutoff_difference.to_csv(os.path.join(model_perf_dir,'sensitivity_cutoff_mean_difference_CI.csv'),index=False)