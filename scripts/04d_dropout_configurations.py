#### Master Script 04d: Compile validation set performance results for configuration dropout of dynamic all-predictor-based models ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile and save bootstrapped validation set performance dataframes
# III. Dropout configurations based on validation set calibration and discrimination information
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
from functions.analysis import calc_val_ORC, calc_val_thresh_calibration

# Set version code
VERSION = 'v7-0'

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Define model performance directory based on version code
model_perf_dir = '/home/sb2406/rds/hpc-work/model_performance/'+VERSION

# Define subdirectory to store validation set bootstrapping results
val_bs_dir = os.path.join(model_perf_dir,'validation_set_bootstrapping')

# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
partitions = cv_splits[['FOLD']].drop_duplicates().reset_index(drop=True)
study_GUPI_GOSE = cv_splits[['GUPI','GOSE']].drop_duplicates()

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Set number of resamples for bootstrapping-based testing set performance
NUM_RESAMP = 1000

# Load the optimised tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

### II. Compile and save bootstrapped validation set performance dataframes
# Search for all performance files
perf_files = []
for path in Path(val_bs_dir).rglob('val_*'):
    perf_files.append(str(path.resolve()))

# Characterise the performance files found
perf_file_info_df = pd.DataFrame({'FILE':perf_files,
                                  'VERSION':[re.search('_performance/(.*)/validation_set_', curr_file).group(1) for curr_file in perf_files],
                                  'METRIC':[re.search('/val_(.*)_rs_', curr_file).group(1) for curr_file in perf_files],
                                  'RESAMPLE_IDX':[int(re.search('_rs_(.*).pkl', curr_file).group(1)) for curr_file in perf_files],
                                 }).sort_values(by=['METRIC','RESAMPLE_IDX']).reset_index(drop=True)

# Separate ORC and calibration file dataframes
orc_file_info_df = perf_file_info_df[perf_file_info_df.METRIC == 'ORC'].reset_index(drop=True)
calibration_file_info_df = perf_file_info_df[perf_file_info_df.METRIC == 'calibration'].reset_index(drop=True)

# Partition performance files across available cores
s = [orc_file_info_df.RESAMPLE_IDX.max() // NUM_CORES for _ in range(NUM_CORES)]
s[:(orc_file_info_df.RESAMPLE_IDX.max() - sum(s))] = [over+1 for over in s[:(orc_file_info_df.RESAMPLE_IDX.max() - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
orc_files_per_core = [(orc_file_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling validation set ORC') for idx in range(len(start_idx))]
calibration_files_per_core = [(calibration_file_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling validation set calibration') for idx in range(len(start_idx))]

# Load validation set discrimination and calibration performance dataframes in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_val_orc = pd.concat(pool.starmap(load_val_performance, orc_files_per_core),ignore_index=True)
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_val_calibration = pd.concat(pool.starmap(load_val_performance, calibration_files_per_core),ignore_index=True)

### III. Dropout configurations based on validation set calibration and discrimination information
## Identify configurations with calibration slope 1 in confidence interval
# Calculate confidence intervals for each tuning configuration
val_CI_calibration_slope = compiled_val_calibration.groupby(['TUNE_IDX','WINDOW_IDX','THRESHOLD','METRIC'],as_index=False)['VALUE'].aggregate({'lo':lambda x: np.quantile(x,.025),'hi':lambda x: np.quantile(x,.975),'resamples':'count'}).reset_index(drop=True)

# Mark TUNE_IDX/WINDOW_IDX combinations which are significantly calibrated
val_CI_calibration_slope['CALIBRATED'] = ((val_CI_calibration_slope['lo']<=1)&(val_CI_calibration_slope['hi']>=1))

# Concatenate tuning indices of significantly calibrated configurations for each window index
val_calibrated_TIs = val_CI_calibration_slope[val_CI_calibration_slope.CALIBRATED].groupby(['WINDOW_IDX'],as_index=False).TUNE_IDX.aggregate(list).rename(columns={'TUNE_IDX':'CALIB_TUNE_IDX'})

## Identify configurations that are significantly worse than the optimal tuning index
# Load optimal tuning configurations for each window index based on validation set performance
opt_val_calibration_configs = pd.read_csv(os.path.join(model_perf_dir,'optimal_val_set_calibration_configurations.csv')).drop(columns=['CALIB_SLOPE','ERROR']).rename(columns={'TUNE_IDX':'OPT_TUNE_IDX'})

# Add optimal tuning index information to compiled validation set calibration dataframe
compiled_val_calibration = compiled_val_calibration.merge(opt_val_calibration_configs,how='left')

# Identify the optimal error for each WINDOW_IDX/RESAMPLE_IDX combination and merge to compiled dataframe
bs_val_opt_errors = compiled_val_calibration[compiled_val_calibration.TUNE_IDX == compiled_val_calibration.OPT_TUNE_IDX].drop(columns=['THRESHOLD','METRIC','VALUE','TUNE_IDX']).rename(columns={'ERROR':'OPT_ERROR'}).reset_index(drop=True)
compiled_val_calibration = compiled_val_calibration.merge(bs_val_opt_errors,how='left')

# For each WINDOW_IDX/TUNE_IDX combination, calculate the number of times the error is better than the optimal configuration error
compiled_val_calibration['BETTER_THAN_OPT'] = (compiled_val_calibration.ERROR <= compiled_val_calibration.OPT_ERROR)
sig_worse_configs = compiled_val_calibration.groupby(['TUNE_IDX','WINDOW_IDX','THRESHOLD','METRIC','OPT_TUNE_IDX'],as_index=False).BETTER_THAN_OPT.aggregate({'BETTER':'sum','RESAMPLES':'count'}).reset_index(drop=True)
sig_worse_configs['p'] = sig_worse_configs.BETTER/sig_worse_configs.RESAMPLES

# For each WINDOW_IDX, concatenate tuning indices of significantly miscalibrated configurations
sig_worse_configs = sig_worse_configs[sig_worse_configs.p<.05].groupby(['WINDOW_IDX'],as_index=False).TUNE_IDX.aggregate(list)

# Merge information of tuning indices which are significantly calibrated
sig_worse_configs = sig_worse_configs.merge(val_calibrated_TIs,how='left')
sig_worse_configs.TUNE_IDX[~sig_worse_configs.CALIB_TUNE_IDX.isna()] = sig_worse_configs[~sig_worse_configs.CALIB_TUNE_IDX.isna()].apply(lambda x:list(set(x['TUNE_IDX'])-set(x['CALIB_TUNE_IDX'])),axis=1)

## Drop out configurations that are consistently out of range and/or significantly underperforming
flattened_TIs = [item for sublist in sig_worse_configs.TUNE_IDX for item in sublist] 
counts_of_removal = OrderedDict(Counter(flattened_TIs).most_common())
tune_idx_to_remove = [k for (k,v) in counts_of_removal.items() if v >= 80]
tune_idx_to_keep = [k for (k,v) in counts_of_removal.items() if v < 80]

## Identify configurations that are significantly less discriminating than the optimal tuning index
# Load validation set ORCs
validation_set_ORCs = pd.read_csv(os.path.join(model_perf_dir,'val_set_ORCs.csv'))

# For each `WINDOW_IDX`, identify the optimal tuning index based on discrimination
opt_val_discrimination_configs = validation_set_ORCs.loc[validation_set_ORCs.groupby('WINDOW_IDX').VALUE.idxmax()].reset_index(drop=True).drop(columns=['METRIC','VALUE']).rename(columns={'TUNE_IDX':'OPT_TUNE_IDX'})

# Add optimal tuning index information to compiled validation set discrimination dataframe
compiled_val_orc = compiled_val_orc.merge(opt_val_discrimination_configs,how='left')

# Identify the optimal ORC for each WINDOW_IDX/RESAMPLE_IDX combination and merge to compiled dataframe
bs_val_opt_orc = compiled_val_orc[compiled_val_orc.TUNE_IDX == compiled_val_orc.OPT_TUNE_IDX].drop(columns=['METRIC','TUNE_IDX']).rename(columns={'VALUE':'OPT_VALUE'}).reset_index(drop=True)
compiled_val_orc = compiled_val_orc.merge(bs_val_opt_orc,how='left')

# For each WINDOW_IDX/TUNE_IDX combination, calculate the number of times the ORC is better than the optimal configuration ORC
compiled_val_orc['BETTER_THAN_OPT'] = (compiled_val_orc.VALUE >= compiled_val_orc.OPT_VALUE)
sig_less_discrim_configs = compiled_val_orc.groupby(['TUNE_IDX','WINDOW_IDX','METRIC','OPT_TUNE_IDX'],as_index=False).BETTER_THAN_OPT.aggregate({'BETTER':'sum','RESAMPLES':'count'}).reset_index(drop=True)
sig_less_discrim_configs['p'] = sig_less_discrim_configs.BETTER/sig_less_discrim_configs.RESAMPLES

# For each WINDOW_IDX, concatenate tuning indices of significantly under-discriminating configurations
sig_less_discrim_configs = sig_less_discrim_configs[sig_less_discrim_configs.p<.05].groupby(['WINDOW_IDX'],as_index=False).TUNE_IDX.aggregate(list)

## Determine which configurations remain after consideration of calibration and discrimination
# Drop out configurations that are consistently under-discriminating
flattened_ORC_TIs = [item for sublist in sig_less_discrim_configs.TUNE_IDX for item in sublist] 
ORC_counts_of_removal = OrderedDict(Counter(flattened_ORC_TIs).most_common())
ORC_tune_idx_to_remove = [k for (k,v) in ORC_counts_of_removal.items() if v >= 80]
ORC_tune_idx_to_keep = [k for (k,v) in ORC_counts_of_removal.items() if v < 80]

# Find the configurations which remain after calibration and discrimination check
final_tune_idx_to_keep = list(set(ORC_tune_idx_to_keep) & set(tune_idx_to_keep))
filt_tuning_grid = tuning_grid[tuning_grid.TUNE_IDX.isin(final_tune_idx_to_keep)].reset_index(drop=True)
filt_tuning_grid.to_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'),index=False)
dropped_tuning_grid = tuning_grid[~tuning_grid.TUNE_IDX.isin(final_tune_idx_to_keep)].reset_index(drop=True)

### IV. Delete folders of underperforming configurations
## From the tuning grid of underperforming configurations, create a list of folders to delete
delete_folders = [os.path.join(model_dir,'fold'+str(dropped_tuning_grid.FOLD[curr_row]).zfill(1),'tune'+str(dropped_tuning_grid.TUNE_IDX[curr_row]).zfill(4)) for curr_row in range(dropped_tuning_grid.shape[0])]

## Delete folders
for curr_folder in tqdm(delete_folders,"Deleting directories corresponding to underperforming tuning configurations"):
    try:
        shutil.rmtree(curr_folder)
    except:
        pass
    
### V. Create bootstrapping resamples for calculating testing set performance
## Load testing set predictions and optimal configurations
# Load the post-dropout tuning grid
filt_tuning_grid = pd.read_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'))

# Load compiled testing set
test_predictions_df = pd.read_pickle(os.path.join(model_dir,'compiled_test_predictions.pkl'))

# Filter out tuning indices that remain after dropout
test_predictions_df = test_predictions_df[test_predictions_df.TUNE_IDX.isin(filt_tuning_grid.TUNE_IDX)].reset_index(drop=True)

## Create bootstrapping resamples
# Create array of unique testing set GUPIs
uniq_GUPIs = test_predictions_df.GUPI.unique()

# Filter out GUPI-GOSE combinations that are in the testing set
test_GUPI_GOSE = study_GUPI_GOSE[study_GUPI_GOSE.GUPI.isin(uniq_GUPIs)].reset_index(drop=True)

# Make stratified resamples for bootstrapping metrics
bs_rs_GUPIs = [resample(test_GUPI_GOSE.GUPI.values,replace=True,n_samples=test_GUPI_GOSE.shape[0],stratify=test_GUPI_GOSE.GOSE.values) for _ in range(NUM_RESAMP)]
bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':bs_rs_GUPIs})

# Save bootstrapping resample dataframe
bs_resamples.to_pickle(os.path.join(model_perf_dir,'test_perf_bs_resamples.pkl'))