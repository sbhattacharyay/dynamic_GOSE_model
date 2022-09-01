#### Master Script 04d: Compile validation set performance results for configuration dropout of dynamic all-predictor-based models ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile and save validation and testing set predictions across partitions
# III. Determine top-performing tuning configurations based on validation set calibration and discrimination
# IV. Create bootstrapping resamples for dropping out poorly calibrated configurations

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
from shutil import rmtree
from ast import literal_eval
import matplotlib.pyplot as plt
from scipy.special import logit
from collections import Counter
from argparse import ArgumentParser
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
from functions.model_building import load_calibrated_predictions
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
# Search for all prediction files
pred_files = []
for path in Path(model_dir).rglob('*_predictions.csv'):
    pred_files.append(str(path.resolve()))

# Characterise the prediction files found
pred_file_info_df = pd.DataFrame({'FILE':pred_files,
                                  'FOLD':[int(re.search('/fold(.*)/tune', curr_file).group(1)) for curr_file in pred_files],
                                  'TUNE_IDX':[int(re.search('/tune(.*)/', curr_file).group(1)) for curr_file in pred_files],
                                  'VERSION':[re.search('_outputs/(.*)/fold', curr_file).group(1) for curr_file in pred_files],
                                  'CALIBRATION':[re.search('tune(.*)_predictions.csv', curr_file).group(1) for curr_file in pred_files],
                                  'SET':[re.search('calibrated_(.*)_predictions.csv', curr_file).group(1) for curr_file in pred_files]
                                 }).sort_values(by=['FOLD','TUNE_IDX','SET']).reset_index(drop=True)
pred_file_info_df['CALIBRATION'] = pred_file_info_df['CALIBRATION'].str.rsplit(pat='/', n=1).apply(lambda x: x[1])
pred_file_info_df['CALIBRATION'] = pred_file_info_df['CALIBRATION'].str.rsplit(pat='_', n=1).apply(lambda x: x[0])

# Focus only on calibrated predictions
pred_file_info_df = pred_file_info_df[pred_file_info_df.CALIBRATION == 'calibrated'].reset_index(drop=True)

# Partition predictions across available cores
s = [pred_file_info_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
s[:(pred_file_info_df.shape[0] - sum(s))] = [over+1 for over in s[:(pred_file_info_df.shape[0] - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
curr_files_per_core = [(pred_file_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling calibrated predictions') for idx in range(len(start_idx))]
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_predictions_df = pd.concat(pool.starmap(load_calibrated_predictions, curr_files_per_core),ignore_index=True)
    
# Separate prediction files by set
val_predictions_df = compiled_predictions_df[compiled_predictions_df.SET == 'val'].reset_index(drop=True)
test_predictions_df = compiled_predictions_df[compiled_predictions_df.SET == 'test'].reset_index(drop=True)

# Save prediction files appropriately
val_predictions_df.to_pickle(os.path.join(model_dir,'compiled_val_predictions.pkl'))
test_predictions_df.to_pickle(os.path.join(model_dir,'compiled_test_predictions.pkl'))

### III. Determine top-performing tuning configurations based on validation set calibration and discrimination
## Load compiled validation set
val_predictions_df = pd.read_pickle(os.path.join(model_dir,'compiled_val_predictions.pkl'))

# Create array of unique tuning indices
uniq_tuning_indices = val_predictions_df.TUNE_IDX.unique()

# Calculate intermediate values for metric calculation
prob_cols = [col for col in val_predictions_df if col.startswith('Pr(GOSE=')]
logit_cols = [col for col in val_predictions_df if col.startswith('z_GOSE=')]
prob_matrix = val_predictions_df[prob_cols]
prob_matrix.columns = list(range(prob_matrix.shape[1]))
index_vector = np.array(list(range(7)), ndmin=2).T
val_predictions_df['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
val_predictions_df['PredLabel'] = prob_matrix.idxmax(axis=1)

# Partition tuning indices across available cores
s = [len(uniq_tuning_indices) // NUM_CORES for _ in range(NUM_CORES)]
s[:(len(uniq_tuning_indices) - sum(s))] = [over+1 for over in s[:(len(uniq_tuning_indices) - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
curr_preds_per_core = [(val_predictions_df[val_predictions_df.TUNE_IDX.isin(uniq_tuning_indices[start_idx[idx]:end_idx[idx]])].reset_index(drop=True),list(range(1,85)),True,'Calculating validation set performance metrics') for idx in range(len(start_idx))]

## Determine average ordinal discrimination (ORC) perfomance of each configuration
# Calculate validation set ORC for every Tuning Index, Window Index combination
with multiprocessing.Pool(NUM_CORES) as pool:
    validation_set_ORCs = pd.concat(pool.starmap(calc_val_ORC, curr_preds_per_core),ignore_index=True)

# Save validation set ORCs
validation_set_ORCs.to_csv(os.path.join(model_perf_dir,'val_set_ORCs.csv'),index=False)
    
# Calculate average ORC for each tuning index
ave_validation_set_ORCs = validation_set_ORCs.groupby('TUNE_IDX',as_index=False).VALUE.mean().rename(columns={'VALUE':'ORC'}).sort_values(by='ORC',ascending=False).reset_index(drop=True)

## Determine average threshold-level calibration slope of each configuration
# Calculate validation set threshold-level calibration slope for every Tuning Index, Window Index combination
with multiprocessing.Pool(NUM_CORES) as pool:
    validation_set_thresh_calibration = pd.concat(pool.starmap(calc_val_thresh_calibration, curr_preds_per_core),ignore_index=True)

# Save validation set threshold-level calibration slopes
macro_average_thresh_calibration = validation_set_thresh_calibration.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
macro_average_thresh_calibration.insert(2,'THRESHOLD',['Average' for idx in range(macro_average_thresh_calibration.shape[0])])
validation_set_thresh_calibration = pd.concat([validation_set_thresh_calibration,macro_average_thresh_calibration],ignore_index=True).sort_values(by=['TUNE_IDX','WINDOW_IDX','THRESHOLD']).reset_index(drop=True)
validation_set_thresh_calibration.to_csv(os.path.join(model_perf_dir,'val_set_thresh_calibration_slopes.csv'),index=False)

# Calculate average threshold-level calibration slopes for each tuning index
ave_validation_set_thresh_calibration = validation_set_thresh_calibration[validation_set_thresh_calibration.THRESHOLD=='Average'].reset_index(drop=True).groupby(['TUNE_IDX','WINDOW_IDX'],as_index=False).VALUE.mean().rename(columns={'VALUE':'CALIB_SLOPE'})
ave_validation_set_thresh_calibration['ERROR'] = (ave_validation_set_thresh_calibration.CALIB_SLOPE - 1).abs()
ave_validation_set_thresh_calibration = ave_validation_set_thresh_calibration.sort_values(by='ERROR',ascending=True).reset_index(drop=True)

# For each `WINDOW_IDX`, identify the optimal tuning index
opt_val_calibration_configs = ave_validation_set_thresh_calibration.loc[ave_validation_set_thresh_calibration.groupby('WINDOW_IDX').ERROR.idxmin()].reset_index(drop=True)
opt_val_calibration_configs.to_csv(os.path.join(model_perf_dir,'optimal_val_set_calibration_configurations.csv'),index=False)

### IV. Create bootstrapping resamples for dropping out poorly calibrated configurations
## Load validation set predictions and optimal configurations
# Load compiled validation set
val_predictions_df = pd.read_pickle(os.path.join(model_dir,'compiled_val_predictions.pkl'))

# Load optimal tuning configurations for each window index based on validation set performance
opt_val_calibration_configs = pd.read_csv(os.path.join(model_perf_dir,'optimal_val_set_calibration_configurations.csv'))

## Create bootstrapping resamples
# Create array of unique validation set GUPIs
uniq_GUPIs = val_predictions_df.GUPI.unique()

# Filter out GUPI-GOSE combinations that are in the validation set
val_GUPI_GOSE = study_GUPI_GOSE[study_GUPI_GOSE.GUPI.isin(uniq_GUPIs)].reset_index(drop=True)

# Make stratified resamples for bootstrapping metrics
bs_rs_GUPIs = [resample(val_GUPI_GOSE.GUPI.values,replace=True,n_samples=val_GUPI_GOSE.shape[0],stratify=val_GUPI_GOSE.GOSE.values) for _ in range(NUM_RESAMP)]
bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':bs_rs_GUPIs})

# Save bootstrapping resample dataframe
bs_resamples.to_pickle(os.path.join(model_perf_dir,'val_dropout_bs_resamples.pkl'))