#### Master Script 4c: Calculate validation set calibration and discrimination for dropout ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate validation set calibration and discrimination based on provided bootstrapping resample row index

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

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Define model performance directory based on version code
model_perf_dir = '/home/sb2406/rds/hpc-work/model_performance/'+VERSION

# Define and create subdirectory to store validation set bootstrapping results
val_bs_dir = os.path.join(model_perf_dir,'validation_set_bootstrapping')
os.makedirs(val_bs_dir,exist_ok=True)

# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
partitions = cv_splits[['FOLD']].drop_duplicates().reset_index(drop=True)
study_GUPI_GOSE = cv_splits[['GUPI','GOSE']].drop_duplicates()

# Load bootstrapping resample dataframe for validation set dropout
bs_resamples = pd.read_pickle(os.path.join(model_perf_dir,'val_dropout_bs_resamples.pkl'))

# Load validation set ORCs
validation_set_ORCs = pd.read_csv(os.path.join(model_perf_dir,'val_set_ORCs.csv'))

# For each `WINDOW_IDX`, identify the optimal tuning index based on discrimination
opt_val_discrimination_configs = validation_set_ORCs.loc[validation_set_ORCs.groupby('WINDOW_IDX').VALUE.idxmax()].reset_index(drop=True)

# Load optimal tuning configurations for each window index based on validation set performance
opt_val_calibration_configs = pd.read_csv(os.path.join(model_perf_dir,'optimal_val_set_calibration_configurations.csv'))

### II. Calculate validation set calibration and discrimination based on provided bootstrapping resample row index
# Argument-induced bootstrapping functions
def main(array_task_id):
    
    # Extract current bootstrapping resample parameters
    curr_rs_idx = bs_resamples.RESAMPLE_IDX[array_task_id]
    curr_GUPIs = bs_resamples.GUPIs[array_task_id]
    
    # Load and filter compiled validation set
    val_predictions_df = pd.read_pickle(os.path.join(model_dir,'compiled_val_predictions.pkl'))
    val_predictions_df = val_predictions_df[val_predictions_df.GUPI.isin(curr_GUPIs)].reset_index(drop=True)
    
    # Remove logit columns from dataframe
    logit_cols = [col for col in val_predictions_df if col.startswith('z_GOSE=')]
    val_predictions_df = val_predictions_df.drop(columns=logit_cols).reset_index(drop=True)
    
    # Calculate intermediate values for metric calculation
    prob_cols = [col for col in val_predictions_df if col.startswith('Pr(GOSE=')]
    prob_matrix = val_predictions_df[prob_cols]
    prob_matrix.columns = list(range(prob_matrix.shape[1]))
    index_vector = np.array(list(range(7)), ndmin=2).T
    val_predictions_df['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
    
    # Create array of unique tuning indices
    uniq_tuning_indices = val_predictions_df.TUNE_IDX.unique()
    
    # Partition tuning indices across available cores
    s = [len(uniq_tuning_indices) // NUM_CORES for _ in range(NUM_CORES)]
    s[:(len(uniq_tuning_indices) - sum(s))] = [over+1 for over in s[:(len(uniq_tuning_indices) - sum(s))]]    
    end_idx = np.cumsum(s)
    start_idx = np.insert(end_idx[:-1],0,0)
    curr_preds_per_core = [(val_predictions_df[val_predictions_df.TUNE_IDX.isin(uniq_tuning_indices[start_idx[idx]:end_idx[idx]])].reset_index(drop=True),list(range(1,85)),True,'Calculating validation set performance metrics') for idx in range(len(start_idx))]

    # Calculate validation set threshold-level calibration slope for every Tuning Index, Window Index combination
    with multiprocessing.Pool(NUM_CORES) as pool:
        validation_set_thresh_calibration = pd.concat(pool.starmap(calc_val_thresh_calibration, curr_preds_per_core),ignore_index=True)

    # Calculate macro-average calibration slopes across the thresholds
    macro_average_thresh_calibration = validation_set_thresh_calibration.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
    macro_average_thresh_calibration.insert(2,'THRESHOLD',['Average' for idx in range(macro_average_thresh_calibration.shape[0])])

    # Calculate error of macro-average calibration slopes and add resample index
    macro_average_thresh_calibration['ERROR'] = (macro_average_thresh_calibration.VALUE - 1).abs()
    macro_average_thresh_calibration['RESAMPLE_IDX'] = curr_rs_idx
    
    # Save current resampling index calibration results
    macro_average_thresh_calibration.to_pickle(os.path.join(val_bs_dir,'val_calibration_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))
    
    # Calculate validation set ORC for every Tuning Index, Window Index combination
    with multiprocessing.Pool(NUM_CORES) as pool:
        validation_set_ORCs = pd.concat(pool.starmap(calc_val_ORC, curr_preds_per_core),ignore_index=True)
    
    # Add resample index and save current resampling index discrimination results
    validation_set_ORCs['RESAMPLE_IDX'] = curr_rs_idx
    validation_set_ORCs.to_pickle(os.path.join(val_bs_dir,'val_ORC_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)