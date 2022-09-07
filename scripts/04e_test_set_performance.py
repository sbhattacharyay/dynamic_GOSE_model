#### Master Script 4e: Calculate metrics for test set performance ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate testing set calibration and discrimination metrics based on provided bootstrapping resample row index

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
from functions.analysis import calc_test_ORC, calc_test_thresh_calibration, calc_test_Somers_D

# Set version code
VERSION = 'v7-0'

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Define model performance directory based on version code
model_perf_dir = '/home/sb2406/rds/hpc-work/model_performance/'+VERSION

# Define and create subdirectory to store testing set bootstrapping results
test_bs_dir = os.path.join(model_perf_dir,'testing_set_bootstrapping')
os.makedirs(test_bs_dir,exist_ok=True)

# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
partitions = cv_splits[['FOLD']].drop_duplicates().reset_index(drop=True)
study_GUPI_GOSE = cv_splits[['GUPI','GOSE']].drop_duplicates()

# Load bootstrapping resample dataframe for testing set performance
bs_resamples = pd.read_pickle(os.path.join(model_perf_dir,'test_perf_bs_resamples.pkl'))

# Load post-dropout tuning grid
filt_tuning_grid = pd.read_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'))

### II. Calculate testing set calibration and discrimination metrics based on provided bootstrapping resample row index
# Argument-induced bootstrapping functions
def main(array_task_id):
    
    # Extract current bootstrapping resample parameters
    curr_rs_idx = bs_resamples.RESAMPLE_IDX[array_task_id]
    curr_GUPIs = bs_resamples.GUPIs[array_task_id]
    
    # Load and filter compiled testing set
    test_predictions_df = pd.read_pickle(os.path.join(model_dir,'compiled_test_predictions.pkl'))
    test_predictions_df = test_predictions_df[(test_predictions_df.GUPI.isin(curr_GUPIs))&(test_predictions_df.TUNE_IDX.isin(filt_tuning_grid.TUNE_IDX))].reset_index(drop=True)
    
    # Remove logit columns from dataframe
    logit_cols = [col for col in test_predictions_df if col.startswith('z_GOSE=')]
    test_predictions_df = test_predictions_df.drop(columns=logit_cols).reset_index(drop=True)
    
    # Calculate intermediate values for metric calculation
    prob_cols = [col for col in test_predictions_df if col.startswith('Pr(GOSE=')]
    prob_matrix = test_predictions_df[prob_cols]
    prob_matrix.columns = list(range(prob_matrix.shape[1]))
    index_vector = np.array(list(range(7)), ndmin=2).T
    test_predictions_df['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
    
    # Create array of unique tuning indices
    uniq_tuning_indices = test_predictions_df.TUNE_IDX.unique()
    
    # Calculate testing set ORC for every Tuning Index, Window Index combination
    testing_set_ORCs = calc_test_ORC(test_predictions_df,list(range(1,85)),True,'Calculating testing set ORC')
    
    # Calculate testing set Somers' D for every Tuning Index, Window Index combination
    testing_set_Somers_D = calc_test_Somers_D(test_predictions_df,list(range(1,85)),True,'Calculating testing set Somers D')
    
    # Concatenate testing discrimination metrics, add resampling index and save
    testing_set_discrimination = pd.concat([testing_set_ORCs,testing_set_Somers_D],ignore_index=True)
    testing_set_discrimination['RESAMPLE_IDX'] = curr_rs_idx
    testing_set_discrimination.to_pickle(os.path.join(test_bs_dir,'test_discrimination_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))
    
    # Calculate testing set threshold-level calibration metrics for every Tuning Index, Window Index combination
    testing_set_thresh_calibration = calc_test_thresh_calibration(test_predictions_df,list(range(1,85)),True,'Calculating testing set threshold calibration metrics')
    
    # Calculate macro-average calibration slopes across the thresholds
    macro_average_thresh_calibration = testing_set_thresh_calibration.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
    macro_average_thresh_calibration.insert(2,'THRESHOLD',['Average' for idx in range(macro_average_thresh_calibration.shape[0])])

    # Add macro-average information to threshold-level calibration dataframe and sort
    testing_set_thresh_calibration = pd.concat([testing_set_thresh_calibration,macro_average_thresh_calibration],ignore_index=True).sort_values(by=['TUNE_IDX','WINDOW_IDX','THRESHOLD','METRIC']).reset_index(drop=True)
    
    # Add resampling index and save
    testing_set_thresh_calibration['RESAMPLE_IDX'] = curr_rs_idx
    testing_set_thresh_calibration.to_pickle(os.path.join(test_bs_dir,'test_calibration_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])
    main(array_task_id)