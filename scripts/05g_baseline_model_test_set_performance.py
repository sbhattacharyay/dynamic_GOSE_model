#### Master Script 5g: Calculate metrics for test set performance in baseline model for comparison ####
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
from functions.analysis import calc_test_ORC, calc_test_thresh_calibration, calc_test_Somers_D, calc_test_thresh_calib_curves

# Set version code
VERSION = 'LOGREG_v1-0'

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/CPM_outputs/'+VERSION

# Define and initialise baseline model performance directory based on version code
model_perf_dir = '/home/sb2406/rds/hpc-work/model_performance/BaselineComparison'
os.makedirs(model_perf_dir,exist_ok=True)

# Define and create subdirectory to store testing set bootstrapping results
test_bs_dir = os.path.join(model_perf_dir,'testing_set_bootstrapping')
os.makedirs(test_bs_dir,exist_ok=True)

# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../legacy_cross_validation_splits.csv')
partitions = cv_splits[['REPEAT','FOLD']].drop_duplicates().reset_index(drop=True)
study_GUPI_GOSE = cv_splits[['GUPI','GOSE']].drop_duplicates()

# Load bootstrapping resample dataframe for testing set performance
# bs_resamples = pd.read_pickle(os.path.join(model_perf_dir,'test_perf_bs_resamples.pkl'))
# bs_resamples = pd.read_pickle(os.path.join(model_perf_dir,'bs_resamples.pkl'))
bs_resamples = pd.read_pickle(os.path.join('/home/sb2406/rds/hpc-work/model_performance/v6-0','bs_resamples.pkl'))

# Load post-dropout tuning grid
# filt_tuning_grid = pd.read_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'))
# filt_tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))
# filt_tuning_grid = filt_tuning_grid[filt_tuning_grid.TUNE_IDX==135].reset_index(drop=True)

### II. Calculate testing set calibration and discrimination metrics based on provided bootstrapping resample row index
# Argument-induced bootstrapping functions
def main(array_task_id):
    
    # Extract current bootstrapping resample parameters
    curr_rs_idx = bs_resamples.RESAMPLE_IDX[array_task_id]
    curr_GUPIs = bs_resamples.GUPIs[array_task_id]
    
    # Load and filter compiled testing set
#     test_predictions_df = pd.read_pickle(os.path.join(model_dir,'compiled_test_predictions.pkl'))
    test_predictions_df = pd.read_csv(os.path.join(model_dir,'compiled_mnlr_test_predictions.csv'))
    test_predictions_df = test_predictions_df[test_predictions_df.GUPI.isin(curr_GUPIs)].reset_index(drop=True)

    # Fix labelling of `TrueLabel`
    old_labels = np.sort(test_predictions_df.TrueLabel.unique())
    test_predictions_df['TrueLabel'] = test_predictions_df.TrueLabel.apply(lambda x: np.where(old_labels == x)[0][0])

    # # Remove logit columns from dataframe
    # logit_cols = [col for col in test_predictions_df if col.startswith('z_GOSE=')]
    # test_predictions_df = test_predictions_df.drop(columns=logit_cols).reset_index(drop=True)
    
    # Calculate intermediate values for metric calculation
    prob_cols = [col for col in test_predictions_df if col.startswith('Pr(GOSE=')]
    prob_matrix = test_predictions_df[prob_cols]
    prob_matrix.columns = list(range(prob_matrix.shape[1]))
    index_vector = np.array(list(range(7)), ndmin=2).T
    test_predictions_df['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
    
    ## Repeat predictions by number of window indices
    # Load study windows and filter to in-resample GUPIs
    study_windows = pd.read_csv('/home/sb2406/rds/hpc-work/timestamps/window_timestamps.csv')
    study_windows = study_windows[study_windows.GUPI.isin(curr_GUPIs)].drop(columns=['TimeStampStart','TimeStampEnd']).reset_index(drop=True)
    
    # Add window indices and totals to static prediction values
    test_predictions_df = test_predictions_df.merge(study_windows).reset_index(drop=True)
    test_predictions_df = test_predictions_df[test_predictions_df.WindowIdx>=12].reset_index(drop=True)
    test_predictions_df['TUNE_IDX'] = 1

    # Calculate from-discharge window indices
    from_discharge_test_predictions_df = test_predictions_df.copy()
    from_discharge_test_predictions_df['WindowIdx'] = from_discharge_test_predictions_df['WindowIdx'] - from_discharge_test_predictions_df['WindowTotal'] - 1
    
    # Create array of unique tuning indices
    uniq_tuning_indices = test_predictions_df.TUNE_IDX.unique()
    
    # Calculate testing set ORC for every Tuning Index, Window Index combination
    testing_set_ORCs = calc_test_ORC(test_predictions_df,list(range(12,85)),True,'Calculating testing set ORC')
    
    # Calculate from-discharge testing set ORC for every Tuning Index, Window Index combination
    from_discharge_testing_set_ORCs = calc_test_ORC(from_discharge_test_predictions_df,list(range(-84,0)),True,'Calculating testing set ORC from discharge')
    
    # Calculate testing set Somers' D for every Tuning Index, Window Index combination
    testing_set_Somers_D = calc_test_Somers_D(test_predictions_df,list(range(12,85)),True,'Calculating testing set Somers D')
    
    # Calculate from-discharge testing set Somers' D for every Tuning Index, Window Index combination
    from_discharge_testing_set_Somers_D = calc_test_Somers_D(from_discharge_test_predictions_df,list(range(-84,0)),True,'Calculating testing set Somers D from discharge')
    
    # Concatenate testing discrimination metrics, add resampling index and save
    testing_set_discrimination = pd.concat([testing_set_ORCs,from_discharge_testing_set_ORCs,testing_set_Somers_D,from_discharge_testing_set_Somers_D],ignore_index=True)
    testing_set_discrimination['RESAMPLE_IDX'] = curr_rs_idx
    testing_set_discrimination.to_pickle(os.path.join(test_bs_dir,'test_discrimination_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))
    
    # Calculate testing set threshold-level calibration metrics for every Tuning Index, Window Index combination
    testing_set_thresh_calibration = calc_test_thresh_calibration(test_predictions_df,list(range(12,85)),True,'Calculating testing set threshold calibration metrics')
    
    # Calculate testing set from-discharge threshold-level calibration metrics for every Tuning Index, Window Index combination
    from_discharge_testing_set_thresh_calibration = calc_test_thresh_calibration(from_discharge_test_predictions_df,list(range(-84,0)),True,'Calculating testing set threshold calibration metrics from discharge')    
    
    # Compile testing calibration from-admission and from-discharge metrics
    testing_set_thresh_calibration = pd.concat([testing_set_thresh_calibration,from_discharge_testing_set_thresh_calibration],ignore_index=True)
    
    # Calculate macro-average calibration slopes across the thresholds
    macro_average_thresh_calibration = testing_set_thresh_calibration.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
    macro_average_thresh_calibration.insert(2,'THRESHOLD',['Average' for idx in range(macro_average_thresh_calibration.shape[0])])

    # Add macro-average information to threshold-level calibration dataframe and sort
    testing_set_thresh_calibration = pd.concat([testing_set_thresh_calibration,macro_average_thresh_calibration],ignore_index=True).sort_values(by=['TUNE_IDX','WINDOW_IDX','THRESHOLD','METRIC']).reset_index(drop=True)
    
    # Add resampling index and save
    testing_set_thresh_calibration['RESAMPLE_IDX'] = curr_rs_idx
    testing_set_thresh_calibration.to_pickle(os.path.join(test_bs_dir,'test_calibration_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))
    
    # # Calculate calbration curves from admission
    # testing_set_thresh_calib_curves = calc_test_thresh_calib_curves(test_predictions_df,[12,24,36,48,60,72,84],True,'Calculating testing set threshold calibration curves')

    # # Calculate calbration curves from discharge
    # from_discharge_testing_set_thresh_calib_curves = calc_test_thresh_calib_curves(from_discharge_test_predictions_df,[-12,-24,-36,-48,-60,-72,-84],True,'Calculating testing set threshold calibration curves from discharge')
    
    # # Compile testing calibration from-admission and from-discharge curves
    # testing_set_thresh_calib_curves = pd.concat([testing_set_thresh_calib_curves,from_discharge_testing_set_thresh_calib_curves],ignore_index=True)
    
    # # Add resampling index and save
    # testing_set_thresh_calib_curves['RESAMPLE_IDX'] = curr_rs_idx
    # testing_set_thresh_calib_curves.to_pickle(os.path.join(test_bs_dir,'test_calib_curves_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])
    main(array_task_id)