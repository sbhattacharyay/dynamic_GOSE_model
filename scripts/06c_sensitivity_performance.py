#### Master Script 6c: Calculate metrics for test set performance for sensitivity analysis ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate testing set sensitivity analysis metrics based on provided bootstrapping resample row index

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
VERSION = 'v6-0'

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Define and initialise baseline model performance directory based on version code
model_perf_dir = '/home/sb2406/rds/hpc-work/model_performance/'+VERSION

# Define and create subdirectory to store testing set bootstrapping results
test_bs_dir = os.path.join(model_perf_dir,'sensitivity_bootstrapping')
os.makedirs(test_bs_dir,exist_ok=True)

# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../legacy_cross_validation_splits.csv')
partitions = cv_splits[['REPEAT','FOLD']].drop_duplicates().reset_index(drop=True)
study_GUPI_GOSE = cv_splits[['GUPI','GOSE']].drop_duplicates()

# Load bootstrapping resample dataframe for testing set performance
bs_resamples = pd.read_pickle(os.path.join(model_perf_dir,'sensitivity_bs_resamples.pkl'))

# Load bootstrapping resample dataframe for testing set cutoff analysis
cutoff_bs_resamples = pd.read_pickle(os.path.join(model_perf_dir,'cutoff_analysis_bs_resamples.pkl'))

### II. Calculate testing set sensitivity analysis metrics based on provided bootstrapping resample row index
# Argument-induced bootstrapping functions
def main(array_task_id):
    
    # Extract current bootstrapping resample parameters
    curr_rs_idx = array_task_id + 1
    curr_bs_rs = bs_resamples[bs_resamples.RESAMPLE_IDX==curr_rs_idx].rename(columns={'WINDOW_IDX':'WindowIdx','GUPIs':'GUPI'}).reset_index(drop=True)
    from_disch_curr_bs_rs = curr_bs_rs.copy()
    from_disch_curr_bs_rs.WindowIdx = -from_disch_curr_bs_rs.WindowIdx

    # Load and filter compiled testing set
    test_predictions_df = pd.read_csv(os.path.join(model_dir,'compiled_test_predictions.csv'))
    test_predictions_df = test_predictions_df[test_predictions_df.TUNE_IDX==135].reset_index(drop=True)

    # Load and filter compiled static-only testing set
    static_predictions_df = pd.read_pickle(os.path.join(model_dir,'compiled_static_only_test_predictions.pkl'))

    # Calculate intermediate values for metric calculation
    prob_cols = [col for col in test_predictions_df if col.startswith('Pr(GOSE=')]
    prob_matrix = test_predictions_df[prob_cols]
    prob_matrix.columns = list(range(prob_matrix.shape[1]))
    static_prob_matrix = static_predictions_df[prob_cols]
    static_prob_matrix.columns = list(range(static_prob_matrix.shape[1]))
    index_vector = np.array(list(range(7)), ndmin=2).T
    test_predictions_df['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
    static_predictions_df['ExpectedValue'] = np.matmul(static_prob_matrix.values,index_vector)

    # Remove logit columns from dataframes
    logit_cols = [col for col in test_predictions_df if col.startswith('z_GOSE=')]
    test_predictions_df = test_predictions_df.drop(columns=logit_cols).reset_index(drop=True)
    static_predictions_df = static_predictions_df.drop(columns=logit_cols).reset_index(drop=True)

    # Calculate study window totals to add 'WindowTotal' column
    study_window_totals = test_predictions_df.groupby(['GUPI','TrueLabel'],as_index=False)['WindowIdx'].max().rename(columns={'WindowIdx':'WindowTotal'})
    test_predictions_df = test_predictions_df.merge(study_window_totals[['GUPI','WindowTotal']],how='left')
    static_predictions_df = static_predictions_df.merge(study_window_totals[['GUPI','WindowTotal']],how='left')

    # Calculate from-discharge window indices
    from_discharge_test_predictions_df = test_predictions_df.copy()
    from_discharge_test_predictions_df['WindowIdx'] = from_discharge_test_predictions_df['WindowIdx'] - from_discharge_test_predictions_df['WindowTotal'] - 1
    from_discharge_static_predictions_df = static_predictions_df.copy()
    from_discharge_static_predictions_df['WindowIdx'] = from_discharge_static_predictions_df['WindowIdx'] - from_discharge_static_predictions_df['WindowTotal'] - 1

    # Calculate testing set ORC for every Tuning Index, Window Index combination of in-sample sets and calculate difference
    testing_set_ORCs = calc_test_ORC(test_predictions_df.merge(curr_bs_rs[['GUPI','WindowIdx']],how='inner'),list(range(1,85)),True,'Calculating testing set ORC')
    static_set_ORCs = calc_test_ORC(static_predictions_df.merge(curr_bs_rs[['GUPI','WindowIdx']],how='inner'),list(range(1,85)),True,'Calculating static-only testing set ORC').rename(columns={'VALUE':'STATIC_VALUE'})
    combined_set_ORCs = testing_set_ORCs.merge(static_set_ORCs,how='left')
    # combined_set_ORCs['ADDED_VALUE'] = combined_set_ORCs['VALUE'] - combined_set_ORCs['STATIC_VALUE']

    # Calculate from-discharge testing set ORC for every Tuning Index, Window Index combination
    from_discharge_testing_set_ORCs = calc_test_ORC(from_discharge_test_predictions_df.merge(from_disch_curr_bs_rs[['GUPI','WindowIdx']],how='inner'),list(range(-85,0)),True,'Calculating testing set ORC from discharge')
    from_discharge_static_set_ORCs = calc_test_ORC(from_discharge_static_predictions_df.merge(from_disch_curr_bs_rs[['GUPI','WindowIdx']],how='inner'),list(range(-85,0)),True,'Calculating static-only testing set ORC from discharge').rename(columns={'VALUE':'STATIC_VALUE'})
    from_discharge_combined_set_ORCs = from_discharge_testing_set_ORCs.merge(from_discharge_static_set_ORCs,how='left')
    # from_discharge_combined_set_ORCs['ADDED_VALUE'] = from_discharge_combined_set_ORCs['VALUE'] - from_discharge_combined_set_ORCs['STATIC_VALUE']

    # Calculate testing set Somers' D for every Tuning Index, Window Index combination
    testing_set_Somers_D = calc_test_Somers_D(test_predictions_df.merge(curr_bs_rs[['GUPI','WindowIdx']],how='inner'),list(range(1,85)),True,'Calculating testing set Somers D')
    static_set_Somers_D = calc_test_Somers_D(static_predictions_df.merge(curr_bs_rs[['GUPI','WindowIdx']],how='inner'),list(range(1,85)),True,'Calculating static-only testing set Somers D').rename(columns={'VALUE':'STATIC_VALUE'})
    combined_set_Somers_D = testing_set_Somers_D.merge(static_set_Somers_D,how='left')
    # combined_set_Somers_D['ADDED_VALUE'] = combined_set_Somers_D['VALUE'] - combined_set_Somers_D['STATIC_VALUE']

    # Calculate from-discharge testing set Somers' D for every Tuning Index, Window Index combination
    from_discharge_testing_set_Somers_D = calc_test_Somers_D(from_discharge_test_predictions_df.merge(from_disch_curr_bs_rs[['GUPI','WindowIdx']],how='inner'),list(range(-85,0)),True,'Calculating testing set Somers D from discharge')
    from_discharge_static_set_Somers_D = calc_test_Somers_D(from_discharge_static_predictions_df.merge(from_disch_curr_bs_rs[['GUPI','WindowIdx']],how='inner'),list(range(-85,0)),True,'Calculating static-only testing set Somers D from discharge').rename(columns={'VALUE':'STATIC_VALUE'})
    from_discharge_combined_set_Somers_D = from_discharge_testing_set_Somers_D.merge(from_discharge_static_set_Somers_D,how='left')
    # from_discharge_combined_set_Somers_D['ADDED_VALUE'] = from_discharge_combined_set_Somers_D['VALUE'] - from_discharge_combined_set_Somers_D['STATIC_VALUE']

    # Concatenate testing discrimination metrics, add resampling index and save
    testing_set_discrimination = pd.concat([combined_set_ORCs,from_discharge_combined_set_ORCs,combined_set_Somers_D,from_discharge_combined_set_Somers_D],ignore_index=True)
    # testing_set_discrimination['RESAMPLE_IDX'] = curr_rs_idx
    # testing_set_discrimination.to_pickle(os.path.join(test_bs_dir,'test_discrimination_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))
    
    # Calculate testing set threshold-level calibration metrics for every Tuning Index, Window Index combination
    testing_set_thresh_calibration = calc_test_thresh_calibration(test_predictions_df.merge(curr_bs_rs[['GUPI','WindowIdx']],how='inner'),list(range(1,85)),True,'Calculating testing set threshold calbration metrics')
    static_set_thresh_calibration = calc_test_thresh_calibration(static_predictions_df.merge(curr_bs_rs[['GUPI','WindowIdx']],how='inner'),list(range(1,85)),True,'Calculating static-only testing set threshold calbration metrics').rename(columns={'VALUE':'STATIC_VALUE'})
    combined_set_thresh_calibration = testing_set_thresh_calibration.merge(static_set_thresh_calibration,how='left')
    # combined_set_thresh_calibration['ADDED_VALUE'] = combined_set_thresh_calibration['STATIC_VALUE'] - combined_set_thresh_calibration['VALUE']
    # combined_set_thresh_calibration.ADDED_VALUE[combined_set_thresh_calibration.METRIC=='CALIB_SLOPE'] = (combined_set_thresh_calibration.STATIC_VALUE[combined_set_thresh_calibration.METRIC=='CALIB_SLOPE'] - 1).abs() - (combined_set_thresh_calibration.VALUE[combined_set_thresh_calibration.METRIC=='CALIB_SLOPE'] - 1).abs()

    # Calculate testing set from-discharge threshold-level calibration metrics for every Tuning Index, Window Index combination
    from_discharge_testing_set_thresh_calibration = calc_test_thresh_calibration(from_discharge_test_predictions_df.merge(from_disch_curr_bs_rs[['GUPI','WindowIdx']],how='inner'),list(range(-85,0)),True,'Calculating testing set threshold calbration metrics from discharge')
    from_discharge_static_set_thresh_calibration = calc_test_thresh_calibration(from_discharge_static_predictions_df.merge(from_disch_curr_bs_rs[['GUPI','WindowIdx']],how='inner'),list(range(-85,0)),True,'Calculating static-only testing set threshold calbration metrics from discharge').rename(columns={'VALUE':'STATIC_VALUE'})
    from_discharge_combined_set_thresh_calibration = from_discharge_testing_set_thresh_calibration.merge(from_discharge_static_set_thresh_calibration,how='left')
    # from_discharge_combined_set_thresh_calibration['ADDED_VALUE'] = from_discharge_combined_set_thresh_calibration['STATIC_VALUE'] - from_discharge_combined_set_thresh_calibration['VALUE']
    # from_discharge_combined_set_thresh_calibration.ADDED_VALUE[from_discharge_combined_set_thresh_calibration.METRIC=='CALIB_SLOPE'] = (from_discharge_combined_set_thresh_calibration.STATIC_VALUE[from_discharge_combined_set_thresh_calibration.METRIC=='CALIB_SLOPE'] - 1).abs() - (from_discharge_combined_set_thresh_calibration.VALUE[from_discharge_combined_set_thresh_calibration.METRIC=='CALIB_SLOPE'] - 1).abs()

    # Compile testing calibration from-admission and from-discharge metrics
    testing_set_thresh_calibration = pd.concat([combined_set_thresh_calibration,from_discharge_combined_set_thresh_calibration],ignore_index=True)
    
    # Calculate macro-average calibration slopes across the thresholds
    macro_average_thresh_calibration = testing_set_thresh_calibration.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
    macro_average_thresh_calibration.insert(2,'THRESHOLD',['Average' for idx in range(macro_average_thresh_calibration.shape[0])])
    static_macro_average_thresh_calibration = testing_set_thresh_calibration.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).STATIC_VALUE.mean()
    static_macro_average_thresh_calibration.insert(2,'THRESHOLD',['Average' for idx in range(static_macro_average_thresh_calibration.shape[0])])
    macro_average_thresh_calibration = macro_average_thresh_calibration.merge(static_macro_average_thresh_calibration,how='left')
    # macro_average_thresh_calibration['ADDED_VALUE'] = macro_average_thresh_calibration['STATIC_VALUE'] - macro_average_thresh_calibration['VALUE']
    # macro_average_thresh_calibration.ADDED_VALUE[macro_average_thresh_calibration.METRIC=='CALIB_SLOPE'] = (macro_average_thresh_calibration.STATIC_VALUE[macro_average_thresh_calibration.METRIC=='CALIB_SLOPE'] - 1).abs() - (macro_average_thresh_calibration.VALUE[macro_average_thresh_calibration.METRIC=='CALIB_SLOPE'] - 1).abs()

    # Add macro-average information to threshold-level calibration dataframe and sort
    testing_set_thresh_calibration = pd.concat([testing_set_thresh_calibration,macro_average_thresh_calibration],ignore_index=True).sort_values(by=['TUNE_IDX','WINDOW_IDX','THRESHOLD','METRIC']).reset_index(drop=True)
    
    # Add resampling index and save
    # testing_set_thresh_calibration['RESAMPLE_IDX'] = curr_rs_idx
    # testing_set_thresh_calibration.to_pickle(os.path.join(test_bs_dir,'test_calibration_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))

    ## Calculate difference between first window performance and performance over time
    # Create empty lists to store discrimination and calibration performance dataframes
    first_window_discrimination_list = []
    first_window_calibrations_list = []

    # Iterate through after-admission window indices
    for curr_wi in tqdm(range(1,85),'Calculating differential increase in performance over time'):
        
        # Extract current GUPI set
        curr_wi_GUPIs = curr_bs_rs[curr_bs_rs.WindowIdx==curr_wi].reset_index(drop=True)
        curr_wi_GUPIs['WindowIdx'] = 1
        
        # Calculate first window ORC of current window index set
        first_window_set_ORCs = calc_test_ORC(test_predictions_df.merge(curr_wi_GUPIs[['GUPI','WindowIdx']],how='inner'),[1],False,'Calculating first window set ORC').rename(columns={'VALUE':'FIRST_WINDOW_VALUE'})
        first_window_set_ORCs['WINDOW_IDX'] = curr_wi

        # Calculate first window Somers' D of current window index set
        first_window_set_Somers_D = calc_test_Somers_D(test_predictions_df.merge(curr_wi_GUPIs[['GUPI','WindowIdx']],how='inner'),[1],False,'Calculating first window set Somers D').rename(columns={'VALUE':'FIRST_WINDOW_VALUE'})
        first_window_set_Somers_D['WINDOW_IDX'] = curr_wi
        
        # Append first window ORC and Somers' D to running discrimination list
        first_window_discrimination_list.append(first_window_set_ORCs)
        first_window_discrimination_list.append(first_window_set_Somers_D)

        # Calculate first window threshold-level calibration of current window index set
        first_window_set_thresh_calibration = calc_test_thresh_calibration(test_predictions_df.merge(curr_wi_GUPIs[['GUPI','WindowIdx']],how='inner'),[1],False,'Calculating first window set threshold calbration metrics').rename(columns={'VALUE':'FIRST_WINDOW_VALUE'})
        first_window_set_thresh_calibration['WINDOW_IDX'] = curr_wi

        # Append first window threshold-calibration to running list
        first_window_calibrations_list.append(first_window_set_thresh_calibration)

    # Iterate through before-discharge window indices
    for curr_wi in tqdm(range(-85,0),'Calculating differential increase in performance over time'):
        
        # Extract current GUPI set
        curr_wi_GUPIs = from_disch_curr_bs_rs[from_disch_curr_bs_rs.WindowIdx==curr_wi].reset_index(drop=True)
        curr_wi_GUPIs['WindowIdx'] = 1
        
        # Calculate first window ORC of current window index set
        from_disch_first_window_set_ORCs = calc_test_ORC(test_predictions_df.merge(curr_wi_GUPIs[['GUPI','WindowIdx']],how='inner'),[1],False,'Calculating first window set ORC').rename(columns={'VALUE':'FIRST_WINDOW_VALUE'})
        from_disch_first_window_set_ORCs['WINDOW_IDX'] = curr_wi

        # Calculate first window Somers' D of current window index set
        from_disch_first_window_set_Somers_D = calc_test_Somers_D(test_predictions_df.merge(curr_wi_GUPIs[['GUPI','WindowIdx']],how='inner'),[1],False,'Calculating first window set Somers D').rename(columns={'VALUE':'FIRST_WINDOW_VALUE'})
        from_disch_first_window_set_Somers_D['WINDOW_IDX'] = curr_wi
        
        # Append first window ORC and Somers' D to running discrimination list
        first_window_discrimination_list.append(from_disch_first_window_set_ORCs)
        first_window_discrimination_list.append(from_disch_first_window_set_Somers_D)

        # Calculate first window threshold-level calibration of current window index set
        from_disch_first_window_set_thresh_calibration = calc_test_thresh_calibration(test_predictions_df.merge(curr_wi_GUPIs[['GUPI','WindowIdx']],how='inner'),[1],False,'Calculating first window set threshold calbration metrics').rename(columns={'VALUE':'FIRST_WINDOW_VALUE'})
        from_disch_first_window_set_thresh_calibration['WINDOW_IDX'] = curr_wi

        # Append first window threshold-calibration to running list
        first_window_calibrations_list.append(from_disch_first_window_set_thresh_calibration)

    # Concatenate running lists into dataframe
    first_window_discriminations = pd.concat(first_window_discrimination_list,ignore_index=True)
    first_window_calibrations = pd.concat(first_window_calibrations_list,ignore_index=True)

    # Calculate and append macro-averaged metrics for threshold-level calibration metrics
    first_window_macro_average_thresh_calibration = first_window_calibrations.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).FIRST_WINDOW_VALUE.mean()
    first_window_macro_average_thresh_calibration.insert(2,'THRESHOLD',['Average' for idx in range(first_window_macro_average_thresh_calibration.shape[0])])
    first_window_calibrations = pd.concat([first_window_calibrations,first_window_macro_average_thresh_calibration],ignore_index=True).sort_values(by=['TUNE_IDX','WINDOW_IDX','THRESHOLD','METRIC']).reset_index(drop=True)

    # Merge first-window metrics to compiled discrimination and calibration dataframes
    testing_set_discrimination = testing_set_discrimination.merge(first_window_discriminations,how='left')
    testing_set_thresh_calibration = testing_set_thresh_calibration.merge(first_window_calibrations,how='left')

    ## Add resampling indices to compiled dataframes and save
    # Add resampling indices
    testing_set_discrimination['RESAMPLE_IDX'] = curr_rs_idx
    testing_set_thresh_calibration['RESAMPLE_IDX'] = curr_rs_idx

    # Save compiled dataframes
    testing_set_discrimination.to_pickle(os.path.join(test_bs_dir,'diff_discrimination_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))
    testing_set_thresh_calibration.to_pickle(os.path.join(test_bs_dir,'diff_calibration_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))

    ## Perform cut-off analysis
    # Extract current remaining dataset
    curr_cutoff_bs_rs = cutoff_bs_resamples[cutoff_bs_resamples.RESAMPLE_IDX==curr_rs_idx].rename(columns={'WINDOW_IDX':'WindowIdx','GUPIs':'GUPI'}).reset_index(drop=True)

    # Initiate empty lists to store cutoff-analysis results
    cutoff_discrimination_list = []
    #cutoff_calibrations_list = []

    # Iterate through cutoffs
    for curr_wi_cutoff in tqdm(curr_cutoff_bs_rs.WindowIdx.unique(),'Calculating differential increase in performance at each cutoff'):
        
        # Extract current cutoff remaining and dropout sets
        remaining_cutoff_GUPIs = curr_cutoff_bs_rs[(curr_cutoff_bs_rs.WindowIdx==curr_wi_cutoff)&(curr_cutoff_bs_rs.SAMPLE=='Remaining')].GUPI.unique()
        dropout_cutoff_GUPIs = curr_cutoff_bs_rs[(curr_cutoff_bs_rs.WindowIdx==curr_wi_cutoff)&(curr_cutoff_bs_rs.SAMPLE=='Dropout')].GUPI.unique()

        # Calculate remaining and drouput set ORCs
        remaining_set_ORCs = calc_test_ORC(test_predictions_df[test_predictions_df.GUPI.isin(remaining_cutoff_GUPIs)],list(range(1,curr_wi_cutoff+1)),False,'Calculating remaining testing set ORC').rename(columns={'VALUE':'REMAINING_VALUE'})
        dropout_set_ORCs = calc_test_ORC(test_predictions_df[test_predictions_df.GUPI.isin(dropout_cutoff_GUPIs)],list(range(1,curr_wi_cutoff+1)),False,'Calculating dropout testing set ORC').rename(columns={'VALUE':'DROPOUT_VALUE'})

        # Calculate remaining and drouput set Somers D
        remaining_set_Somers_D = calc_test_Somers_D(test_predictions_df[test_predictions_df.GUPI.isin(remaining_cutoff_GUPIs)],list(range(1,curr_wi_cutoff+1)),False,'Calculating remaining testing set Somers D').rename(columns={'VALUE':'REMAINING_VALUE'})
        dropout_set_Somers_D = calc_test_Somers_D(test_predictions_df[test_predictions_df.GUPI.isin(dropout_cutoff_GUPIs)],list(range(1,curr_wi_cutoff+1)),False,'Calculating dropout testing set Somers D').rename(columns={'VALUE':'DROPOUT_VALUE'})

        # Calculate remaining and drouput set threshold-level calibration metrics
        #remaining_set_thresh_calibration = calc_test_thresh_calibration(test_predictions_df[test_predictions_df.GUPI.isin(remaining_cutoff_GUPIs)],list(range(1,curr_wi_cutoff+1)),False,'Calculating remaining testing set threshold-level calibration metrics').rename(columns={'VALUE':'REMAINING_VALUE'})
        #dropout_set_thresh_calibration = calc_test_thresh_calibration(test_predictions_df[test_predictions_df.GUPI.isin(dropout_cutoff_GUPIs)],list(range(1,curr_wi_cutoff+1)),False,'Calculating dropout testing set threshold-level calibration metrics').rename(columns={'VALUE':'DROPOUT_VALUE'})

        # Merge remaining and dropout dataframes
        cutoff_ORCs = remaining_set_ORCs.merge(dropout_set_ORCs,how='left')
        cutoff_Somers_D = remaining_set_Somers_D.merge(dropout_set_Somers_D,how='left')
        #cutoff_thresh_calibration = remaining_set_thresh_calibration.merge(dropout_set_thresh_calibration,how='left')

        # Add cutoff information
        cutoff_ORCs.insert(2,'CUTOFF_IDX',[curr_wi_cutoff for idx in range(cutoff_ORCs.shape[0])])
        cutoff_Somers_D.insert(2,'CUTOFF_IDX',[curr_wi_cutoff for idx in range(cutoff_Somers_D.shape[0])])
        #cutoff_thresh_calibration.insert(2,'CUTOFF_IDX',[curr_wi_cutoff for idx in range(cutoff_thresh_calibration.shape[0])])

        # Append dataframes to running lists
        cutoff_discrimination_list.append(cutoff_ORCs)
        cutoff_discrimination_list.append(cutoff_Somers_D)
        #cutoff_calibrations_list.append(cutoff_thresh_calibration)

    # Concatenate cutoff lists
    cutoff_discrimination = pd.concat(cutoff_discrimination_list,ignore_index=True)
    #cutoff_calibrations = pd.concat(cutoff_calibrations_list,ignore_index=True)

    # Calculate and append macro-averaged metrics for threshold-level calibration metrics
    #remaining_macro_average_thresh_calibration = cutoff_calibrations.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).REMAINING_VALUE.mean()
    #dropout_macro_average_thresh_calibration = cutoff_calibrations.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).DROPOUT_VALUE.mean()
    #cutoff_macro_average_thresh_calibration = remaining_macro_average_thresh_calibration.merge(dropout_macro_average_thresh_calibration,how='left')
    #cutoff_macro_average_thresh_calibration.insert(2,'THRESHOLD',['Average' for idx in range(cutoff_macro_average_thresh_calibration.shape[0])])
    #cutoff_calibrations = pd.concat([cutoff_calibrations,cutoff_macro_average_thresh_calibration],ignore_index=True).sort_values(by=['TUNE_IDX','WINDOW_IDX','THRESHOLD','METRIC']).reset_index(drop=True)

    # Add resampling indices
    cutoff_discrimination['RESAMPLE_IDX'] = curr_rs_idx
    #cutoff_calibrations['RESAMPLE_IDX'] = curr_rs_idx

    # Save compiled dataframes
    cutoff_discrimination.to_pickle(os.path.join(test_bs_dir,'cutoff_discrimination_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))
    #cutoff_calibrations.to_pickle(os.path.join(test_bs_dir,'cutoff_calibration_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])
    main(array_task_id)
