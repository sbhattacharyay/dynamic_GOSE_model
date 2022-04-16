#### Master Script 22g: Assess model new ORC ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate performance metrics on resamples

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
import seaborn as sns
import multiprocessing
from scipy import stats
from pathlib import Path
from ast import literal_eval
import matplotlib.pyplot as plt
from collections import Counter
from scipy.special import logit
from argparse import ArgumentParser
from pandas.api.types import CategoricalDtype
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, recall_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder, StandardScaler, minmax_scale
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

# StatsModel methods
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

# TQDM for progress tracking
from tqdm import tqdm

# Custom analysis functions
from functions.analysis import calc_ORC, calc_ECE, calc_thresh_calibration, calc_MCE, calc_Somers_D

# Create directory to store model performance metrics
VERSION = 'v6-0'
perf_dir = '/home/sb2406/rds/hpc-work/model_performance/'+VERSION
os.makedirs(perf_dir,exist_ok=True)

# Establish number of resamples for bootstrapping
NUM_RESAMP = 1000

# Load cross-validation information to get GOSE and GUPIs
cv_splits = pd.read_csv('../cross_validation_splits.csv')
study_GUPI_GOSE = cv_splits[['GUPI','GOSE']].drop_duplicates()

# If bootstrapping resamples don't exist, create them
if not os.path.exists(os.path.join(perf_dir,'bs_resamples.pkl')):
    
    # Make resamples for bootstrapping metrics
    bs_rs_GUPIs = []
    for i in range(NUM_RESAMP):
        np.random.seed(i)
        curr_GUPIs = np.unique(np.random.choice(study_GUPI_GOSE.GUPI,size=study_GUPI_GOSE.shape[0],replace=True))
        bs_rs_GUPIs.append(curr_GUPIs)
        
    # Create Data Frame to store bootstrapping resmaples 
    bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':bs_rs_GUPIs})
    
    # Save bootstrapping resample dataframe
    bs_resamples.to_pickle(os.path.join(perf_dir,'bs_resamples.pkl'))
    
# Otherwise, load the pre-defined bootstrapping resamples
else:
    bs_resamples = pd.read_pickle(os.path.join(perf_dir,'bs_resamples.pkl'))

# Define model version directory
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Load tuning grid of current model version
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv')).rename(columns={'repeat':'REPEAT','fold':'FOLD'})
tuning_grid = tuning_grid[tuning_grid.TUNE_IDX.isin([69,135])].drop(columns=['REPEAT','FOLD']).drop_duplicates().reset_index(drop=True)
set_grid = pd.DataFrame({'SET':['val','test'],'key':1})
tuning_grid['key'] = 1
bs_resamples['key'] = 1
rs_model_combos = pd.merge(bs_resamples,tuning_grid,how='outer',on='key').merge(set_grid,how='outer',on='key').drop(columns='key')

### II. Calculate performance metrics on resamples
# Define metric calculation function
def main(array_task_id):

    # Get resample information for current trial
    curr_gupis = rs_model_combos.GUPIs[array_task_id]
    curr_rs_idx = rs_model_combos.RESAMPLE_IDX[array_task_id]
    curr_tune_idx = rs_model_combos.TUNE_IDX[array_task_id]
    curr_set = rs_model_combos.SET[array_task_id]
    
    # Create directory to save current combination outputs
    metric_dir = os.path.join(perf_dir,'tune'+str(curr_tune_idx).zfill(4),'resample'+str(curr_rs_idx).zfill(4))
    os.makedirs(metric_dir,exist_ok=True)
    
    # Load compiled set predictions
    compiled_set_preds = pd.read_csv(os.path.join(model_dir,'compiled_'+curr_set+'_predictions.csv'))
    
    # Filter out predictions of current tuning index
    compiled_set_preds = compiled_set_preds[compiled_set_preds.TUNE_IDX == curr_tune_idx].reset_index(drop=True)
    
    # Define sequence of window indices for model assessment
    window_indices = list(range(1,85))
    
    # Filter out current in-group GUPIs
    curr_is_preds = compiled_set_preds[compiled_set_preds.GUPI.isin(curr_gupis)].reset_index(drop=True)
    curr_is_preds.TrueLabel = curr_is_preds.TrueLabel.astype(int)
    curr_is_preds.TUNE_IDX = curr_is_preds.TUNE_IDX.astype(int)

    # Calculate cumulative probabilities at each threshold
    prob_cols = [col for col in curr_is_preds if col.startswith('Pr(GOSE=')]
    logit_cols = [col for col in curr_is_preds if col.startswith('z_GOSE=')]
    
    # Isolate probability matrix
    curr_is_preds = curr_is_preds.drop(columns=logit_cols)
    prob_matrix = curr_is_preds[prob_cols]
    prob_matrix.columns = list(range(prob_matrix.shape[1]))
    index_vector = np.array(list(range(7)), ndmin=2).T
    curr_is_preds['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
    curr_is_preds['PredLabel'] = prob_matrix.idxmax(axis=1)
    
    ### New ORC
    orcs = []
    for curr_wi in tqdm(window_indices):
        filt_is_preds = curr_is_preds[curr_is_preds.WindowIdx == curr_wi].reset_index(drop=True)
        aucs = []
        for ix, (a, b) in enumerate(itertools.combinations(np.sort(filt_is_preds.TrueLabel.unique()), 2)):
            filt_prob_matrix = filt_is_preds[filt_is_preds.TrueLabel.isin([a,b])].reset_index(drop=True)
            filt_prob_matrix['ConditLabel'] = (filt_prob_matrix.TrueLabel == b).astype(int)
            aucs.append(roc_auc_score(filt_prob_matrix['ConditLabel'],filt_prob_matrix['ExpectedValue']))
        orcs.append(pd.DataFrame({'TUNE_IDX':curr_tune_idx,
                                  'RESAMPLE_IDX':curr_rs_idx,
                                  'WINDOW_IDX':curr_wi,
                                  'METRIC':'NEW_ORC',
                                  'VALUE':np.mean(aucs)},index=[0]))
    orcs = pd.concat(orcs,ignore_index=True)
    
    ### Compile overall metrics into a single dataframe
    orcs.to_csv(os.path.join(metric_dir,'performance_set_'+curr_set+'_new_orc.csv'),index=False)
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])
    main(array_task_id)