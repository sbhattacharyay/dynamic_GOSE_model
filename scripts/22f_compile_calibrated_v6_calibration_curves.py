#### Master Script 22f: Compile performance metrics and calculate confidence intervals ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile all performance metrics
# III. Calculate confidence intervals on performance metrics

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
from argparse import ArgumentParser
from pandas.api.types import CategoricalDtype
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# TQDM for progress tracking
from tqdm import tqdm

# Custom analysis functions
from functions.analysis import collect_metrics, collect_calib_curves

# Define directories in which performance metrics are saved
VERSION = 'v6-0'
performance_dir = '/home/sb2406/rds/hpc-work/model_performance/'+VERSION

# Define number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()

### II. Compile all performance metrics
# Search for all performance metric files in the directory
adm_calib_curves = []
for path in Path(os.path.join(performance_dir)).rglob('*_from_adm.pkl'):
    adm_calib_curves.append(str(path.resolve()))

# Search for all performance metric files in the directory
disch_calib_curves = []
for path in Path(os.path.join(performance_dir)).rglob('*_from_disch.pkl'):
    disch_calib_curves.append(str(path.resolve()))

# Compile calibration curve files
calib_curves = adm_calib_curves + disch_calib_curves
    
# Characterise list of discovered performance calibration curves
calib_curves_info_df = pd.DataFrame({'file':calib_curves,
                                     'TUNE_IDX':[re.search('tune(.*)/resample', curr_file).group(1) for curr_file in calib_curves],
                                     'RESAMPLE_IDX':[int(re.search('/resample(.*)/performance_', curr_file).group(1)) for curr_file in calib_curves],
                                     'METRIC':[re.search('/performance_(.*)_set_', curr_file).group(1) for curr_file in calib_curves],
                                     'SET':[re.search('_set_(.*)_from_', curr_file).group(1) for curr_file in calib_curves],
                                     'ADM_OR_DISCH':[re.search('_from_(.*).pkl', curr_file).group(1) for curr_file in calib_curves]
                                    }).sort_values(by=['TUNE_IDX','RESAMPLE_IDX','METRIC']).reset_index(drop=True)

calib_curves_info_df = calib_curves_info_df[calib_curves_info_df.SET == 'test'].reset_index(drop=True)

# Iterate through unique metric types and compile results into a single dataframe
for curr_metric in calib_curves_info_df.METRIC.unique():

    # Filter files of current metric
    curr_calib_curves_info_df = calib_curves_info_df[calib_curves_info_df.METRIC == curr_metric].reset_index(drop=True)
    curr_set = curr_calib_curves_info_df.SET.unique()[0]
    
    # Partition current metric files among cores
    s = [curr_calib_curves_info_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
    s[:(curr_calib_curves_info_df.shape[0] - sum(s))] = [over+1 for over in s[:(curr_calib_curves_info_df.shape[0] - sum(s))]]    
    end_idx = np.cumsum(s)
    start_idx = np.insert(end_idx[:-1],0,0)

    # Collect current metric performance files in parallel
    curr_files_per_core = [(curr_calib_curves_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Metric extraction: '+curr_metric) for idx in range(len(start_idx))]
    with multiprocessing.Pool(NUM_CORES) as pool:
        compiled_curr_metric_values = pd.concat(pool.starmap(collect_calib_curves, curr_files_per_core),ignore_index=True)
        
    if curr_metric == 'thresh_calib_curves':
        CI_thresh_calib_curves = compiled_curr_metric_values.groupby(['TUNE_IDX','ADM_OR_DISCH','WINDOW_IDX','THRESHOLD','PRED_PROB'],as_index=False)['TRUE_PROB'].aggregate({'TRUE_PROB_mean':np.mean,'TRUE_PROB_std':np.std,'TRUE_PROB_median':np.median,'TRUE_PROB_lo':lambda x: np.quantile(x,.025),'TRUE_PROB_hi':lambda x: np.quantile(x,.975),'resamples':'count'}).reset_index(drop=True)
        CI_thresh_calib_curves[['PRED_PROB','TRUE_PROB_mean','TRUE_PROB_std','TRUE_PROB_median','TRUE_PROB_lo','TRUE_PROB_hi']] = CI_thresh_calib_curves[['PRED_PROB','TRUE_PROB_mean','TRUE_PROB_std','TRUE_PROB_median','TRUE_PROB_lo','TRUE_PROB_hi']].clip(0,1)
        CI_thresh_calib_curves.to_csv(os.path.join(performance_dir,'CI_thresh_calibration_curves.csv'),index=False)
        
    elif curr_metric == 'acc_conf_calib_curves':
        CI_acc_conf_calib_curves = compiled_curr_metric_values.groupby(['TUNE_IDX','ADM_OR_DISCH','WINDOW_IDX','CONFIDENCE'],as_index=False)['ACCURACY'].aggregate({'ACCURACY_mean':np.mean,'ACCURACY_std':np.std,'ACCURACY_median':np.median,'ACCURACY_lo':lambda x: np.quantile(x,.025),'ACCURACY_hi':lambda x: np.quantile(x,.975),'resamples':'count'}).reset_index(drop=True)
        CI_acc_conf_calib_curves[['CONFIDENCE','ACCURACY_mean','ACCURACY_std','ACCURACY_median','ACCURACY_lo','ACCURACY_hi']] = CI_acc_conf_calib_curves[['CONFIDENCE','ACCURACY_mean','ACCURACY_std','ACCURACY_median','ACCURACY_lo','ACCURACY_hi']].clip(0,1)
        CI_acc_conf_calib_curves.to_csv(os.path.join(performance_dir,'CI_acc_conf_calib_curves.csv'),index=False)