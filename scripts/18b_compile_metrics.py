#### Master Script 18b: Compile performance metrics and calculate confidence intervals ####
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
from functions.analysis import collect_metrics

# Define directories in which performance metrics are saved
VERSION = 'v5-0'
performance_dir = '../model_performance/'+VERSION

# Define number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()

### II. Compile all performance metrics
# Search for all performance metric files in the directory
metric_files = []
for path in Path(os.path.join(performance_dir)).rglob('*.csv'):
    metric_files.append(str(path.resolve()))

# Characterise list of discovered performance metric files
metric_info_df = pd.DataFrame({'file':metric_files,
                               'TUNE_IDX':[re.search('tune(.*)/resample', curr_file).group(1) for curr_file in metric_files],
                               'RESAMPLE_IDX':[int(re.search('/resample(.*)/', curr_file).group(1)) for curr_file in metric_files],
                               'METRIC':[re.search('/resample(.*).csv', curr_file).group(1) for curr_file in metric_files],
                               'SET':[re.search('metrics_(.*)_set.csv', curr_file).group(1) for curr_file in metric_files]
                              }).sort_values(by=['TUNE_IDX','RESAMPLE_IDX','METRIC']).reset_index(drop=True)
metric_info_df['METRIC'] = metric_info_df['METRIC'].str.rsplit(pat='/', n=1).apply(lambda x: x[1])

# Iterate through unique metric types and compile APM_deep results into a single dataframe
for curr_metric in metric_info_df.METRIC.unique():
    
    # Calculate 95% confidence intervals
    if curr_metric == 'ROCs':
        continue
    elif curr_metric == 'calibration_curves':
        continue
        
    # Filter files of current metric
    curr_metric_info_df = metric_info_df[metric_info_df.METRIC == curr_metric].reset_index(drop=True)
    curr_set = curr_metric_info_df.SET.unique()[0]
    
    # Partition current metric files among cores
    s = [curr_metric_info_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
    s[:(curr_metric_info_df.shape[0] - sum(s))] = [over+1 for over in s[:(curr_metric_info_df.shape[0] - sum(s))]]    
    end_idx = np.cumsum(s)
    start_idx = np.insert(end_idx[:-1],0,0)

    # Collect current metric performance files in parallel
    curr_files_per_core = [(curr_metric_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Metric extraction: '+curr_metric) for idx in range(len(start_idx))]
    with multiprocessing.Pool(NUM_CORES) as pool:
        compiled_curr_metric_values = pd.concat(pool.starmap(collect_metrics, curr_files_per_core),ignore_index=True)
    
    # Calculate 95% confidence intervals
    if curr_metric == 'ROCs':
        CI_compiled_ROCs = compiled_curr_metric_values.groupby(['ADM_OR_DISCH','TUNE_IDX','WINDOW_IDX','THRESHOLD','FPR'],as_index=False)['TPR'].aggregate({'TPR_mean':np.mean,'TPR_std':np.std,'TPR_median':np.median,'TPR_lo':lambda x: np.quantile(x,.025),'TPR_hi':lambda x: np.quantile(x,.975),'resamples':'count'}).reset_index(drop=True)
        CI_compiled_ROCs[['FPR','TPR_mean','TPR_std','TPR_median','TPR_lo','TPR_hi']] = CI_compiled_ROCs[['FPR','TPR_mean','TPR_std','TPR_median','TPR_lo','TPR_hi']].clip(0,1)
        CI_compiled_ROCs.to_csv(os.path.join(performance_dir,'CI_ROCs.csv'))
        
    elif curr_metric == 'calibration_curves':
        CI_compiled_calibration = compiled_curr_metric_values.groupby(['ADM_OR_DISCH','TUNE_IDX','WINDOW_IDX','THRESHOLD','PREDPROB'],as_index=False)['TRUEPROB'].aggregate({'TRUEPROB_mean':np.mean,'TRUEPROB_std':np.std,'TRUEPROB_median':np.median,'TRUEPROB_lo':lambda x: np.quantile(x,.025),'TRUEPROB_hi':lambda x: np.quantile(x,.975),'resamples':'count'}).reset_index(drop=True)
        CI_compiled_calibration[['PREDPROB','TRUEPROB_mean','TRUEPROB_std','TRUEPROB_median','TRUEPROB_lo','TRUEPROB_hi']] = CI_compiled_calibration[['PREDPROB','TRUEPROB_mean','TRUEPROB_std','TRUEPROB_median','TRUEPROB_lo','TRUEPROB_hi']].clip(0,1)
        CI_compiled_calibration.to_csv(os.path.join(performance_dir,'CI_calibration.csv'))

    elif curr_metric.startswith('overall_metrics'):
        CI_overall = compiled_curr_metric_values.groupby(['ADM_OR_DISCH','TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False)['VALUE'].aggregate({'mean':np.mean,'std':np.std,'median':np.median,'lo':lambda x: np.quantile(x,.025),'hi':lambda x: np.quantile(x,.975),'resamples':'count'}).reset_index(drop=True)
        CI_overall['SET'] = curr_set
        CI_overall.to_csv(os.path.join(performance_dir,curr_set+'_CI_overall_metrics.csv'),index=False)

    elif curr_metric.startswith('threshold_metrics'):
        macro_compiled_threshold = compiled_curr_metric_values.groupby(['ADM_OR_DISCH','TUNE_IDX','WINDOW_IDX','RESAMPLE_IDX','METRIC'],as_index=False)['VALUE'].mean()
        macro_compiled_threshold['THRESHOLD'] = 'Average'
        compiled_curr_metric_values = pd.concat([compiled_curr_metric_values,macro_compiled_threshold],ignore_index=True)
        CI_threshold = compiled_curr_metric_values.groupby(['ADM_OR_DISCH','TUNE_IDX','WINDOW_IDX','THRESHOLD','METRIC'],as_index=False)['VALUE'].aggregate({'mean':np.mean,'std':np.std,'median':np.median,'lo':lambda x: np.quantile(x,.025),'hi':lambda x: np.quantile(x,.975),'resamples':'count'}).reset_index(drop=True)
        CI_threshold['SET'] = curr_set
        CI_threshold.to_csv(os.path.join(performance_dir,curr_set+'_CI_threshold_metrics.csv'),index=False)