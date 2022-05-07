#### Master Script 23c: Compile performance metrics and calculate confidence intervals ####
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
from functions.analysis import collect_LBM

# Define directories in which performance metrics are saved
VERSION = 'v6-0'
performance_dir = '/home/sb2406/rds/hpc-work/model_performance/'+VERSION

# Define number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Define a directory for the storage of LBM values
lbm_dir = os.path.join('/home/sb2406/rds/hpc-work/model_interpretations/'+VERSION,'LBM')

# Load model checkpoint information dataframe
ckpt_info = pd.read_pickle(os.path.join(lbm_dir,'ckpt_info.pkl'))

# Load number of total windows per patient
total_windows_per_pt = pd.read_csv('/home/sb2406/rds/hpc-work/timestamps/window_limit_timestamps.csv')[['GUPI','WindowTotal']].drop_duplicates().reset_index(drop=True)

### II. Compile all LBM values
# Search for all performance metric files in the directory
attribution_files = []
for path in Path(os.path.join(lbm_dir)).rglob('LBM_dataframe.pkl'):
    attribution_files.append(str(path.resolve()))

# Characterise list of discovered performance metric files
lbm_attr_info_df = pd.DataFrame({'file':attribution_files,
                                 'VERSION':[re.search('model_interpretations/(.*)/LBM/repeat', curr_file).group(1) for curr_file in attribution_files],
                                 'REPEAT':[int(re.search('repeat(.*)/fold', curr_file).group(1)) for curr_file in attribution_files],
                                 'FOLD':[int(re.search('/fold(.*)/GUPI_', curr_file).group(1)) for curr_file in attribution_files],
                                 'GUPI':[re.search('GUPI_(.*)_THRESHOLDIDX_', curr_file).group(1) for curr_file in attribution_files],
                                 'THRESHOLD_IDX':[int(re.search('THRESHOLDIDX_(.*)/LBM_dataframe.pkl', curr_file).group(1)) for curr_file in attribution_files]
                              }).sort_values(by=['FOLD','GUPI','THRESHOLD_IDX']).reset_index(drop=True)
threshold_vector = ['GOSE<=1','GOSE<=3','GOSE<=4','GOSE<=5','GOSE<=6','GOSE<=7']
lbm_attr_info_df['THRESHOLD'] = lbm_attr_info_df['THRESHOLD_IDX'].apply(lambda x: threshold_vector[x])
lbm_attr_info_df = lbm_attr_info_df.merge(total_windows_per_pt,how='left',on='GUPI')

# Load all LBM dataframes in parallel 
s = [lbm_attr_info_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
s[:(lbm_attr_info_df.shape[0] - sum(s))] = [over+1 for over in s[:(lbm_attr_info_df.shape[0] - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)

# Collect current metric performance files in parallel
curr_files_per_core = [(lbm_attr_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'LBM extraction') for idx in range(len(start_idx))]
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_LBM = pd.concat(pool.starmap(collect_LBM, curr_files_per_core),ignore_index=True)

aggregated_LBM = compiled_LBM.groupby(['TOKEN','THRESHOLD','TUNE_IDX'],as_index=False).SUM_ATTRIBUTION.sum()
aggregated_LBM['BASELINE'] = aggregated_LBM['TOKEN'].str.startswith('Baseline')
aggregated_LBM['NUMERIC'] = aggregated_LBM['TOKEN'].str.contains('_BIN')
aggregated_LBM['MISSING'] = ((aggregated_LBM.NUMERIC)&(aggregated_LBM['TOKEN'].str.endswith('_BIN_missing')))|((~aggregated_LBM.NUMERIC)&(aggregated_LBM['TOKEN'].str.endswith('_NA')))
aggregated_LBM = aggregated_LBM[~aggregated_LBM.MISSING].reset_index(drop=True)
aggregated_LBM['FINAL_ATTRIBUTION'] = aggregated_LBM['SUM_ATTRIBUTION']/1550
group_maxes = aggregated_LBM.groupby(['BASELINE','THRESHOLD'],as_index=False)['FINAL_ATTRIBUTION'].aggregate({'GROUP_MAX':'max'})
aggregated_LBM = aggregated_LBM.merge(group_maxes,how='left',on=['BASELINE','THRESHOLD'])
aggregated_LBM['FINAL_ATTRIBUTION'] = (aggregated_LBM['FINAL_ATTRIBUTION'])/(aggregated_LBM['GROUP_MAX'])
aggregated_LBM = aggregated_LBM.sort_values(by=['FINAL_ATTRIBUTION'],ascending=[False]).reset_index(drop=True)

