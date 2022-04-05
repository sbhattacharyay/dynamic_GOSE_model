#### Master Script 21e: Compile calibration performance metrics ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile all calibration performance metrics

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

# PyTorch, PyTorch.Text, and Lightning-PyTorch methods
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# SciKit-Learn methods
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder, StandardScaler
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
from functions.analysis import collect_metrics
from functions.model_building import collate_batch, format_tokens, format_time_tokens, T_scaling, vector_scaling

# Set version code
VERSION = 'v6-0'

# Initialise model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION
val_performance_dir = os.path.join(model_dir,'validation_performance')
calibration_dir = os.path.join(model_dir,'calibration_performance')

# Load the optimised tuning grid
calibration_grid = pd.read_csv(os.path.join(calibration_dir,'calibration_grid.csv'))

# Define number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()

### II. Compile all calibration performance metrics
# Search for all calibration metric files in the directory
calib_metric_files = []
for path in Path(os.path.join(calibration_dir)).rglob('*.pkl'):
    calib_metric_files.append(str(path.resolve()))
    
# Characterise calibration metric files
calib_file_info_df = pd.DataFrame({'file':calib_metric_files,
                                   'TUNE_IDX':[int(re.search('/tune_idx_(.*)_opt_', curr_file).group(1)) for curr_file in calib_metric_files],
                                   'OPTIMIZATION':[re.search('_opt_(.*)_window_idx_', curr_file).group(1) for curr_file in calib_metric_files],
                                   'WINDOW_IDX':[int(re.search('_window_idx_(.*)_scaling_', curr_file).group(1)) for curr_file in calib_metric_files],
                                   'SCALING':[re.search('_scaling_(.*).pkl', curr_file).group(1) for curr_file in calib_metric_files],
                                   'VERSION':[re.search('_outputs/(.*)/calibration_performance', curr_file).group(1) for curr_file in calib_metric_files],
                                   'REPEAT':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in calib_metric_files],
                                   'FOLD':[int(re.search('/fold(.*)/tune_idx_', curr_file).group(1)) for curr_file in calib_metric_files]
                                  }).sort_values(['TUNE_IDX','REPEAT','FOLD','WINDOW_IDX','OPTIMIZATION','SCALING']).reset_index(drop=True)

# Load calibration metric files
calibration_metrics_df = pd.concat([pd.read_pickle(f) for f in tqdm(calib_metric_files)],ignore_index=True)

# Average metrics over folds
ave_cal_metrics = calibration_metrics_df.groupby(['TUNE_IDX','OPTIMIZATION','WINDOW_IDX','THRESHOLD','SET','CALIBRATION','METRIC'],as_index=False)['VALUE'].mean()

# Save average calibration metrics
ave_cal_metrics.to_pickle(os.path.join(calibration_dir,'average_metrics.pkl'))

# Load average calibration metrics
ave_cal_metrics = pd.read_pickle(os.path.join(calibration_dir,'average_metrics.pkl'))
ave_cal_metrics = ave_cal_metrics[(ave_cal_metrics.CALIBRATION != 'None')|(ave_cal_metrics.OPTIMIZATION != 'ordinal')].reset_index(drop=True)
ave_cal_metrics = ave_cal_metrics[ave_cal_metrics.SET == 'test'].drop(columns='SET').reset_index(drop=True)
overall_cal_metrics = ave_cal_metrics[ave_cal_metrics.THRESHOLD == 'Overall'].drop(columns='THRESHOLD').reset_index(drop=True)
overall_cal_metrics = overall_cal_metrics.pivot(index=['TUNE_IDX','OPTIMIZATION','WINDOW_IDX','CALIBRATION'], columns='METRIC', values='VALUE').reset_index()
ave_cal_metrics = ave_cal_metrics[ave_cal_metrics.THRESHOLD != 'Overall'].reset_index(drop=True)
ave_cal_metrics = ave_cal_metrics.pivot(index=['TUNE_IDX','OPTIMIZATION','WINDOW_IDX','THRESHOLD','CALIBRATION'], columns='METRIC', values='VALUE').reset_index()
ave_cal_metrics = ave_cal_metrics.merge(overall_cal_metrics,how='left')
ave_cal_metrics.columns.name = None

# Measure abs_diff of calibration slope from 1
ave_cal_metrics['CALIB_SLOPE_ERROR'] = (ave_cal_metrics['CALIB_SLOPE']-1).abs()

# Extract best calibrated configurations for each tuning index and winodow index combination
ave_cal_slopes = ave_cal_metrics[ave_cal_metrics.THRESHOLD == 'Average'].reset_index(drop=True)
best_combos_slopes = ave_cal_slopes.loc[ave_cal_slopes.groupby(['TUNE_IDX','WINDOW_IDX']).CALIB_SLOPE_ERROR.idxmin()].reset_index(drop=True)
best_combos_slopes['BEST'] = 'CALIB_SLOPE'

# Discrimination performance
best_combos_ORCs = ave_cal_metrics.loc[ave_cal_metrics.groupby(['TUNE_IDX','WINDOW_IDX']).ORC.idxmax()].reset_index(drop=True)
best_combos_ORCs['BEST'] = 'ORC'

# ECE performance
best_combos_ECEs = ave_cal_metrics.loc[ave_cal_metrics.groupby(['TUNE_IDX','WINDOW_IDX']).ECE.idxmin()].reset_index(drop=True)
best_combos_ECEs['BEST'] = 'ECE'

# Observe average calibration ICIs
ave_ICI = ave_cal_metrics[ave_cal_metrics.THRESHOLD == 'Average'].reset_index(drop=True)
best_combos_ICIs = ave_ICI.loc[ave_ICI.groupby(['TUNE_IDX','WINDOW_IDX']).ICI.idxmin()].reset_index(drop=True)
best_combos_ICIs['BEST'] = 'ICI'

# Concatenate best performing combos per metric for each tuning index
group_labels = [col for col in best_combos_slopes if col not in ['THRESHOLD','BEST']]
best_combos = pd.concat([best_combos_slopes,best_combos_ORCs,best_combos_ECEs,best_combos_ICIs],ignore_index=True).groupby(group_labels,as_index=False)['BEST'].aggregate(list).sort_values(by=['TUNE_IDX','WINDOW_IDX','CALIBRATION','OPTIMIZATION'],ignore_index=True)
best_combos['OPT_COUNT'] = best_combos['BEST'].apply(len)
#best_combos[best_combos.OPT_COUNT != 4]

# TI 135 best calibration combos
best_135_combos = best_combos[best_combos.TUNE_IDX==135].reset_index(drop=True)
best_135_combos = best_135_combos[(best_135_combos.CALIBRATION == 'None')|(best_135_combos.WINDOW_IDX<=3)].reset_index(drop=True)
best_135_combos = best_135_combos[(best_135_combos.WINDOW_IDX > 3)|((best_135_combos.WINDOW_IDX <= 3)&(best_135_combos.OPTIMIZATION == 'nominal')&(best_135_combos.CALIBRATION == 'vector'))].reset_index(drop=True)
best_135_combos.to_pickle(os.path.join(calibration_dir,'best_configurations_tune_0135.pkl'))

# TI 69 best calibration combos
best_69_combos = best_combos[best_combos.TUNE_IDX==69].reset_index(drop=True)
best_69_combos = best_69_combos.merge(best_combos_ORCs[best_combos_ORCs.TUNE_IDX==69].rename(columns={'ORC':'BEST_ORC'}).reset_index(drop=True)[['TUNE_IDX','WINDOW_IDX','BEST_ORC']],how='left')
best_69_combos['ORC_ERROR'] = (best_69_combos['BEST_ORC'] - best_69_combos['ORC']).abs()
best_69_combos = best_69_combos[best_69_combos['ORC_ERROR'] < .02].reset_index(drop=True)
best_69_combos = best_69_combos.loc[best_69_combos.groupby(['TUNE_IDX','WINDOW_IDX']).CALIB_SLOPE_ERROR.idxmin()].reset_index(drop=True)
best_69_combos.to_pickle(os.path.join(calibration_dir,'best_configurations_tune_0069.pkl'))

# Identify best calibration combinations per metric
plt.plot(best_combos_slopes.WINDOW_IDX[best_combos_slopes.TUNE_IDX==135],best_combos_slopes.ORC[best_combos_slopes.TUNE_IDX==135],label='SLOPES')
plt.plot(best_combos_ORCs.WINDOW_IDX[best_combos_ORCs.TUNE_IDX==135],best_combos_ORCs.ORC[best_combos_ORCs.TUNE_IDX==135],label='ORC')
plt.plot(best_combos_ECEs.WINDOW_IDX[best_combos_ECEs.TUNE_IDX==135],best_combos_ECEs.ORC[best_combos_ECEs.TUNE_IDX==135],label='ECE')
plt.plot(best_combos_ICIs.WINDOW_IDX[best_combos_ICIs.TUNE_IDX==135],best_combos_ICIs.ORC[best_combos_ICIs.TUNE_IDX==135],label='ICI')
plt.ylabel('ORC')
plt.ylim([0.675, .735])
plt.legend()
plt.show()

plt.plot(best_combos_slopes.WINDOW_IDX[best_combos_slopes.TUNE_IDX==135],best_combos_slopes.CALIB_SLOPE[best_combos_slopes.TUNE_IDX==135],label='SLOPES')
plt.plot(best_combos_ORCs.WINDOW_IDX[best_combos_ORCs.TUNE_IDX==135],best_combos_ORCs.CALIB_SLOPE[best_combos_ORCs.TUNE_IDX==135],label='ORC')
plt.plot(best_combos_ECEs.WINDOW_IDX[best_combos_ECEs.TUNE_IDX==135],best_combos_ECEs.CALIB_SLOPE[best_combos_ECEs.TUNE_IDX==135],label='ECE')
plt.plot(best_combos_ICIs.WINDOW_IDX[best_combos_ICIs.TUNE_IDX==135],best_combos_ICIs.CALIB_SLOPE[best_combos_ICIs.TUNE_IDX==135],label='ICI')
plt.ylabel('Calibration slope')
plt.ylim([0.9, 1.4])
plt.legend()
plt.show()

plt.plot(best_combos_slopes.WINDOW_IDX[best_combos_slopes.TUNE_IDX==135],best_combos_slopes.ECE[best_combos_slopes.TUNE_IDX==135],label='SLOPES')
plt.plot(best_combos_ORCs.WINDOW_IDX[best_combos_ORCs.TUNE_IDX==135],best_combos_ORCs.ECE[best_combos_ORCs.TUNE_IDX==135],label='ORC')
plt.plot(best_combos_ECEs.WINDOW_IDX[best_combos_ECEs.TUNE_IDX==135],best_combos_ECEs.ECE[best_combos_ECEs.TUNE_IDX==135],label='ECE')
plt.plot(best_combos_ICIs.WINDOW_IDX[best_combos_ICIs.TUNE_IDX==135],best_combos_ICIs.ECE[best_combos_ICIs.TUNE_IDX==135],label='ICI')
plt.ylabel('ECE')
plt.legend()
plt.show()

plt.plot(best_combos_slopes.WINDOW_IDX[best_combos_slopes.TUNE_IDX==135],best_combos_slopes.ICI[best_combos_slopes.TUNE_IDX==135],label='SLOPES')
plt.plot(best_combos_ORCs.WINDOW_IDX[best_combos_ORCs.TUNE_IDX==135],best_combos_ORCs.ICI[best_combos_ORCs.TUNE_IDX==135],label='ORC')
plt.plot(best_combos_ECEs.WINDOW_IDX[best_combos_ECEs.TUNE_IDX==135],best_combos_ECEs.ICI[best_combos_ECEs.TUNE_IDX==135],label='ECE')
plt.plot(best_combos_ICIs.WINDOW_IDX[best_combos_ICIs.TUNE_IDX==135],best_combos_ICIs.ICI[best_combos_ICIs.TUNE_IDX==135],label='ICI')
plt.ylabel('ICI')
plt.legend()
plt.show()

#########

plt.plot(best_combos_slopes.WINDOW_IDX[best_combos_slopes.TUNE_IDX==69],best_combos_slopes.ORC[best_combos_slopes.TUNE_IDX==69],label='SLOPES')
plt.plot(best_combos_ORCs.WINDOW_IDX[best_combos_ORCs.TUNE_IDX==69],best_combos_ORCs.ORC[best_combos_ORCs.TUNE_IDX==69],label='ORC')
plt.plot(best_combos_ECEs.WINDOW_IDX[best_combos_ECEs.TUNE_IDX==69],best_combos_ECEs.ORC[best_combos_ECEs.TUNE_IDX==69],label='ECE')
plt.plot(best_combos_ICIs.WINDOW_IDX[best_combos_ICIs.TUNE_IDX==69],best_combos_ICIs.ORC[best_combos_ICIs.TUNE_IDX==69],label='ICI')
plt.ylabel('ORC')
plt.legend()
plt.show()

plt.plot(best_combos_slopes.WINDOW_IDX[best_combos_slopes.TUNE_IDX==69],best_combos_slopes.CALIB_SLOPE[best_combos_slopes.TUNE_IDX==69],label='SLOPES')
plt.plot(best_combos_ORCs.WINDOW_IDX[best_combos_ORCs.TUNE_IDX==69],best_combos_ORCs.CALIB_SLOPE[best_combos_ORCs.TUNE_IDX==69],label='ORC')
plt.plot(best_combos_ECEs.WINDOW_IDX[best_combos_ECEs.TUNE_IDX==69],best_combos_ECEs.CALIB_SLOPE[best_combos_ECEs.TUNE_IDX==69],label='ECE')
plt.plot(best_combos_ICIs.WINDOW_IDX[best_combos_ICIs.TUNE_IDX==69],best_combos_ICIs.CALIB_SLOPE[best_combos_ICIs.TUNE_IDX==69],label='ICI')
plt.ylabel('Calibration slope')
plt.legend()
plt.show()

#################################################################################################################################################################
calibrated_preds_combos = calib_file_info_df[['TUNE_IDX','REPEAT','FOLD']].drop_duplicates(ignore_index=True)
calibrated_preds_combos = calibrated_preds_combos[calibrated_preds_combos.TUNE_IDX != 55].drop_duplicates(ignore_index=True)

for curr_combo_row_idx in tqdm(range(calibrated_preds_combos.shape[0])):
    
    curr_tune_idx = calibrated_preds_combos.TUNE_IDX[curr_combo_row_idx]
    curr_repeat = calibrated_preds_combos.REPEAT[curr_combo_row_idx]
    curr_fold = calibrated_preds_combos.FOLD[curr_combo_row_idx]
    
    tune_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold).zfill(1),'tune'+str(curr_tune_idx).zfill(4))

    if curr_tune_idx == 69:
        curr_best_combos = best_69_combos.copy()
        
    elif curr_tune_idx == 135:
        curr_best_combos = best_135_combos.copy()
        
    uncalib_val_preds = pd.read_csv(os.path.join(tune_dir,'uncalibrated_val_predictions.csv'))
    uncalib_val_preds['WindowIdx'] = uncalib_val_preds.groupby('GUPI').cumcount(ascending=True)+1
    uncalib_val_preds['REPEAT'] = curr_repeat
    uncalib_val_preds['FOLD'] = curr_fold
    calib_val_preds = []
    calib_val_preds.append(uncalib_val_preds[uncalib_val_preds.WindowIdx >= 85].reset_index(drop=True))
    none_window_indices = curr_best_combos[curr_best_combos.CALIBRATION == 'None'].WINDOW_IDX.unique()
    calib_val_preds.append(uncalib_val_preds[uncalib_val_preds.WindowIdx.isin(none_window_indices)].reset_index(drop=True))
    remaining_combos = curr_best_combos[curr_best_combos.CALIBRATION != 'None'].reset_index(drop=True)
    for curr_rem_row_idx in range(remaining_combos.shape[0]):
        curr_optimization = remaining_combos.OPTIMIZATION[curr_rem_row_idx]
        curr_calibration = remaining_combos.CALIBRATION[curr_rem_row_idx]
        curr_window_index = remaining_combos.WINDOW_IDX[curr_rem_row_idx]
        curr_calib_preds = pd.read_pickle(os.path.join(tune_dir,'set_val_opt_'+curr_optimization+'_window_idx_'+str(curr_window_index).zfill(2)+'_scaling_'+curr_calibration+'.pkl'))
        calib_val_preds.append(curr_calib_preds)
    calib_val_preds = pd.concat(calib_val_preds,ignore_index=True).sort_values(['GUPI','WindowIdx']).reset_index(drop=True)
    calib_val_preds.to_csv(os.path.join(tune_dir,'calibrated_val_predictions.csv'),index=False)
    
    uncalib_test_preds = pd.read_csv(os.path.join(tune_dir,'uncalibrated_test_predictions.csv'))
    uncalib_test_preds['WindowIdx'] = uncalib_test_preds.groupby('GUPI').cumcount(ascending=True)+1
    uncalib_test_preds['REPEAT'] = curr_repeat
    uncalib_test_preds['FOLD'] = curr_fold
    calib_test_preds = []
    calib_test_preds.append(uncalib_test_preds[uncalib_test_preds.WindowIdx >= 85].reset_index(drop=True))
    none_window_indices = curr_best_combos[curr_best_combos.CALIBRATION == 'None'].WINDOW_IDX.unique()
    calib_test_preds.append(uncalib_test_preds[uncalib_test_preds.WindowIdx.isin(none_window_indices)].reset_index(drop=True))
    remaining_combos = curr_best_combos[curr_best_combos.CALIBRATION != 'None'].reset_index(drop=True)
    for curr_rem_row_idx in range(remaining_combos.shape[0]):
        curr_optimization = remaining_combos.OPTIMIZATION[curr_rem_row_idx]
        curr_calibration = remaining_combos.CALIBRATION[curr_rem_row_idx]
        curr_window_index = remaining_combos.WINDOW_IDX[curr_rem_row_idx]
        curr_calib_preds = pd.read_pickle(os.path.join(tune_dir,'set_test_opt_'+curr_optimization+'_window_idx_'+str(curr_window_index).zfill(2)+'_scaling_'+curr_calibration+'.pkl'))
        calib_test_preds.append(curr_calib_preds)
    calib_test_preds = pd.concat(calib_test_preds,ignore_index=True).sort_values(['GUPI','WindowIdx']).reset_index(drop=True)
    calib_test_preds.to_csv(os.path.join(tune_dir,'calibrated_test_predictions.csv'),index=False)