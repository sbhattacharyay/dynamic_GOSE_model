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
                                  }).reset_index(drop=True)

# Load calibration metric files
calibration_metrics_df = pd.concat([pd.read_pickle(f) for f in tqdm(calib_file_info_df.file)],ignore_index=True)

# Average metrics over folds
ave_cal_metrics = calibration_metrics_df.groupby(['TUNE_IDX','OPTIMIZATION','WINDOW_IDX','THRESHOLD','SET','CALIBRATION','METRIC'],as_index=False)['VALUE'].mean()

# Observe average calibration slopes
ave_cal_slopes = ave_cal_metrics[(ave_cal_metrics.METRIC == 'CALIB_SLOPE')&(ave_cal_metrics.THRESHOLD == 'Average')&(ave_cal_metrics.SET == 'test')].reset_index(drop=True)

# Measure abs_diff from 1
ave_cal_slopes['CALIB_SLOPE_ERROR'] = (ave_cal_slopes['VALUE']-1).abs()

# Extract best calibrated configurations for each tuning index and winodow index combination
best_combos_slopes = ave_cal_slopes.loc[ave_cal_slopes.groupby(['TUNE_IDX','WINDOW_IDX']).CALIB_SLOPE_ERROR.idxmin()].reset_index(drop=True)

best_combos_slopes[best_combos_slopes.TUNE_IDX == 135].VALUE