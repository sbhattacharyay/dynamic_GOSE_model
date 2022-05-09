#### Master Script 25b: Calculate pruning statistics for TimeSHAP for dynAPM_DeepMN ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Find all top-performing model checkpoint files for SHAP calculation
# III. Calculate SHAP values based on given parameters

### I. Initialisation
# Fundamental libraries
import os
import re
import sys
import time
import glob
import copy
import math
import random
import datetime
import warnings
import operator
import itertools
import functools
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
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# PyTorch, PyTorch.Text, and Lightning-PyTorch methods
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# SciKit-Learn methods
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample, shuffle
from sklearn.utils.class_weight import compute_class_weight

# TQDM for progress tracking
from tqdm import tqdm

# Import TimeSHAP methods
import timeshap.explainer as tsx
import timeshap.plot as tsp
from timeshap.wrappers import TorchModelWrapper
from timeshap.utils import get_avg_score_with_avg_event

# Custom methods
from classes.datasets import DYN_ALL_PREDICTOR_SET
from models.dynamic_APM import GOSE_model, timeshap_GOSE_model
from functions.model_building import format_shap, format_tokens, format_time_tokens, df_to_multihot_matrix

# Import TimeSHAP-specific themes
# import altair as alt
# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))
# from timeshap.plot import timeshap_theme
# alt.themes.register("timeshap_theme", timeshap_theme)
# alt.themes.enable("timeshap_theme")

# Set version code
VERSION = 'v6-0'

# Define number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Load the current version tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

# Load cross-validation split information to extract testing resamples
cv_splits = pd.read_csv('../cross_validation_splits.csv')
test_splits = cv_splits[cv_splits.set == 'test'].rename(columns={'repeat':'REPEAT','fold':'FOLD','set':'SET'}).reset_index(drop=True)
uniq_GUPIs = test_splits.GUPI.unique()
uniq_GOSEs = np.sort(cv_splits.GOSE.unique())

# Define a directory for the storage of SHAP values
shap_dir = os.path.join('/home/sb2406/rds/hpc-work/model_interpretations/'+VERSION,'TimeSHAP')
os.makedirs(shap_dir,exist_ok=True)

# Define vector of GOSE thresholds
GOSE_thresholds = ['GOSE<2','GOSE<4','GOSE<5','GOSE<6','GOSE<7','GOSE<8']

### II. Find all top-performing model checkpoint files for SHAP calculation
# Find all pruning indices
prun_idx_files = []
for path in Path(shap_dir).rglob('prun_idx_thresh_idx_*'):
    prun_idx_files.append(str(path.resolve()))

# Categorize model checkpoint files based on name
prun_idx_info = pd.DataFrame({'file':prun_idx_files,
                              'VERSION':[re.search('model_interpretations/(.*)/TimeSHAP', curr_file).group(1) for curr_file in prun_idx_files],
                              'REPEAT':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in prun_idx_files],
                              'FOLD':[int(re.search('/fold(.*)/prun_idx_', curr_file).group(1)) for curr_file in prun_idx_files],
                              'THRESHOLD_IDX':[int(re.search('/prun_idx_thresh_idx_(.*).pkl', curr_file).group(1)) for curr_file in prun_idx_files]
                             }).sort_values(by=['REPEAT','FOLD','THRESHOLD_IDX']).reset_index(drop=True)

# Load all LBM dataframes in parallel 
s = [prun_idx_info.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
s[:(prun_idx_info.shape[0] - sum(s))] = [over+1 for over in s[:(prun_idx_info.shape[0] - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
