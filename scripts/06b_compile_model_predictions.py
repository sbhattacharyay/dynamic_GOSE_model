#### Master Script 6b: Compile predictions of dynamic all-predictor-based models ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile and save validation and testing set predictions across partitions

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
from shutil import rmtree
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

# TQDM for progress tracking
from tqdm import tqdm

# Custom methods
from functions.model_building import load_tune_predictions

# Set version code
VERSION = 'v2-0'

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
study_GUPI_GOSE = cv_splits[['GUPI','GOSE']].drop_duplicates()

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Load the optimised tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

### II. Compile and save validation and testing set predictions across partitions
# Search for all prediction files
pred_files = []
for path in Path(model_dir).rglob('*_predictions.csv'):
    pred_files.append(str(path.resolve()))

# Characterise the prediction files found
pred_file_info_df = pd.DataFrame({'file':pred_files,
                                  'adm_or_disch':[re.search('fold(.*)/tune', curr_file).group(1) for curr_file in pred_files],
                                  'tune_idx':[int(re.search('/tune(.*)/', curr_file).group(1)) for curr_file in pred_files],
                                  'VERSION':[re.search('_outputs/(.*)/repeat', curr_file).group(1) for curr_file in pred_files],
                                  'repeat':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in pred_files],
                                  'fold':[re.search('/fold(.*)/', curr_file).group(1) for curr_file in pred_files],
                                  'test_or_val':[re.search('/fold(.*)_predictions', curr_file).group(1) for curr_file in pred_files]
                                 }).sort_values(by=['repeat','fold','adm_or_disch','tune_idx','VERSION']).reset_index(drop=True)
pred_file_info_df['fold'] = pred_file_info_df['fold'].str.replace('\\/.*','',1,regex=True).astype(int)
pred_file_info_df['adm_or_disch'] = pred_file_info_df['adm_or_disch'].str.slice(start=2)
pred_file_info_df['test_or_val'] = pred_file_info_df['test_or_val'].str.rsplit(pat='/', n=1).apply(lambda x: x[1])
pred_file_info_df['OUTPUT_ACTIVATION'] = 'softmax'

# Separate prediction files by adm/disch and testing vs. validation
adm_val_info_df = pred_file_info_df[(pred_file_info_df.adm_or_disch == 'adm') & (pred_file_info_df.test_or_val == 'val')].reset_index(drop=True)
adm_test_info_df = pred_file_info_df[(pred_file_info_df.adm_or_disch == 'adm') & (pred_file_info_df.test_or_val == 'test')].reset_index(drop=True)

disch_val_info_df = pred_file_info_df[(pred_file_info_df.adm_or_disch == 'disch') & (pred_file_info_df.test_or_val == 'val')].reset_index(drop=True)
disch_test_info_df = pred_file_info_df[(pred_file_info_df.adm_or_disch == 'disch') & (pred_file_info_df.test_or_val == 'test')].reset_index(drop=True)

# Compile predictions into single dataframes
adm_val_preds = load_tune_predictions(adm_val_info_df, progress_bar_desc='Compiling validation set predictions from admission')
adm_test_preds = load_tune_predictions(adm_test_info_df, progress_bar_desc='Compiling testing set predictions from admission')

disch_val_preds = load_tune_predictions(disch_val_info_df, progress_bar_desc='Compiling validation set predictions from discharge')
disch_test_preds = load_tune_predictions(disch_test_info_df, progress_bar_desc='Compiling testing set predictions from discharge')

# Save prediction files appropriately
adm_val_preds.to_csv(os.path.join(model_dir,'compiled_val_predictions_from_adm.csv'),index=True)
adm_test_preds.to_csv(os.path.join(model_dir,'compiled_test_predictions_from_adm.csv'),index=True)

disch_val_preds.to_csv(os.path.join(model_dir,'compiled_val_predictions_from_disch.csv'),index=True)
disch_test_preds.to_csv(os.path.join(model_dir,'compiled_test_predictions_from_disch.csv'),index=True)