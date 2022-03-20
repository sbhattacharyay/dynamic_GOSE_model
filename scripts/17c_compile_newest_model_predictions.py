#### Master Script 17b: Compile predictions of dynamic all-predictor-based models ####
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
VERSION = 'v5-0'

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
study_GUPI_GOSE = cv_splits[['GUPI','GOSE']].drop_duplicates()

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Load the optimised tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

### II. Compile and save training, validation and testing set predictions across partitions
# Search for all prediction files
pred_files = []
for path in Path(model_dir).rglob('*_predictions.csv'):
    pred_files.append(str(path.resolve()))

# Characterise the prediction files found
pred_file_info_df = pd.DataFrame({'file':pred_files,
                                  'tune_idx':[int(re.search('/tune(.*)/', curr_file).group(1)) for curr_file in pred_files],
                                  'VERSION':[re.search('_outputs/(.*)/repeat', curr_file).group(1) for curr_file in pred_files],
                                  'repeat':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in pred_files],
                                  'fold':[re.search('/fold(.*)/', curr_file).group(1) for curr_file in pred_files],
                                  'set':[re.search('/fold(.*)_predictions', curr_file).group(1) for curr_file in pred_files]
                                 }).sort_values(by=['repeat','fold','tune_idx','VERSION']).reset_index(drop=True)
pred_file_info_df['fold'] = pred_file_info_df['fold'].str.replace('\\/.*','',1,regex=True).astype(int)
pred_file_info_df['set'] = pred_file_info_df['set'].str.rsplit(pat='/', n=1).apply(lambda x: x[1])
pred_file_info_df['OUTPUT_ACTIVATION'] = 'softmax'
pred_file_info_df['adm_or_disch'] = 'adm'

# Separate prediction files by adm/disch and testing vs. validation
train_info_df = pred_file_info_df[pred_file_info_df.set == 'train'].reset_index(drop=True)
val_info_df = pred_file_info_df[pred_file_info_df.set == 'val'].reset_index(drop=True)
test_info_df = pred_file_info_df[pred_file_info_df.set == 'test'].reset_index(drop=True)

# Compile predictions into single dataframes
train_preds = load_tune_predictions(train_info_df, progress_bar_desc='Compiling training set predictions from admission')
val_preds = load_tune_predictions(val_info_df, progress_bar_desc='Compiling validation set predictions from admission')
test_preds = load_tune_predictions(test_info_df, progress_bar_desc='Compiling testing set predictions from admission')

# Save prediction files appropriately
train_preds.to_csv(os.path.join(model_dir,'compiled_train_predictions_from_adm.csv'),index=True)
val_preds.to_csv(os.path.join(model_dir,'compiled_val_predictions_from_adm.csv'),index=True)
test_preds.to_csv(os.path.join(model_dir,'compiled_test_predictions_from_adm.csv'),index=True)