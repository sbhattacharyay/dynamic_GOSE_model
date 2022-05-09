#### Master Script 25a: Compute pruning characteristics for TimeSHAP for dynAPM_DeepMN ####
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
# Either create or load APM checkpoint information for SHAP value 
if not os.path.exists(os.path.join(shap_dir,'pruning_ckpt_info.pkl')):
    
    # Find all model checkpoint files in APM output directory
    ckpt_files = []
    for path in Path(model_dir).rglob('*.ckpt'):
        ckpt_files.append(str(path.resolve()))

    # Categorize model checkpoint files based on name
    ckpt_info = pd.DataFrame({'file':ckpt_files,
                              'TUNE_IDX':[int(re.search('tune(.*)/epoch=', curr_file).group(1)) for curr_file in ckpt_files],
                              'VERSION':[re.search('model_outputs/(.*)/repeat', curr_file).group(1) for curr_file in ckpt_files],
                              'REPEAT':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in ckpt_files],
                              'FOLD':[int(re.search('/fold(.*)/tune', curr_file).group(1)) for curr_file in ckpt_files],
                              'VAL_LOSS':[float(re.search('val_loss=(.*).ckpt', curr_file).group(1)) for curr_file in ckpt_files]
                             }).sort_values(by=['REPEAT','FOLD','TUNE_IDX','VERSION']).reset_index(drop=True)
    ckpt_info = ckpt_info[ckpt_info.TUNE_IDX == 135].reset_index(drop=True)
    
    # For each partition, select the file that minimizes validation set loss
    ckpt_info = ckpt_info.loc[ckpt_info.groupby(['TUNE_IDX','VERSION','REPEAT','FOLD'])['VAL_LOSS'].idxmin()].reset_index(drop=True)
    
    # Fix analysis to FIRST REPEAT
    ckpt_info = ckpt_info[ckpt_info.REPEAT == 1].reset_index(drop=True)
    ckpt_info['key'] = 1
    
    # Add threshold combinations to checkpoint info dataframe
    threshold_df = pd.DataFrame({'THRESHOLD_IDX':list(range(6)),'key':1})
    ckpt_info = ckpt_info.merge(threshold_df,how='left',on=['key']).drop(columns='key')
    
    # Sort checkpoint dataframe by GUPI, then REPEAT, FOLD
    ckpt_info = ckpt_info.sort_values(by=['REPEAT','FOLD','THRESHOLD_IDX','TUNE_IDX','VERSION']).reset_index(drop=True)
    
    # Save model checkpoint information dataframe
    ckpt_info.to_pickle(os.path.join(shap_dir,'pruning_ckpt_info.pkl'))
    
else:
    
    # Read model checkpoint information dataframe
    ckpt_info = pd.read_pickle(os.path.join(shap_dir,'pruning_ckpt_info.pkl'))

### III. Calculate TimeSHAP pruning scores based on given parameters
def main(array_task_id):
    
    # Extract current file, repeat, and fold information
    curr_file = ckpt_info.file[array_task_id]
    curr_tune_idx = ckpt_info.TUNE_IDX[array_task_id]
    curr_repeat = ckpt_info.REPEAT[array_task_id]
    curr_fold = ckpt_info.FOLD[array_task_id]
    curr_threshold_idx = ckpt_info.THRESHOLD_IDX[array_task_id]
    
    # Define current fold directory based on current information
    tune_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold),'tune'+str(curr_tune_idx).zfill(4))

    # Filter out current tuning directory configuration hyperparameters
    curr_tune_hp = tuning_grid[(tuning_grid.TUNE_IDX == curr_tune_idx)&(tuning_grid.fold == curr_fold)].reset_index(drop=True)
    
    # Based on current threshold_idx, identify threshold and positive GUPIs
    curr_threshold_GOSE = GOSE_thresholds[curr_threshold_idx]
    pos_GOSEs = uniq_GOSEs[:(curr_threshold_idx+1)]
    
    # Refine current testing set split based on repeat, fold, and positive GOSEs
    curr_testing_split = test_splits[(test_splits.REPEAT == curr_repeat)&(test_splits.FOLD == curr_fold)&(test_splits.GOSE.isin(pos_GOSEs))].reset_index(drop=True)
    
    # Extract current training and testing sets for current repeat and fold combination
    token_dir = os.path.join('/home/sb2406/rds/hpc-work/tokens','repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold))
    training_set = pd.read_pickle(os.path.join(token_dir,'from_adm_strategy_'+curr_tune_hp.STRATEGY[0]+'_training_indices.pkl'))
    testing_set = pd.read_pickle(os.path.join(token_dir,'from_adm_strategy_'+curr_tune_hp.STRATEGY[0]+'_testing_indices.pkl'))
    
    # Filter out testing set in current "positve" set
    testing_set = testing_set[testing_set.GUPI.isin(curr_testing_split.GUPI)].reset_index(drop=True)
    
    # Load current token dictionary
    curr_vocab = cp.load(open(os.path.join(token_dir,'from_adm_strategy_'+curr_tune_hp.STRATEGY[0]+'_token_dictionary.pkl'),"rb"))
    unknown_index = curr_vocab['<unk>']
    
    # Format time tokens of index sets based on current tuning configuration
    training_set,time_tokens_mask = format_time_tokens(training_set,curr_tune_hp.TIME_TOKENS[0],True)
    training_set['DischWindowIdx'] = -(training_set['WindowTotal'] - training_set['WindowIdx'] + 1)
    testing_set,_ = format_time_tokens(testing_set,curr_tune_hp.TIME_TOKENS[0],False)
    testing_set['SeqLength'] = testing_set.VocabIndex.apply(len)
    testing_set['Unknowns'] = testing_set.VocabIndex.apply(lambda x: x.count(unknown_index))    
    testing_set['DischWindowIdx'] = -(testing_set['WindowTotal'] - testing_set['WindowIdx'] + 1)
    
    # Number of columns to add
    cols_to_add = max(testing_set['Unknowns'].max(),1) - 1
    
    # Define token labels from current vocab
    token_labels = curr_vocab.get_itos() + [curr_vocab.get_itos()[unknown_index]+'_'+str(i+1).zfill(3) for i in range(cols_to_add)]
    token_labels[unknown_index] = token_labels[unknown_index]+'_000'
    
    # Convert testing set dataframes to multihot matrix
    testing_multihot = df_to_multihot_matrix(testing_set, len(curr_vocab), unknown_index, cols_to_add)
    testing_multihot_df = pd.DataFrame(testing_multihot.numpy(),columns=token_labels)
    testing_multihot_df.insert(0,'GUPI',testing_set.GUPI)
    testing_multihot_df.insert(1,'WindowIdx',testing_set.WindowIdx)

    # Calculate baseline ('average') values based on training set
    flattened_training_set = training_set.groupby(['GUPI','WindowTotal'],as_index=False).VocabIndex.aggregate(list)
    flattened_training_set['IndexCounts'] = flattened_training_set.VocabIndex.apply(lambda x: [item for sublist in x for item in sublist]).apply(lambda x: dict(Counter(x)))
    flattened_training_set['IndexCounts'] = flattened_training_set.apply(lambda x: {k: v / x.WindowTotal for k, v in x.IndexCounts.items()}, axis=1)
    IndexCounts = dict(functools.reduce(operator.add,map(Counter, flattened_training_set['IndexCounts'].to_list())))
    IndexCounts = {k: v/flattened_training_set.shape[0] for k, v in IndexCounts.items() if (v/flattened_training_set.shape[0])>.5}
    BaselineIndices = np.sort(list(IndexCounts.keys()))
    AverageEvent = np.zeros([1,len(curr_vocab)+cols_to_add])
    AverageEvent[0,BaselineIndices] = 1
    AverageEvent = pd.DataFrame(AverageEvent,columns=token_labels).astype(int)
    
    # Load current pretrained model
    gose_model = GOSE_model.load_from_checkpoint(curr_file)
    gose_model.eval()

    # Initialize custom TimeSHAP model
    ts_GOSE_model = timeshap_GOSE_model(gose_model,curr_threshold_idx,unknown_index,cols_to_add)
    wrapped_gose_model = TorchModelWrapper(ts_GOSE_model)
    f_hs = lambda x, y=None: wrapped_gose_model.predict_last_hs(x, y)

    # Define pruning parameters
    pruning_dict = {'tol': [0.00625, 0.0125, 0.025, 0.05], 'path': os.path.join(shap_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold),'prun_all_thresh_idx_'+str(curr_threshold_idx)+'.csv')}
    prun_indexes = tsx.prune_all(f_hs, testing_multihot_df, 'GUPI', AverageEvent, pruning_dict, token_labels, 'WindowIdx')
    prun_indexes.to_pickle(os.path.join(shap_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold),'prun_idx_thresh_idx_'+str(curr_threshold_idx)+'.pkl'))
    
if __name__ == '__main__':

    array_task_id = int(sys.argv[1])    
    main(array_task_id)