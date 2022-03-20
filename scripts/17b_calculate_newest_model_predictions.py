#### Master Script 17b: Calculate predictions of dynamic all-predictor-based models on each set ####
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
from classes.datasets import DYN_ALL_PREDICTOR_SET
from functions.model_building import collate_batch, format_tokens, load_tune_predictions
from models.dynamic_APM import GOSE_model

# Set version code
VERSION = 'v5-0'

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
study_GUPIs = cv_splits[['GUPI','GOSE']].drop_duplicates()

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Load the optimised tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

### II. Create grid of model checkpoint files and sets
# If model checkpoint dataframe already exists, load it. Otherwise create it
if not os.path.exists(os.path.join(model_dir,'model_ckpt_grid.csv')):

    # Search for all saved model checkpoint files
    model_ckpt_files = []
    for path in Path(model_dir).rglob('*.ckpt'):
        model_ckpt_files.append(str(path.resolve()))

    model_ckpt_info_df = pd.DataFrame({'file':model_ckpt_files,
                                       'tune_idx':[int(re.search('/tune(.*)/epoch=', curr_file).group(1)) for curr_file in model_ckpt_files],
                                       'VERSION':[re.search('_outputs/(.*)/repeat', curr_file).group(1) for curr_file in model_ckpt_files],
                                       'repeat':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in model_ckpt_files],
                                       'fold':[re.search('/fold(.*)/tune', curr_file).group(1) for curr_file in model_ckpt_files],
                                       'ORC':[float(re.search('-val_ORC=(.*).ckpt', curr_file).group(1)) for curr_file in model_ckpt_files]
                                     }).sort_values(by=['tune_idx','repeat','fold','VERSION','ORC']).reset_index(drop=True)

    # If there are multiple model checkpoint files for the same tuning index-repeat-fold combination, select the one with the higest ORC
    model_ckpt_info_df = model_ckpt_info_df[model_ckpt_info_df.groupby(['tune_idx','VERSION','repeat','fold'])['ORC'].transform(max) == model_ckpt_info_df['ORC']].reset_index(drop=True)
    model_ckpt_info_df['key'] = 1

    # Create dataframe for combinations of training, validation, or testing set
    set_combos = pd.DataFrame({'set':['train','val','test'],'key':1})

    # Create combinations of prediction calculations
    model_ckpt_info_df = model_ckpt_info_df.merge(set_combos,on='key',how='outer').drop(columns='key').reset_index(drop=True)
    
    # Save combinations in model directory
    model_ckpt_info_df.to_csv(os.path.join(model_dir,'model_ckpt_grid.csv'),index=False)
    
else:
    
    # Load optimised tuning grid
    model_ckpt_info_df = pd.read_csv(os.path.join(model_dir,'model_ckpt_grid.csv'))

### III. Predict based on the set and model checkpoint combinations
# Argument-induced training functions
def main(array_task_id):
    
    # Extract current repeat, fold, and tune_idx information
    curr_ckpt_file = model_ckpt_info_df.file[array_task_id]
    curr_repeat = model_ckpt_info_df.repeat[array_task_id]
    curr_fold = model_ckpt_info_df.fold[array_task_id]
    curr_tune_idx = model_ckpt_info_df.tune_idx[array_task_id]
    curr_set = model_ckpt_info_df.set[array_task_id]
    curr_strategy = tuning_grid.STRATEGY[(tuning_grid.tune_idx == curr_tune_idx)&(tuning_grid.repeat == curr_repeat)&(tuning_grid.fold == curr_fold)].reset_index(drop=True)[0]

    # Set current tuning configuration directory
    repeat_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(2))
    fold_dir = os.path.join(repeat_dir,'fold'+str(curr_fold).zfill(int(np.log10(tuning_grid.fold.max()))+1))
    tune_dir = os.path.join(fold_dir,'tune'+str(curr_tune_idx).zfill(4))
    
    # Load current token-indexed training and testing sets
    if curr_set == 'test':
        index_set = pd.read_pickle('/home/sb2406/rds/hpc-work/tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/from_adm_strategy_'+curr_strategy+'_testing_indices.pkl')
    else:
        val_GUPIs = pd.read_csv(os.path.join(tune_dir,'val_predictions.csv')).GUPI.unique()
        index_set = pd.read_pickle('/home/sb2406/rds/hpc-work/tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/from_adm_strategy_'+curr_strategy+'_training_indices.pkl')
        if curr_set == 'train':
            index_set = index_set[~index_set.GUPI.isin(val_GUPIs)].reset_index(drop=True)
        elif curr_set == 'val':
            index_set = index_set[index_set.GUPI.isin(val_GUPIs)].reset_index(drop=True)
    
    # Add GOSE scores to current index set
    index_set = pd.merge(index_set,study_GUPIs,how='left',on='GUPI')
    
    # Convert index set to PyTorch Dataset object
    curr_Dataset = DYN_ALL_PREDICTOR_SET(index_set,'softmax')
    
    # Create PyTorch DataLoader object
    curr_DL = DataLoader(curr_Dataset,
                         batch_size=len(curr_Dataset), 
                         shuffle=False,
                         collate_fn=collate_batch)
    
    # Load model from current checkpoint file
    best_model = GOSE_model.load_from_checkpoint(curr_ckpt_file)
    best_model.eval()
    
    # Calculate and save current set probabilities
    for i, (curr_label_list, curr_idx_list, curr_bin_offsets, curr_gupi_offsets, curr_gupis) in enumerate(curr_DL):
        
        (yhat, out_gupi_offsets) = best_model(curr_idx_list, curr_bin_offsets, curr_gupi_offsets)
        curr_labels = torch.cat([curr_label_list],dim=0).cpu().numpy()
        
        curr_probs = torch.cat([F.softmax(yhat.detach())],dim=0).cpu().numpy()
        curr_preds = pd.DataFrame(curr_probs,columns=['Pr(GOSE=1)','Pr(GOSE=2/3)','Pr(GOSE=4)','Pr(GOSE=5)','Pr(GOSE=6)','Pr(GOSE=7)','Pr(GOSE=8)'])
        curr_preds['TrueLabel'] = curr_labels

        curr_preds.insert(loc=0, column='GUPI', value=curr_gupis)        
        curr_preds['tune_idx'] = curr_tune_idx

        curr_preds.to_csv(os.path.join(tune_dir,curr_set+'_predictions.csv'),index=False)
        
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)