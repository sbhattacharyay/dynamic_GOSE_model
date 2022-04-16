#### Master Script 24a: Calculate LBM for dynAPM_DeepMN ####
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

# Custom methods
from models.dynamic_APM import GOSE_model

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

# Define a directory for the storage of model interpretation values
interp_dir = '/home/sb2406/rds/hpc-work/model_interpretations/'+VERSION
os.makedirs(interp_dir,exist_ok=True)

# Define a directory for the storage of LBM values
relevance_dir = os.path.join(interp_dir,'relevance_layer')
os.makedirs(relevance_dir,exist_ok=True)

### II. Find all top-performing model checkpoint files for SHAP calculation
# Either create or load APM checkpoint information for SHAP value 
if not os.path.exists(os.path.join(relevance_dir,'ckpt_info.pkl')):
    
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
    
    # Create combinations for each possible testing set GUPI
    ckpt_info = ckpt_info.merge(test_splits,how='left',on=['REPEAT','FOLD'])
    
    # Sort checkpoint dataframe by GUPI, then REPEAT, FOLD
    ckpt_info = ckpt_info.sort_values(by=['GUPI','REPEAT','FOLD','TUNE_IDX','VERSION']).reset_index(drop=True)
    
    # Save model checkpoint information dataframe
    ckpt_info.to_pickle(os.path.join(relevance_dir,'ckpt_info.pkl'))
    
else:
    
    # Read model checkpoint information dataframe
    ckpt_info = pd.read_pickle(os.path.join(relevance_dir,'ckpt_info.pkl'))

    
### Work while RDS is cramped
ckpt_info = pd.read_pickle(os.path.join(os.path.join(interp_dir,'LBM'),'ckpt_info.pkl'))
ckpt_info = ckpt_info[['file','TUNE_IDX','VERSION','REPEAT','FOLD']].drop_duplicates().reset_index(drop=True)

compiled_relevance_layers = []
for array_task_id in tqdm(range(ckpt_info.shape[0])):
    # Extract current file, repeat, and fold information
    curr_file = ckpt_info.file[array_task_id]
    curr_tune_idx = ckpt_info.TUNE_IDX[array_task_id]
    curr_repeat = ckpt_info.REPEAT[array_task_id]
    curr_fold = ckpt_info.FOLD[array_task_id]
    
    # Define current fold directory based on current information
    tune_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold),'tune'+str(curr_tune_idx).zfill(4))
    
    # Filter out current tuning directory configuration hyperparameters
    curr_tune_hp = tuning_grid[(tuning_grid.TUNE_IDX == curr_tune_idx)&(tuning_grid.fold == curr_fold)].reset_index(drop=True)
    
    # Load current token dictionary
    token_dir = os.path.join('/home/sb2406/rds/hpc-work/tokens','repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold))
    curr_vocab = cp.load(open(os.path.join(token_dir,'from_adm_strategy_'+curr_tune_hp.STRATEGY[0]+'_token_dictionary.pkl'),"rb"))
    unknown_index = curr_vocab['<unk>']
    
    # Load current pretrained model
    gose_model = GOSE_model.load_from_checkpoint(curr_file)
    gose_model.eval()
    
    # Extract relevance layer values
    with torch.no_grad():
        relevance_layer = torch.exp(gose_model.embedW.weight.detach().squeeze(1)).numpy()
        token_labels = curr_vocab.get_itos()        
        curr_relevance_df = pd.DataFrame({'TUNE_IDX':curr_tune_idx,
                                          'TOKEN':token_labels,
                                          'RELEVANCE':relevance_layer,
                                          'REPEAT':curr_repeat,
                                          'FOLD':curr_fold})
    compiled_relevance_layers.append(curr_relevance_df)
    
compiled_relevance_layers = pd.concat(compiled_relevance_layers, ignore_index=True)
agg_relevance_layers = compiled_relevance_layers.groupby(['TUNE_IDX','TOKEN'],as_index=False)['RELEVANCE'].aggregate({'mean':np.mean,'std':np.std,'median':np.median,'min':np.min,'max':np.max,'Q1':lambda x: np.quantile(x,.25),'Q3':lambda x: np.quantile(x,.75),'resamples':'count'}).reset_index(drop=True)

### CHARACTERISE TOKENS
# Determine whether tokens are baseline
agg_relevance_layers['Baseline'] = agg_relevance_layers['TOKEN'].str.startswith('Baseline')

# Determine whether tokens are numeric
agg_relevance_layers['Numeric'] = agg_relevance_layers['TOKEN'].str.contains('_BIN')

# Determine wheter tokens represent missing values
agg_relevance_layers['Missing'] = ((agg_relevance_layers.Numeric)&(agg_relevance_layers['TOKEN'].str.endswith('_BIN_missing')))|((~agg_relevance_layers.Numeric)&(agg_relevance_layers['TOKEN'].str.endswith('_NA')))

# Create empty column for predictor base token
agg_relevance_layers['BaseToken'] = ''

# For numeric tokens, extract the portion of the string before '_BIN' as the BaseToken
agg_relevance_layers.BaseToken[agg_relevance_layers.Numeric] = agg_relevance_layers.TOKEN[agg_relevance_layers.Numeric].str.replace('\\_BIN.*','',1,regex=True)

# For non-numeric tokens, extract everything before the final underscore, if one exists, as the BaseToken
agg_relevance_layers.BaseToken[~agg_relevance_layers.Numeric] = agg_relevance_layers.TOKEN[~agg_relevance_layers.Numeric].str.replace('_[^_]*$','',1,regex=True)

# For baseline tokens, remove the "Baseline" prefix in the BaseToken
agg_relevance_layers.BaseToken[agg_relevance_layers.Baseline] = agg_relevance_layers.BaseToken[agg_relevance_layers.Baseline].str.replace('Baseline','',1,regex=False)

### Load dictionary
old_token_dictionary = pd.read_csv('/home/sb2406/rds/hpc-work/tokens/old_token_dictionary.csv').drop(columns=['Token','count','UnderscoreCount','Baseline','Numeric']).drop_duplicates(ignore_index=True)

# Merge old token dictionary with vocab dataframes
agg_relevance_layers = agg_relevance_layers.merge(old_token_dictionary,how='left',on=['BaseToken'])

# Add 'TimeOfDay' token information to vocab dataframes
agg_relevance_layers.ICUIntervention[agg_relevance_layers.BaseToken=='TimeOfDay'] = False
agg_relevance_layers.ClinicianInput[agg_relevance_layers.BaseToken=='TimeOfDay'] = False
agg_relevance_layers.Type[agg_relevance_layers.BaseToken=='TimeOfDay'] = 'Time of day'

agg_relevance_layers.ICUIntervention[agg_relevance_layers.BaseToken=='TimeFromAdm'] = False
agg_relevance_layers.ClinicianInput[agg_relevance_layers.BaseToken=='TimeFromAdm'] = False
agg_relevance_layers.Type[agg_relevance_layers.BaseToken=='TimeFromAdm'] = 'Time from admission'

# Remove blank token from aggregated relevance layer values
agg_relevance_layers = agg_relevance_layers[~(agg_relevance_layers.TOKEN == '')].reset_index(drop=True)

# Temporarily save aggregated relevance layer values in current directory
agg_relevance_layers.to_csv('aggregated_relevances.csv',index=False)