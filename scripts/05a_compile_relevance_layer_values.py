#### Master Script 05a: Extract relevance layer values from trained dynAPM models ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Find all top-performing model checkpoint files for relevance layer extraction
# III. Iteratively extract relevance layers from top-performing tuning configurations
# IV. Characterise and summarise relevance layer tokens based on `BaseToken`

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
from tqdm import tqdm
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

# Custom methods
from models.dynamic_APM import GOSE_model

# Set version code
VERSION = 'v7-0'

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Load the current version tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

# Load cross-validation split information to extract testing resamples
cv_splits = pd.read_csv('../cross_validation_splits.csv')
test_splits = cv_splits[cv_splits.SET == 'test'].reset_index(drop=True)
uniq_GUPIs = test_splits.GUPI.unique()

# Define a directory for the storage of model interpretation values
interp_dir = '/home/sb2406/rds/hpc-work/model_interpretations/'+VERSION
os.makedirs(interp_dir,exist_ok=True)

# Define a directory for the storage of relevance layer values
relevance_dir = os.path.join(interp_dir,'relevance_layer')
os.makedirs(relevance_dir,exist_ok=True)

### II. Find all top-performing model checkpoint files for relevance layer extraction
# Either create or load dynAPM checkpoint information for relevance layer extraction
if not os.path.exists(os.path.join(relevance_dir,'ckpt_info.pkl')):
    
    # Find all model checkpoint files in APM output directory
    ckpt_files = []
    for path in Path(model_dir).rglob('*.ckpt'):
        ckpt_files.append(str(path.resolve()))

    # Categorize model checkpoint files based on name
    ckpt_info = pd.DataFrame({'file':ckpt_files,
                              'TUNE_IDX':[int(re.search('tune(.*)/epoch=', curr_file).group(1)) for curr_file in ckpt_files],
                              'VERSION':[re.search('model_outputs/(.*)/fold', curr_file).group(1) for curr_file in ckpt_files],
                              'FOLD':[int(re.search('/fold(.*)/tune', curr_file).group(1)) for curr_file in ckpt_files],
                              'VAL_ORC':[re.search('val_ORC=(.*).ckpt', curr_file).group(1) for curr_file in ckpt_files]
                             }).sort_values(by=['FOLD','TUNE_IDX','VERSION']).reset_index(drop=True)
    ckpt_info.VAL_ORC = ckpt_info.VAL_ORC.str.split('-').str[0].astype(float)
    
    # Save model checkpoint information dataframe
    ckpt_info.to_pickle(os.path.join(relevance_dir,'ckpt_info.pkl'))
    
else:
    
    # Read model checkpoint information dataframe
    ckpt_info = pd.read_pickle(os.path.join(relevance_dir,'ckpt_info.pkl'))

### III. Iteratively extract relevance layers from top-performing tuning configurations
## Initiate empty list to store compiled relevance layers
compiled_relevance_layers = []

## Iterate through identified checkpoints to aggregate relevance layers
for ckpt_row in tqdm(range(ckpt_info.shape[0]),'Iterating through checkpoints to extract relevance layers'):
    
    # Extract current file, tune index, and fold information
    curr_file = ckpt_info.file[ckpt_row]
    curr_tune_idx = ckpt_info.TUNE_IDX[ckpt_row]
    curr_fold = ckpt_info.FOLD[ckpt_row]
    
    # Define current fold directory based on current information
    tune_dir = os.path.join(model_dir,'fold'+str(curr_fold),'tune'+str(curr_tune_idx).zfill(4))
    
    # Filter out current tuning directory configuration hyperparameters
    curr_tune_hp = tuning_grid[(tuning_grid.TUNE_IDX == curr_tune_idx)&(tuning_grid.FOLD == curr_fold)].reset_index(drop=True)
    
    # Load current token dictionary
    token_dir = os.path.join('/home/sb2406/rds/hpc-work/tokens','fold'+str(curr_fold))
    curr_vocab = cp.load(open(os.path.join(token_dir,'token_dictionary.pkl'),"rb"))
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
                                          'FOLD':curr_fold})
    compiled_relevance_layers.append(curr_relevance_df)

## Concatenate all extracted relevance layers and calculate summary statistics for each `TUNE_IDX`-`TOKEN` combination
compiled_relevance_layers = pd.concat(compiled_relevance_layers, ignore_index=True)
agg_relevance_layers = compiled_relevance_layers.groupby(['TUNE_IDX','TOKEN'],as_index=False)['RELEVANCE'].aggregate({'mean':np.mean,'std':np.std,'median':np.median,'min':np.min,'max':np.max,'Q1':lambda x: np.quantile(x,.25),'Q3':lambda x: np.quantile(x,.75),'resamples':'count'}).reset_index(drop=True)

### IV. Characterise and summarise relevance layer tokens based on `BaseToken`
## First, extract the `BaseToken` from the `TOKEN` value
# Parse out `BaseToken` and `Value` from `Token`
agg_relevance_layers['BaseToken'] = agg_relevance_layers.TOKEN.str.split('_').str[0]
agg_relevance_layers['Value'] = agg_relevance_layers.TOKEN.str.split('_',n=1).str[1].fillna('')

# Determine wheter tokens represent missing values
agg_relevance_layers['Missing'] = agg_relevance_layers.TOKEN.str.endswith('_NAN')

# Determine whether tokens are numeric
agg_relevance_layers['Numeric'] = agg_relevance_layers.TOKEN.str.contains('_BIN')

# Determine whether tokens are baseline or discharge
agg_relevance_layers['Baseline'] = agg_relevance_layers.TOKEN.str.startswith('Baseline')
agg_relevance_layers['Discharge'] = agg_relevance_layers.TOKEN.str.startswith('Discharge')

# For baseline and discharge tokens, remove prefix from `BaseToken` entry
agg_relevance_layers.BaseToken[agg_relevance_layers.Baseline] = agg_relevance_layers.BaseToken[agg_relevance_layers.Baseline].str.replace('Baseline','',1,regex=False)
agg_relevance_layers.BaseToken[agg_relevance_layers.Discharge] = agg_relevance_layers.BaseToken[agg_relevance_layers.Discharge].str.replace('Discharge','',1,regex=False)

## Characterise tokens based on matches in manually constructed `BaseToken` directory
# Load manually corrected `BaseToken` categorization key
base_token_key = pd.read_excel('/home/sb2406/rds/hpc-work/tokens/base_token_keys.xlsx')
base_token_key['BaseToken'] = base_token_key['BaseToken'].fillna('')

# Merge base token key information to aggregated relevance layer dataframeßß
agg_relevance_layers = agg_relevance_layers.merge(base_token_key,how='left',on=['BaseToken','Numeric','Baseline','Discharge'])

# Remove blank token from aggregated relevance layer values
agg_relevance_layers = agg_relevance_layers[~(agg_relevance_layers.TOKEN == '')].reset_index(drop=True)

# Save aggregated relevance layer values in the current relevance directory
agg_relevance_layers.to_csv(os.path.join(relevance_dir,'agg_relevance_layers.csv'),index=False)

## Summarise token relevances for plotting
# Load aggregated relevance layer values from the current relevance directory
agg_relevance_layers = pd.read_csv(os.path.join(relevance_dir,'agg_relevance_layers.csv'))

# Remove missing token values
nonmissing_agg_relevance_layers = agg_relevance_layers[~agg_relevance_layers.Missing].reset_index(drop=True)

# Take only maximum (nonmissing) token values per predictor
predictor_relevances = nonmissing_agg_relevance_layers.loc[nonmissing_agg_relevance_layers.groupby(['TUNE_IDX','BaseToken'])['median'].idxmax()].sort_values(by='median',ascending=False).reset_index(drop=True)

# Identify top 20 and bottom 3 baseline predictors per tuning index
baseline_relevance_layers = predictor_relevances[predictor_relevances.Baseline].sort_values(by=['TUNE_IDX','median'],ascending=[True,False]).reset_index(drop=True)
specific_baseline_relevances = pd.concat([baseline_relevance_layers.groupby('TUNE_IDX').head(20),baseline_relevance_layers.groupby('TUNE_IDX').tail(3)],ignore_index=True)

# For baseline predictors not in the top 20 or bottom 3, calculate summary statistics
unspecific_baseline_relevances = baseline_relevance_layers.merge(specific_baseline_relevances,how='left', indicator=True)
unspecific_baseline_relevances = unspecific_baseline_relevances[unspecific_baseline_relevances._merge=='left_only'].drop(columns='_merge').reset_index(drop=True)[['TUNE_IDX','TOKEN']]
unspecific_baseline_relevances = compiled_relevance_layers.merge(unspecific_baseline_relevances,how='inner').groupby(['TUNE_IDX'],as_index=False)['RELEVANCE'].aggregate({'mean':np.mean,'std':np.std,'median':np.median,'min':np.min,'max':np.max,'Q1':lambda x: np.quantile(x,.25),'Q3':lambda x: np.quantile(x,.75),'resamples':'count'}).reset_index(drop=True)

# Concatenate baseline predictor relevances and save
plot_df_baseline_predictors = pd.concat([specific_baseline_relevances,unspecific_baseline_relevances],ignore_index=True).sort_values(by=['TUNE_IDX','median'],ascending=[True,False]).reset_index(drop=True)
plot_df_baseline_predictors.TOKEN[plot_df_baseline_predictors.TOKEN.isna()] = 'Other'
plot_df_baseline_predictors.BaseToken[plot_df_baseline_predictors.BaseToken.isna()] = 'Other'
plot_df_baseline_predictors.to_csv(os.path.join(relevance_dir,'baseline_relevances_plot_df.csv'),index=False)

# Identify top 20 and bottom 3 dynamic predictors per tuning index
dynamic_relevance_layers = predictor_relevances[~predictor_relevances.Baseline].sort_values(by=['TUNE_IDX','median'],ascending=[True,False]).reset_index(drop=True)
specific_dynamic_relevances = pd.concat([dynamic_relevance_layers.groupby('TUNE_IDX').head(20),dynamic_relevance_layers.groupby('TUNE_IDX').tail(3)],ignore_index=True)

# For dynamic predictors not in the top 20 or bottom 3, calculate summary statistics
unspecific_dynamic_relevances = dynamic_relevance_layers.merge(specific_dynamic_relevances,how='left', indicator=True)
unspecific_dynamic_relevances = unspecific_dynamic_relevances[unspecific_dynamic_relevances._merge=='left_only'].drop(columns='_merge').reset_index(drop=True)[['TUNE_IDX','TOKEN']]
unspecific_dynamic_relevances = compiled_relevance_layers.merge(unspecific_dynamic_relevances,how='inner').groupby(['TUNE_IDX'],as_index=False)['RELEVANCE'].aggregate({'mean':np.mean,'std':np.std,'median':np.median,'min':np.min,'max':np.max,'Q1':lambda x: np.quantile(x,.25),'Q3':lambda x: np.quantile(x,.75),'resamples':'count'}).reset_index(drop=True)

# Concatenate dynamic predictor relevances and save
plot_df_dynamic_predictors = pd.concat([specific_dynamic_relevances,unspecific_dynamic_relevances],ignore_index=True).sort_values(by=['TUNE_IDX','median'],ascending=[True,False]).reset_index(drop=True)
plot_df_dynamic_predictors.TOKEN[plot_df_dynamic_predictors.TOKEN.isna()] = 'Other'
plot_df_dynamic_predictors.BaseToken[plot_df_dynamic_predictors.BaseToken.isna()] = 'Other'
plot_df_dynamic_predictors.to_csv(os.path.join(relevance_dir,'dynamic_relevances_plot_df.csv'),index=False)

# Identify top 20 and bottom 3 intervention predictors per tuning index
intervention_relevance_layers = predictor_relevances[predictor_relevances.ICUIntervention].sort_values(by=['TUNE_IDX','median'],ascending=[True,False]).reset_index(drop=True)
specific_intervention_relevances = pd.concat([intervention_relevance_layers.groupby('TUNE_IDX').head(20),intervention_relevance_layers.groupby('TUNE_IDX').tail(3)],ignore_index=True)

# For intervention predictors not in the top 20 or bottom 3, calculate summary statistics
unspecific_intervention_relevances = intervention_relevance_layers.merge(specific_intervention_relevances,how='left', indicator=True)
unspecific_intervention_relevances = unspecific_intervention_relevances[unspecific_intervention_relevances._merge=='left_only'].drop(columns='_merge').reset_index(drop=True)[['TUNE_IDX','TOKEN']]
unspecific_intervention_relevances = compiled_relevance_layers.merge(unspecific_intervention_relevances,how='inner').groupby(['TUNE_IDX'],as_index=False)['RELEVANCE'].aggregate({'mean':np.mean,'std':np.std,'median':np.median,'min':np.min,'max':np.max,'Q1':lambda x: np.quantile(x,.25),'Q3':lambda x: np.quantile(x,.75),'resamples':'count'}).reset_index(drop=True)

# Concatenate intervention predictor relevances and save
plot_df_intervention_predictors = pd.concat([specific_intervention_relevances,unspecific_intervention_relevances],ignore_index=True).sort_values(by=['TUNE_IDX','median'],ascending=[True,False]).reset_index(drop=True)
plot_df_intervention_predictors.TOKEN[plot_df_intervention_predictors.TOKEN.isna()] = 'Other'
plot_df_intervention_predictors.BaseToken[plot_df_intervention_predictors.BaseToken.isna()] = 'Other'
plot_df_intervention_predictors.to_csv(os.path.join(relevance_dir,'intervention_relevances_plot_df.csv'),index=False)