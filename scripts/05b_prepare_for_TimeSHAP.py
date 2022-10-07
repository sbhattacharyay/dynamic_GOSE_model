#### Master Script 05b: Prepare environment to calculate TimeSHAP for dynAPM_DeepMN models ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Identify transition points in testing set predictions for TimeSHAP focus
# III. Partition significant transition points for parallel TimeSHAP calculation
# IV. Calculate average training set predictions per tuning configuration

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

# Import TimeSHAP methods
import timeshap.explainer as tsx
import timeshap.plot as tsp
from timeshap.wrappers import TorchModelWrapper
from timeshap.utils import get_avg_score_with_avg_event

# Custom methods
from classes.datasets import DYN_ALL_PREDICTOR_SET
from models.dynamic_APM import GOSE_model, timeshap_GOSE_model
from functions.model_building import collate_batch, format_shap, format_tokens, format_time_tokens, df_to_multihot_matrix

# Set version code
VERSION = 'v7-0'

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Define directory in which tokens are stored
tokens_dir = '/home/sb2406/rds/hpc-work/tokens'

# Load the current version tuning grid
post_tuning_grid = pd.read_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'))

# Load cross-validation split information to extract testing resamples
cv_splits = pd.read_csv('../cross_validation_splits.csv')
study_GUPIs = cv_splits[['GUPI','GOSE']].drop_duplicates()
test_splits = cv_splits[cv_splits.SET == 'test'].reset_index(drop=True)
uniq_GUPIs = test_splits.GUPI.unique()

# Define a directory for the storage of model interpretation values
interp_dir = '/home/sb2406/rds/hpc-work/model_interpretations/'+VERSION
os.makedirs(interp_dir,exist_ok=True)

# Define a directory for the storage of TimeSHAP values
shap_dir = os.path.join(interp_dir,'timeSHAP')
os.makedirs(shap_dir,exist_ok=True)

### II. Identify transition points in testing set predictions for TimeSHAP focus
# Load and filter testing set predictions
test_predictions_df = pd.read_pickle(os.path.join(model_dir,'compiled_test_predictions.pkl'))
test_predictions_df = test_predictions_df[(test_predictions_df.TUNE_IDX.isin(post_tuning_grid.TUNE_IDX))].reset_index(drop=True)

# Remove logit columns
logit_cols = [col for col in test_predictions_df if col.startswith('z_GOSE=')]
test_predictions_df = test_predictions_df.drop(columns=logit_cols).reset_index(drop=True)

# Calculate threshold-based prediction probabilities
prob_cols = [col for col in test_predictions_df if col.startswith('Pr(GOSE=')]
thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
for thresh in range(1,len(prob_cols)):
    cols_gt = prob_cols[thresh:]
    prob_gt = test_predictions_df[cols_gt].sum(1).values
    gt = (test_predictions_df['TrueLabel'] >= thresh).astype(int).values
    test_predictions_df['Pr('+thresh_labels[thresh-1]+')'] = prob_gt
    test_predictions_df[thresh_labels[thresh-1]] = gt

# Remove GOSE probability columns
test_predictions_df = test_predictions_df.drop(columns=prob_cols).reset_index(drop=True)

# Filter out window indices that are outside area of analysis
test_predictions_df = test_predictions_df[(test_predictions_df.WindowIdx>3)&(test_predictions_df.WindowIdx<85)].reset_index(drop=True)

## Iterate through thresholds and identify significant points of transition per patient
# First iterate through each threshold, GUPI, and tuning index to identify points of prognostic change in correct direction during region of analysis
diff_values = []
for curr_thresh in tqdm(thresh_labels,'Identifying transition points at each GOSE threshold'):
    
    below_thresh_preds = test_predictions_df[test_predictions_df[curr_thresh] == 0].reset_index(drop=True)
    for curr_below_GUPI in tqdm(below_thresh_preds.GUPI.unique(),'Iterating through patients below threshold: '+curr_thresh):
        curr_GUPI_preds = below_thresh_preds[below_thresh_preds.GUPI==curr_below_GUPI].reset_index(drop=True)
        for curr_tune_idx in curr_GUPI_preds.TUNE_IDX.unique():
            curr_TI_preds = curr_GUPI_preds[curr_GUPI_preds.TUNE_IDX==curr_tune_idx][['GUPI','TUNE_IDX','WindowIdx','Pr('+curr_thresh+')']].reset_index(drop=True)
            curr_TI_preds['Diff'] = curr_TI_preds['Pr('+curr_thresh+')'].diff()
            curr_TI_preds = curr_TI_preds[curr_TI_preds.Diff < 0].drop(columns=['Pr('+curr_thresh+')']).reset_index(drop=True)
            curr_TI_preds['Threshold'] = curr_thresh
            diff_values.append(curr_TI_preds)
    
    above_thresh_preds = test_predictions_df[test_predictions_df[curr_thresh] == 1].reset_index(drop=True)
    for curr_above_GUPI in tqdm(above_thresh_preds.GUPI.unique(),'Iterating through patients above threshold: '+curr_thresh):
        curr_GUPI_preds = above_thresh_preds[above_thresh_preds.GUPI==curr_above_GUPI].reset_index(drop=True)
        for curr_tune_idx in curr_GUPI_preds.TUNE_IDX.unique():
            curr_TI_preds = curr_GUPI_preds[curr_GUPI_preds.TUNE_IDX==curr_tune_idx][['GUPI','TUNE_IDX','WindowIdx','Pr('+curr_thresh+')']].reset_index(drop=True)
            curr_TI_preds['Diff'] = curr_TI_preds['Pr('+curr_thresh+')'].diff()
            curr_TI_preds = curr_TI_preds[curr_TI_preds.Diff > 0].drop(columns=['Pr('+curr_thresh+')']).reset_index(drop=True)
            curr_TI_preds['Threshold'] = curr_thresh
            diff_values.append(curr_TI_preds)
diff_values = pd.concat(diff_values,ignore_index=True)

# Save calculated points of prognostic transition
diff_values.to_pickle(os.path.join(shap_dir,'all_transition_points.pkl'))

# Load calculated points of prognostic transition
diff_values = pd.read_pickle(os.path.join(shap_dir,'all_transition_points.pkl'))

# Add a marker to designate cases above and below threshold
diff_values['Above'] = diff_values['Diff'] > 0

# For each `TUNE_IDX`-`Threshold`-`Above` combination, find the quantile of differences
quantile_diffs = diff_values.groupby(['TUNE_IDX','Threshold','Above'],as_index=False)['Diff'].aggregate({'lo':lambda x: np.quantile(x,.1),'median':np.median,'hi':lambda x: np.quantile(x,.9)}).reset_index(drop=True)

# Arbitrarily define significant transitions by 90% quantile
positive_diffs = quantile_diffs[quantile_diffs.Above].rename(columns={'hi':'Cutoff'}).drop(columns=['lo','median']).reset_index(drop=True)
negative_diffs = quantile_diffs[~quantile_diffs.Above].rename(columns={'lo':'Cutoff'}).drop(columns=['hi','median']).reset_index(drop=True)
diff_cutoffs = pd.concat([positive_diffs,negative_diffs],ignore_index=True)

# Merge cutoffs to full difference values and filter transitions beyond cutoffs
diff_values = diff_values.merge(diff_cutoffs,how='left',on=['TUNE_IDX','Threshold','Above'])
sig_transitions = diff_values[diff_values.Diff.abs()>diff_values.Cutoff.abs()].reset_index(drop=True)

# Save significant points of prognostic transition
sig_transitions.to_pickle(os.path.join(shap_dir,'significant_transition_points.pkl'))

### III. Partition significant transition points for parallel TimeSHAP calculation
## Partition evenly for parallel calculation
# Load significant points of prognostic transition
sig_transitions = pd.read_pickle(os.path.join(shap_dir,'significant_transition_points.pkl'))

# Add fold information for instance in testing set and sort
sig_transitions = sig_transitions.merge(test_splits[['FOLD','GUPI']],how='left').sort_values(by=['TUNE_IDX','GUPI','Threshold','WindowIdx']).reset_index(drop=True)

# Partition evenly along number of available array tasks
max_array_tasks = 10000
s = [sig_transitions.shape[0] // max_array_tasks for _ in range(max_array_tasks)]
s[:(sig_transitions.shape[0] - sum(s))] = [over+1 for over in s[:(sig_transitions.shape[0] - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
timeshap_partitions = [sig_transitions.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True) for idx in range(len(start_idx))]

# Save derived partitions
cp.dump(timeshap_partitions, open(os.path.join(shap_dir,'timeSHAP_partitions.pkl'), "wb" ))

### IV. Calculate average training set predictions per tuning configuration
## Extract checkpoints for all top-performing tuning configurations
# Either create or load dynAPM checkpoint information for TimeSHAP calculation
if not os.path.exists(os.path.join(shap_dir,'ckpt_info.pkl')):
    
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
    ckpt_info.to_pickle(os.path.join(shap_dir,'ckpt_info.pkl'))
    
else:
    
    # Read model checkpoint information dataframe
    ckpt_info = pd.read_pickle(os.path.join(shap_dir,'ckpt_info.pkl'))

## Calculate and summarise training set predictions for each checkpoint
# Define variable to store summarised training set predictions
summ_train_preds = []

# Iterate through folds
for curr_fold in tqdm(ckpt_info.FOLD.unique(),'Iterating through folds to summarise training set predictions'):
    
    # Define current fold token subdirectory
    token_fold_dir = os.path.join(tokens_dir,'fold'+str(curr_fold))
    
    # Load current token-indexed training set
    training_set = pd.read_pickle(os.path.join(token_fold_dir,'training_indices.pkl'))
    
    # Filter training set predictions based on `WindowIdx`
    training_set = training_set[training_set.WindowIdx <= 84].reset_index(drop=True)
    
    # Iterate through tuning indices
    for curr_tune_idx in tqdm(ckpt_info[ckpt_info.FOLD==curr_fold].TUNE_IDX.unique(),'Iterating through tuning indices in fold '+str(curr_fold)+' to summarise training set predictions'):
        
        # Extract current file and required hyperparameter information
        curr_file = ckpt_info.file[(ckpt_info.FOLD==curr_fold)&(ckpt_info.TUNE_IDX==curr_tune_idx)].values[0]
        curr_time_tokens = post_tuning_grid.TIME_TOKENS[(post_tuning_grid.TUNE_IDX==curr_tune_idx)&(post_tuning_grid.FOLD==curr_fold)].values[0]
        
        # Format time tokens of index sets based on current tuning configuration
        format_training_set,time_tokens_mask = format_time_tokens(training_set.copy(),curr_time_tokens,True)
        
        # Add GOSE scores to training set
        format_training_set = pd.merge(format_training_set,study_GUPIs,how='left',on='GUPI')
        
        # Create PyTorch Dataset object
        train_Dataset = DYN_ALL_PREDICTOR_SET(format_training_set,post_tuning_grid.OUTPUT_ACTIVATION[(post_tuning_grid.TUNE_IDX==curr_tune_idx)&(post_tuning_grid.FOLD==curr_fold)].values[0])
        
        # Create PyTorch DataLoader objects
        curr_train_DL = DataLoader(train_Dataset,
                                   batch_size=len(train_Dataset),
                                   shuffle=False,
                                   collate_fn=collate_batch)
        
        # Load current pretrained model
        gose_model = GOSE_model.load_from_checkpoint(curr_file)
        gose_model.eval()
        
        # Calculate uncalibrated training set predictions
        with torch.no_grad():
            for i, (curr_train_label_list, curr_train_idx_list, curr_train_bin_offsets, curr_train_gupi_offsets, curr_train_gupis) in enumerate(curr_train_DL):
                (train_yhat, out_train_gupi_offsets) = gose_model(curr_train_idx_list, curr_train_bin_offsets, curr_train_gupi_offsets)
                curr_train_labels = torch.cat([curr_train_label_list],dim=0).cpu().numpy()
                if post_tuning_grid.OUTPUT_ACTIVATION[(post_tuning_grid.TUNE_IDX==curr_tune_idx)&(post_tuning_grid.FOLD==curr_fold)].values[0] == 'softmax': 
                    curr_train_preds = pd.DataFrame(F.softmax(torch.cat([train_yhat.detach()],dim=0)).cpu().numpy(),columns=['Pr(GOSE=1)','Pr(GOSE=2/3)','Pr(GOSE=4)','Pr(GOSE=5)','Pr(GOSE=6)','Pr(GOSE=7)','Pr(GOSE=8)'])
                    curr_train_preds['TrueLabel'] = curr_train_labels
                else:
                    raise ValueError("Invalid output layer type. Must be 'softmax' or 'sigmoid'")
                curr_train_preds.insert(loc=0, column='GUPI', value=curr_train_gupis)        
                curr_train_preds['TUNE_IDX'] = curr_tune_idx
        curr_train_preds['WindowIdx'] = curr_train_preds.groupby('GUPI').cumcount(ascending=True)+1
        
        # Calculate threshold-level probabilities of each prediction
        prob_cols = [col for col in curr_train_preds if col.startswith('Pr(GOSE=')]
        thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
        for thresh in range(1,len(prob_cols)):
            cols_gt = prob_cols[thresh:]
            prob_gt = curr_train_preds[cols_gt].sum(1).values
            curr_train_preds['Pr('+thresh_labels[thresh-1]+')'] = prob_gt
            
        # Remove GOSE probability columns
        curr_train_preds = curr_train_preds.drop(columns=prob_cols).reset_index(drop=True)
        
        # Melt dataframe into long form
        curr_train_preds = curr_train_preds.melt(id_vars=['GUPI','TrueLabel','TUNE_IDX','WindowIdx'],var_name='Threshold',value_name='Probability')
        
        # Calculate average threshold probability per threshold and window index
        curr_summ_train_preds = curr_train_preds.groupby(['TUNE_IDX','Threshold','WindowIdx'],as_index=False)['Probability'].mean()
        curr_summ_train_preds['FOLD'] = curr_fold
        
        # Append dataframe to running list
        summ_train_preds.append(curr_summ_train_preds)
        
# Concatenate summarised prediction list into dataframe
summ_train_preds = pd.concat(summ_train_preds,ignore_index=True)

# Sort summarised prediction dataframe and reorganize columns
summ_train_preds = summ_train_preds.sort_values(by=['TUNE_IDX','FOLD','Threshold','WindowIdx']).reset_index(drop=True)
summ_train_preds = summ_train_preds[['TUNE_IDX','FOLD','Threshold','WindowIdx','Probability']]

# Save summarised training set predictions into TimeSHAP directory
summ_train_preds.to_pickle(os.path.join(shap_dir,'summarised_training_set_predictions.pkl'))

## Calculate "average event" predictions for TimeSHAP
# Define variable to store average-event predictions
avg_event_preds = []

# Iterate through folds
for curr_fold in tqdm(ckpt_info.FOLD.unique(),'Iterating through folds to calculate average event predictions'):
    
    # Define current fold token subdirectory
    token_fold_dir = os.path.join(tokens_dir,'fold'+str(curr_fold))
    
    # Load current token-indexed testing set
    testing_set = pd.read_pickle(os.path.join(token_fold_dir,'testing_indices.pkl'))
    
    # Filter testing set predictions based on `WindowIdx`
    testing_set = testing_set[testing_set.WindowIdx <= 84].reset_index(drop=True)
    
    # Load current token-indexed training set
    training_set = pd.read_pickle(os.path.join(token_fold_dir,'training_indices.pkl'))
    
    # Filter training set predictions based on `WindowIdx`
    training_set = training_set[training_set.WindowIdx <= 84].reset_index(drop=True)
    
    # Load current token dictionary
    curr_vocab = cp.load(open(os.path.join(token_fold_dir,'token_dictionary.pkl'),"rb"))
    unknown_index = curr_vocab['<unk>']
    
    # Iterate through tuning indices
    for curr_tune_idx in tqdm(ckpt_info[ckpt_info.FOLD==curr_fold].TUNE_IDX.unique(),'Iterating through tuning indices in fold '+str(curr_fold)+' to calculate average event predictions'):
        
        # Extract current file and required hyperparameter information
        curr_file = ckpt_info.file[(ckpt_info.FOLD==curr_fold)&(ckpt_info.TUNE_IDX==curr_tune_idx)].values[0]
        curr_time_tokens = post_tuning_grid.TIME_TOKENS[(post_tuning_grid.TUNE_IDX==curr_tune_idx)&(post_tuning_grid.FOLD==curr_fold)].values[0]
        
        # Format time tokens of index sets based on current tuning configuration
        format_testing_set,time_tokens_mask = format_time_tokens(testing_set.copy(),curr_time_tokens,True)
        format_testing_set['SeqLength'] = format_testing_set.VocabIndex.apply(len)
        format_testing_set['Unknowns'] = format_testing_set.VocabIndex.apply(lambda x: x.count(unknown_index))        
        
        # Calculate number of columns to add
        cols_to_add = max(format_testing_set['Unknowns'].max(),1) - 1
        
        # Define token labels from current vocab
        token_labels = curr_vocab.get_itos() + [curr_vocab.get_itos()[unknown_index]+'_'+str(i+1).zfill(3) for i in range(cols_to_add)]
        token_labels[unknown_index] = token_labels[unknown_index]+'_000'
        
        # Define zero-token dataframe for "average event"
        average_event = pd.DataFrame(0, index=np.arange(1), columns=token_labels)
        
        # Load current pretrained model
        gose_model = GOSE_model.load_from_checkpoint(curr_file)
        gose_model.eval()
        
        # Calculate threshold-level probabilities of each prediction
        thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
        for thresh in range(len(thresh_labels)):     
            
            # Initialize custom TimeSHAP model
            ts_GOSE_model = timeshap_GOSE_model(gose_model,thresh,unknown_index,cols_to_add)
            wrapped_gose_model = TorchModelWrapper(ts_GOSE_model)
            f_hs = lambda x, y=None: wrapped_gose_model.predict_last_hs(x, y)
            
            # Calculate average prediction over time based on average event
            avg_score_over_len = get_avg_score_with_avg_event(f_hs, average_event, top=84)
            avg_score_over_len = pd.DataFrame(avg_score_over_len.items(),columns=['WindowIdx','Probability'])
            
            # Add metadata
            avg_score_over_len['Threshold'] = thresh_labels[thresh]
            avg_score_over_len['FOLD'] = curr_fold
            avg_score_over_len['TUNE_IDX'] = curr_tune_idx
            
            # Append dataframe to running list
            avg_event_preds.append(avg_score_over_len)

# Concatenate average-event prediction list into dataframe
avg_event_preds = pd.concat(avg_event_preds,ignore_index=True)

# Sort average-event prediction dataframe and reorganize columns
avg_event_preds = avg_event_preds.sort_values(by=['TUNE_IDX','FOLD','Threshold','WindowIdx']).reset_index(drop=True)
avg_event_preds = avg_event_preds[['TUNE_IDX','FOLD','Threshold','WindowIdx','Probability']]

# Save average-event predictions into TimeSHAP directory
avg_event_preds.to_pickle(os.path.join(shap_dir,'average_event_predictions.pkl'))