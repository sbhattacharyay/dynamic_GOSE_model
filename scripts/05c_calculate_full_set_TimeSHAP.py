#### Master Script 05c: Calculating TimeSHAP for dynAPM_DeepMN models in parallel ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate testing set TimeSHAP values based on provided TimeSHAP partition row index

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
VERSION = 'v6-0'

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Define directory in which tokens are stored
tokens_dir = '/home/sb2406/rds/hpc-work/tokens'

# Load the current version tuning grid
# post_tuning_grid = pd.read_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'))
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))
post_tuning_grid = tuning_grid[tuning_grid.TUNE_IDX==135].reset_index(drop=True)

# Load cross-validation split information to extract testing resamples
cv_splits = pd.read_csv('../cross_validation_splits.csv')
study_GUPIs = cv_splits[['GUPI','GOSE']].drop_duplicates()
test_splits = cv_splits[cv_splits.SET == 'test'].reset_index(drop=True)
uniq_GUPIs = test_splits.GUPI.unique()

# Define a directory for the storage of model interpretation values
interp_dir = '/home/sb2406/rds/hpc-work/model_interpretations/'+VERSION

# Define a directory for the storage of TimeSHAP values
shap_dir = os.path.join(interp_dir,'timeSHAP')

# Create a subdirectory for the storage of TimeSHAP values
sub_shap_dir = os.path.join(shap_dir,'parallel_results')
os.makedirs(sub_shap_dir,exist_ok=True)

# Create a subdirectory for the storage of missed TimeSHAP transitions
missed_transition_dir = os.path.join(shap_dir,'missed_transitions')
os.makedirs(missed_transition_dir,exist_ok=True)

# Load partitioned significant clinical transitions for allocated TimeSHAP calculation
timeshap_partitions = cp.load(open(os.path.join(shap_dir,'timeSHAP_partitions.pkl'),"rb"))
# timeshap_partitions = cp.load(open(os.path.join(shap_dir,'remaining_timeSHAP_partitions.pkl'),"rb"))
# remaining_partition_indices = cp.load(open(os.path.join(shap_dir,'remaining_timeSHAP_partition_indices.pkl'),"rb"))

# Read model checkpoint information dataframe
ckpt_info = pd.read_pickle(os.path.join(shap_dir,'ckpt_info.pkl'))

# # Load summarised training set predictions from TimeSHAP directory
# summ_train_preds = pd.read_pickle(os.path.join(shap_dir,'summarised_training_set_predictions.pkl'))

# # Load average-event predictions from TimeSHAP directory
# avg_event_preds = pd.read_pickle(os.path.join(shap_dir,'average_event_predictions.pkl'))

### II. Calculate testing set TimeSHAP values based on provided TimeSHAP partition row index
# Argument-induced bootstrapping functions
def main(array_task_id):

    ## Calculate the "average-event" for TimeSHAP
    # Extract current significant clinical transitions based on `array_task_id`
    curr_transitions = timeshap_partitions[array_task_id]
    
    # Identify unique CV partitions in current batch to load training set predictions
    unique_cv_partitons = curr_transitions[['REPEAT','FOLD','TUNE_IDX']].drop_duplicates().reset_index(drop=True)
    
    # Create empty lists to store average events and zero events
    avg_event_lists = []
    zero_event_lists = []
    
    # Create empty list to store current transition testing set predictions
    curr_testing_sets = []
    
    # Iterate through unique CV partitions to load and calculate average event
    for curr_cv_row in tqdm(range(unique_cv_partitons.shape[0]),'Iterating through unique cross-validation partitions to calculate TimeSHAP'):
        
        # Extract current repeat, fold, and tuning index
        curr_repeat = unique_cv_partitons.REPEAT[curr_cv_row]
        curr_fold = unique_cv_partitons.FOLD[curr_cv_row]
        curr_tune_idx = unique_cv_partitons.TUNE_IDX[curr_cv_row]
        
        # Define current fold token subdirectory
        # token_fold_dir = os.path.join(tokens_dir,'fold'+str(curr_fold))
        token_fold_dir = os.path.join(tokens_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold))
            
        # Load current token-indexed training set
        # training_set = pd.read_pickle(os.path.join(token_fold_dir,'training_indices.pkl'))
        training_set = pd.read_pickle(os.path.join(token_fold_dir,'from_adm_strategy_abs_training_indices.pkl'))
            
        # Filter training set predictions based on `WindowIdx`
        training_set = training_set[training_set.WindowIdx <= 84].reset_index(drop=True)

        # Load current token-indexed testing set
    #             testing_set = pd.read_pickle(os.path.join(token_fold_dir,'testing_indices.pkl'))
        testing_set = pd.read_pickle(os.path.join(token_fold_dir,'from_adm_strategy_abs_testing_indices.pkl'))

        # Filter testing set predictions based on `WindowIdx`
        testing_set = testing_set[testing_set.WindowIdx <= 84].reset_index(drop=True)

        # Retrofit dataframe
        training_set = training_set.rename(columns={'VocabTimeFromAdmIndex':'VocabDaysSinceAdmIndex'})        
        testing_set = testing_set.rename(columns={'VocabTimeFromAdmIndex':'VocabDaysSinceAdmIndex'})

        # Load current token dictionary
#             curr_vocab = cp.load(open(os.path.join(token_fold_dir,'token_dictionary.pkl'),"rb"))
        curr_vocab = cp.load(open(os.path.join(token_fold_dir,'from_adm_strategy_abs_token_dictionary.pkl'),"rb"))
        unknown_index = curr_vocab['<unk>']
        
        # Extract current configuration time-token parameter
        curr_time_tokens = post_tuning_grid.TIME_TOKENS[post_tuning_grid.TUNE_IDX==curr_tune_idx].values[0]

        # Format time tokens of index sets based on current tuning configuration
        format_training_set,time_tokens_mask = format_time_tokens(training_set.copy(),curr_time_tokens,False)
                
        # Format time tokens of index sets based on current tuning configuration
        format_testing_set,time_tokens_mask = format_time_tokens(testing_set.copy(),curr_time_tokens,False)
        format_testing_set['SeqLength'] = format_testing_set.VocabIndex.apply(len)
        format_testing_set['Unknowns'] = format_testing_set.VocabIndex.apply(lambda x: x.count(unknown_index))        

        # Calculate number of columns to add
        cols_to_add = max(format_testing_set['Unknowns'].max(),1) - 1

        # Define token labels from current vocab
        token_labels = curr_vocab.get_itos() + [curr_vocab.get_itos()[unknown_index]+'_'+str(i+1).zfill(3) for i in range(cols_to_add)]
        token_labels[unknown_index] = token_labels[unknown_index]+'_000'
        
        # Convert training set dataframe to multihot matrix
        training_multihot = df_to_multihot_matrix(format_training_set, len(curr_vocab), unknown_index, cols_to_add)
        
        # For first-pass, define average-token dataframe from training set for "average event"
        training_token_frequencies = training_multihot.sum(0)/training_multihot.shape[0]
        average_event = pd.DataFrame(np.expand_dims((training_token_frequencies>0.5).astype(int),0), index=np.arange(1), columns=token_labels)
        
        # Add CV partition and tuning index to average event dataframe and append to empty list
        average_event.insert(0,'REPEAT',curr_repeat)
        average_event.insert(1,'FOLD',curr_fold)
        average_event.insert(2,'TUNE_IDX',curr_tune_idx)
        avg_event_lists.append(average_event)
        
        # Define zero-token dataframe for second-pass "average event"
        zero_event = pd.DataFrame(0, index=np.arange(1), columns=token_labels)
        
        # Add CV partition and tuning index to average event dataframe and append to empty list
        zero_event.insert(0,'REPEAT',curr_repeat)
        zero_event.insert(1,'FOLD',curr_fold)
        zero_event.insert(2,'TUNE_IDX',curr_tune_idx)
        zero_event_lists.append(zero_event)
        
        # Add cross-validation partition and tuning configuration information to testing set dataframe
        format_testing_set['REPEAT'] = curr_repeat
        format_testing_set['FOLD']= curr_fold
        format_testing_set['TUNE_IDX'] = curr_tune_idx
        
        # Filter testing set and store
        curr_testing_sets.append(format_testing_set[format_testing_set.GUPI.isin(curr_transitions[(curr_transitions.REPEAT==curr_repeat)&(curr_transitions.FOLD==curr_fold)&(curr_transitions.TUNE_IDX==curr_tune_idx)].GUPI.unique())].reset_index(drop=True))
        
    # Concatenate average- and zero-event lists for storage
    avg_event_lists = pd.concat(avg_event_lists,ignore_index=True)
    zero_event_lists = pd.concat(zero_event_lists,ignore_index=True)
    
    # Concatenate filtered testing sets for storage
    curr_testing_sets = pd.concat(curr_testing_sets,ignore_index=True)
    
    ## Calculate TimeSHAP values for prediction contributions to expected GOSE
    # Isolate unique combinations for expected GOSE contribution calculations
    unique_expected_GOSE_transitions = curr_transitions[['REPEAT','FOLD','GUPI','TUNE_IDX','WindowIdx']].drop_duplicates().reset_index(drop=True)
    
    # Initialize empty list to compile TimeSHAP dataframes
    compiled_expected_GOSE_ts = []
    
    # Initialize empty list to compile missed transitions
    compiled_expected_GOSE_missed = []
    
    # Iterate through unique combinations and calculate TimeSHAP    
    for curr_uniq_row in tqdm(range(unique_expected_GOSE_transitions.shape[0]),'Iterating through unique combinations to calculate expected GOSE TimeSHAP'):
        
        # Extract current repeat, fold, GUPI, tuning index, and window index
        curr_repeat = unique_expected_GOSE_transitions.REPEAT[curr_uniq_row]
        curr_fold = unique_expected_GOSE_transitions.FOLD[curr_uniq_row]
        curr_GUPI = unique_expected_GOSE_transitions.GUPI[curr_uniq_row]
        curr_tune_idx = unique_expected_GOSE_transitions.TUNE_IDX[curr_uniq_row]
        curr_wi = unique_expected_GOSE_transitions.WindowIdx[curr_uniq_row]
        
        # Extract average- and zero-events based on current combination parameters
        curr_avg_event = avg_event_lists[(avg_event_lists.REPEAT==curr_repeat)&(avg_event_lists.FOLD==curr_fold)&(avg_event_lists.TUNE_IDX==curr_tune_idx)].drop(columns=['REPEAT','FOLD','TUNE_IDX']).reset_index(drop=True)
        curr_zero_event = zero_event_lists[(zero_event_lists.REPEAT==curr_repeat)&(zero_event_lists.FOLD==curr_fold)&(zero_event_lists.TUNE_IDX==curr_tune_idx)].drop(columns=['REPEAT','FOLD','TUNE_IDX']).reset_index(drop=True)
        
        # Extract testing set predictions based on current combination parameters
        filt_testing_set = curr_testing_sets[(curr_testing_sets.GUPI==curr_GUPI)&(curr_testing_sets.REPEAT==curr_repeat)&(curr_testing_sets.FOLD==curr_fold)&(curr_testing_sets.TUNE_IDX==curr_tune_idx)].reset_index(drop=True)
        
        # Define current fold token subdirectory
#       token_fold_dir = os.path.join(tokens_dir,'fold'+str(curr_fold))
        token_fold_dir = os.path.join(tokens_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold))

        # Load current token dictionary
#       curr_vocab = cp.load(open(os.path.join(token_fold_dir,'token_dictionary.pkl'),"rb"))
        curr_vocab = cp.load(open(os.path.join(token_fold_dir,'from_adm_strategy_abs_token_dictionary.pkl'),"rb"))
        unknown_index = curr_vocab['<unk>']

        # Convert filtered testing set dataframe to multihot matrix
        testing_multihot = df_to_multihot_matrix(filt_testing_set, len(curr_vocab), unknown_index, curr_avg_event.shape[1]-len(curr_vocab))

        # Filter testing multihot matrix up to the window index of focus
        filt_testing_multihot = np.expand_dims(testing_multihot[:curr_wi,:],axis=0)
                        
        # Extract current file and required hyperparameter information
        curr_file = ckpt_info.file[(ckpt_info.REPEAT==curr_repeat)&(ckpt_info.FOLD==curr_fold)&(ckpt_info.TUNE_IDX==curr_tune_idx)].values[0]
        #curr_time_tokens = post_tuning_grid.TIME_TOKENS[(post_tuning_grid.TUNE_IDX==curr_tune_idx)&(post_tuning_grid.FOLD==curr_fold)].values[0]
        curr_time_tokens = post_tuning_grid.TIME_TOKENS[post_tuning_grid.TUNE_IDX==curr_tune_idx].values[0]
        curr_rnn_type = post_tuning_grid[post_tuning_grid.TUNE_IDX==curr_tune_idx].RNN_TYPE.values[0]
            
        # Load current pretrained model
        gose_model = GOSE_model.load_from_checkpoint(curr_file)
        gose_model.eval()

        # Initialize custom TimeSHAP model for expected value effect calculation
        ts_GOSE_model = timeshap_GOSE_model(gose_model,curr_rnn_type,-1,unknown_index,curr_avg_event.shape[1]-len(curr_vocab))
        wrapped_gose_model = TorchModelWrapper(ts_GOSE_model)
        f_hs = lambda x, y=None: wrapped_gose_model.predict_last_hs(x, y)

        # First, try to calculate expected GOSE TimeSHAP values with average event baseline
        try:
            # Prune timepoints based on tolerance of 0.15
            _,prun_idx = tsx.local_pruning(f_hs, filt_testing_multihot, {'tol': 0.15}, curr_avg_event, entity_uuid=None, entity_col=None, verbose=True)

            # Calculate local feature-level TimeSHAP values after pruning
            feature_dict = {'rs': 2022, 'nsamples': 3200, 'feature_names': curr_avg_event.columns.to_list()}
            ts_feature_data = tsx.local_feat(f_hs, filt_testing_multihot, feature_dict, entity_uuid=None, entity_col=None, baseline=curr_avg_event, pruned_idx=filt_testing_multihot.shape[1]+prun_idx)
        
            # Find features that exist within unpruned region
            existing_features = np.asarray(curr_avg_event.columns.to_list())[filt_testing_multihot[:,filt_testing_multihot.shape[1]+prun_idx:,:].sum(1).squeeze(0) > 0]

            # Filter feature-level TimeSHAP values to existing features
            ts_feature_data = ts_feature_data[ts_feature_data.Feature.isin(existing_features)].reset_index(drop=True)

            # Add metadata to TimeSHAP feature dataframe
            ts_feature_data['REPEAT'] = curr_repeat
            ts_feature_data['FOLD'] = curr_fold
            ts_feature_data['TUNE_IDX'] = curr_tune_idx
            ts_feature_data['Threshold'] = 'ExpectedValue'
            ts_feature_data['GUPI'] = curr_GUPI
            ts_feature_data['WindowIdx'] = curr_wi
            ts_feature_data['BaselineFeatures'] = 'Average'
            #ts_feature_data = ts_feature_data.merge(curr_transitions,how='left')

            #Append current TimeSHAP dataframe to compilation list
            compiled_expected_GOSE_ts.append(ts_feature_data)
        
        except:
            # Second, try to calculate expected GOSE TimeSHAP values with zero event baseline
            try:                 
                # Prune timepoints based on tolerance of 0.15
                _,prun_idx = tsx.local_pruning(f_hs, filt_testing_multihot, {'tol': 0.15}, curr_zero_event, entity_uuid=None, entity_col=None, verbose=True)

                # Calculate local feature-level TimeSHAP values after pruning
                feature_dict = {'rs': 2022, 'nsamples': 3200, 'feature_names': curr_zero_event.columns.to_list()}
                ts_feature_data = tsx.local_feat(f_hs, filt_testing_multihot, feature_dict, entity_uuid=None, entity_col=None, baseline=curr_zero_event, pruned_idx=filt_testing_multihot.shape[1]+prun_idx)
            
                # Find features that exist within unpruned region
                existing_features = np.asarray(curr_zero_event.columns.to_list())[filt_testing_multihot[:,filt_testing_multihot.shape[1]+prun_idx:,:].sum(1).squeeze(0) > 0]

                # Filter feature-level TimeSHAP values to existing features
                ts_feature_data = ts_feature_data[ts_feature_data.Feature.isin(existing_features)].reset_index(drop=True)

                # Add metadata to TimeSHAP feature dataframe
                ts_feature_data['REPEAT'] = curr_repeat
                ts_feature_data['FOLD'] = curr_fold
                ts_feature_data['TUNE_IDX'] = curr_tune_idx
                ts_feature_data['Threshold'] = 'ExpectedValue'
                ts_feature_data['GUPI'] = curr_GUPI
                ts_feature_data['WindowIdx'] = curr_wi
                ts_feature_data['BaselineFeatures'] = 'Zero'
                #ts_feature_data = ts_feature_data.merge(curr_transitions,how='left')

                #Append current TimeSHAP dataframe to compilation list
                compiled_expected_GOSE_ts.append(ts_feature_data)
            
            except:
                # Identify significant transitions for which TimeSHAP cannot be calculated
                curr_missed_transition = unique_expected_GOSE_transitions.iloc[[curr_uniq_row]].reset_index(drop=True)

                # Append to running list of missing transitions
                compiled_expected_GOSE_missed.append(curr_missed_transition)
                
    # If missed transitions exist, compile and save
    if compiled_expected_GOSE_missed:
        
        # Compile list of missed transitions
        compiled_expected_GOSE_missed = pd.concat(compiled_expected_GOSE_missed,ignore_index=True)
        
        # Save compiled missed transition values into SHAP subdirectory
        compiled_expected_GOSE_missed.to_pickle(os.path.join(missed_transition_dir,'exp_GOSE_missing_transitions_partition_idx_'+str(array_task_id).zfill(4)+'.pkl'))
#         compiled_expected_GOSE_missed.to_pickle(os.path.join(missed_transition_dir,'missing_transitions_partition_idx_'+str(remaining_partition_indices[array_task_id]).zfill(4)+'.pkl'))
    
    # If TimeSHAP feature values exist, compile and save
    if compiled_expected_GOSE_ts:
        
        # Compile list of TimeSHAP dataframes
        compiled_expected_GOSE_ts = pd.concat(compiled_expected_GOSE_ts,ignore_index=True)

        # Rename `Shapley Value` column
        compiled_expected_GOSE_ts = compiled_expected_GOSE_ts.rename(columns={'Shapley Value':'SHAP'})

        # Save compiled TimeSHAP values into SHAP subdirectory
        compiled_expected_GOSE_ts.to_pickle(os.path.join(sub_shap_dir,'exp_GOSE_timeSHAP_values_partition_idx_'+str(array_task_id).zfill(4)+'.pkl'))
#         compiled_expected_GOSE_ts.to_pickle(os.path.join(sub_shap_dir,'timeSHAP_values_partition_idx_'+str(remaining_partition_indices[array_task_id]).zfill(4)+'.pkl'))

    ## Calculate TimeSHAP values for prediction contributions to GOSE thresholds
    # Define list of possible GOSE thresholds
    thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
    
    # Initialize empty list to compile TimeSHAP dataframes
    compiled_threshold_GOSE_ts = []
    
    # Initialize empty list to compile missed transitions
    compiled_threshold_GOSE_missed = []
    
    # Iterate through unique combinations and calculate TimeSHAP    
    for curr_trans_row in tqdm(range(curr_transitions.shape[0]),'Iterating through unique combinations to calculate threshold GOSE TimeSHAP'):
        
        # Extract current repeat, fold, GUPI, tuning index, and window index
        curr_repeat = curr_transitions.REPEAT[curr_trans_row]
        curr_fold = curr_transitions.FOLD[curr_trans_row]
        curr_GUPI = curr_transitions.GUPI[curr_trans_row]
        curr_tune_idx = curr_transitions.TUNE_IDX[curr_trans_row]
        curr_wi = curr_transitions.WindowIdx[curr_trans_row]
        curr_thresh = curr_transitions.Threshold[curr_trans_row]
        curr_thresh_idx = thresh_labels.index(curr_thresh)
        
        # Extract average- and zero-events based on current combination parameters
        curr_avg_event = avg_event_lists[(avg_event_lists.REPEAT==curr_repeat)&(avg_event_lists.FOLD==curr_fold)&(avg_event_lists.TUNE_IDX==curr_tune_idx)].drop(columns=['REPEAT','FOLD','TUNE_IDX']).reset_index(drop=True)
        curr_zero_event = zero_event_lists[(zero_event_lists.REPEAT==curr_repeat)&(zero_event_lists.FOLD==curr_fold)&(zero_event_lists.TUNE_IDX==curr_tune_idx)].drop(columns=['REPEAT','FOLD','TUNE_IDX']).reset_index(drop=True)
        
        # Extract testing set predictions based on current combination parameters
        filt_testing_set = curr_testing_sets[(curr_testing_sets.GUPI==curr_GUPI)&(curr_testing_sets.REPEAT==curr_repeat)&(curr_testing_sets.FOLD==curr_fold)&(curr_testing_sets.TUNE_IDX==curr_tune_idx)].reset_index(drop=True)
        
        # Define current fold token subdirectory
#       token_fold_dir = os.path.join(tokens_dir,'fold'+str(curr_fold))
        token_fold_dir = os.path.join(tokens_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold))

        # Load current token dictionary
#       curr_vocab = cp.load(open(os.path.join(token_fold_dir,'token_dictionary.pkl'),"rb"))
        curr_vocab = cp.load(open(os.path.join(token_fold_dir,'from_adm_strategy_abs_token_dictionary.pkl'),"rb"))
        unknown_index = curr_vocab['<unk>']

        # Convert filtered testing set dataframe to multihot matrix
        testing_multihot = df_to_multihot_matrix(filt_testing_set, len(curr_vocab), unknown_index, curr_avg_event.shape[1]-len(curr_vocab))

        # Filter testing multihot matrix up to the window index of focus
        filt_testing_multihot = np.expand_dims(testing_multihot[:curr_wi,:],axis=0)
                        
        # Extract current file and required hyperparameter information
        curr_file = ckpt_info.file[(ckpt_info.REPEAT==curr_repeat)&(ckpt_info.FOLD==curr_fold)&(ckpt_info.TUNE_IDX==curr_tune_idx)].values[0]
        #curr_time_tokens = post_tuning_grid.TIME_TOKENS[(post_tuning_grid.TUNE_IDX==curr_tune_idx)&(post_tuning_grid.FOLD==curr_fold)].values[0]
        curr_time_tokens = post_tuning_grid.TIME_TOKENS[post_tuning_grid.TUNE_IDX==curr_tune_idx].values[0]
        curr_rnn_type = post_tuning_grid[post_tuning_grid.TUNE_IDX==curr_tune_idx].RNN_TYPE.values[0]
            
        # Load current pretrained model
        gose_model = GOSE_model.load_from_checkpoint(curr_file)
        gose_model.eval()

        # Initialize custom TimeSHAP model for threshold value effect calculation
        ts_GOSE_model = timeshap_GOSE_model(gose_model,curr_rnn_type,curr_thresh_idx,unknown_index,curr_avg_event.shape[1]-len(curr_vocab))
        wrapped_gose_model = TorchModelWrapper(ts_GOSE_model)
        f_hs = lambda x, y=None: wrapped_gose_model.predict_last_hs(x, y)

        # First, try to calculate threshold GOSE TimeSHAP values with average event baseline
        try:
            # Prune timepoints based on tolerance of 0.025
            _,prun_idx = tsx.local_pruning(f_hs, filt_testing_multihot, {'tol': 0.025}, curr_avg_event, entity_uuid=None, entity_col=None, verbose=True)

            # Calculate local feature-level TimeSHAP values after pruning
            feature_dict = {'rs': 2022, 'nsamples': 3200, 'feature_names': curr_avg_event.columns.to_list()}
            ts_feature_data = tsx.local_feat(f_hs, filt_testing_multihot, feature_dict, entity_uuid=None, entity_col=None, baseline=curr_avg_event, pruned_idx=filt_testing_multihot.shape[1]+prun_idx)
        
            # Find features that exist within unpruned region
            existing_features = np.asarray(curr_avg_event.columns.to_list())[filt_testing_multihot[:,filt_testing_multihot.shape[1]+prun_idx:,:].sum(1).squeeze(0) > 0]

            # Filter feature-level TimeSHAP values to existing features
            ts_feature_data = ts_feature_data[ts_feature_data.Feature.isin(existing_features)].reset_index(drop=True)

            # Add metadata to TimeSHAP feature dataframe
            ts_feature_data['REPEAT'] = curr_repeat
            ts_feature_data['FOLD'] = curr_fold
            ts_feature_data['TUNE_IDX'] = curr_tune_idx
            ts_feature_data['Threshold'] = curr_thresh
            ts_feature_data['GUPI'] = curr_GUPI
            ts_feature_data['WindowIdx'] = curr_wi
            ts_feature_data['BaselineFeatures'] = 'Average'
            #ts_feature_data = ts_feature_data.merge(curr_transitions,how='left')

            #Append current TimeSHAP dataframe to compilation list
            compiled_threshold_GOSE_ts.append(ts_feature_data)
        
        except:
            # Second, try to calculate threshold GOSE TimeSHAP values with zero event baseline
            try:                 
                # Prune timepoints based on tolerance of 0.025
                _,prun_idx = tsx.local_pruning(f_hs, filt_testing_multihot, {'tol': 0.025}, curr_zero_event, entity_uuid=None, entity_col=None, verbose=True)

                # Calculate local feature-level TimeSHAP values after pruning
                feature_dict = {'rs': 2022, 'nsamples': 3200, 'feature_names': curr_zero_event.columns.to_list()}
                ts_feature_data = tsx.local_feat(f_hs, filt_testing_multihot, feature_dict, entity_uuid=None, entity_col=None, baseline=curr_zero_event, pruned_idx=filt_testing_multihot.shape[1]+prun_idx)
            
                # Find features that exist within unpruned region
                existing_features = np.asarray(curr_zero_event.columns.to_list())[filt_testing_multihot[:,filt_testing_multihot.shape[1]+prun_idx:,:].sum(1).squeeze(0) > 0]

                # Filter feature-level TimeSHAP values to existing features
                ts_feature_data = ts_feature_data[ts_feature_data.Feature.isin(existing_features)].reset_index(drop=True)

                # Add metadata to TimeSHAP feature dataframe
                ts_feature_data['REPEAT'] = curr_repeat
                ts_feature_data['FOLD'] = curr_fold
                ts_feature_data['TUNE_IDX'] = curr_tune_idx
                ts_feature_data['Threshold'] = curr_thresh
                ts_feature_data['GUPI'] = curr_GUPI
                ts_feature_data['WindowIdx'] = curr_wi
                ts_feature_data['BaselineFeatures'] = 'Zero'
                #ts_feature_data = ts_feature_data.merge(curr_transitions,how='left')

                #Append current TimeSHAP dataframe to compilation list
                compiled_threshold_GOSE_ts.append(ts_feature_data)
            
            except:
                # Identify significant transitions for which TimeSHAP cannot be calculated
                curr_missed_transition = curr_transitions.iloc[[curr_trans_row]].reset_index(drop=True)

                # Append to running list of missing transitions
                compiled_threshold_GOSE_missed.append(curr_missed_transition)
                
    # If missed transitions exist, compile and save
    if compiled_threshold_GOSE_missed:
        
        # Compile list of missed transitions
        compiled_threshold_GOSE_missed = pd.concat(compiled_threshold_GOSE_missed,ignore_index=True)
        
        # Save compiled missed transition values into SHAP subdirectory
        compiled_threshold_GOSE_missed.to_pickle(os.path.join(missed_transition_dir,'thresh_GOSE_missing_transitions_partition_idx_'+str(array_task_id).zfill(4)+'.pkl'))
#         compiled_threshold_GOSE_missed.to_pickle(os.path.join(missed_transition_dir,'missing_transitions_partition_idx_'+str(remaining_partition_indices[array_task_id]).zfill(4)+'.pkl'))
    
    # If TimeSHAP feature values exist, compile and save
    if compiled_threshold_GOSE_ts:
        
        # Compile list of TimeSHAP dataframes
        compiled_threshold_GOSE_ts = pd.concat(compiled_threshold_GOSE_ts,ignore_index=True)

        # Rename `Shapley Value` column
        compiled_threshold_GOSE_ts = compiled_threshold_GOSE_ts.rename(columns={'Shapley Value':'SHAP'})

        # Save compiled TimeSHAP values into SHAP subdirectory
        compiled_threshold_GOSE_ts.to_pickle(os.path.join(sub_shap_dir,'thresh_GOSE_timeSHAP_values_partition_idx_'+str(array_task_id).zfill(4)+'.pkl'))
#         compiled_threshold_GOSE_ts.to_pickle(os.path.join(sub_shap_dir,'timeSHAP_values_partition_idx_'+str(remaining_partition_indices[array_task_id]).zfill(4)+'.pkl'))

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])
    main(array_task_id)