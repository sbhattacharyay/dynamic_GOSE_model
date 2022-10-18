#### Master Script 05c (i): First pass of calculating TimeSHAP for dynAPM_DeepMN models in parallel ####
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

# Load summarised training set predictions from TimeSHAP directory
summ_train_preds = pd.read_pickle(os.path.join(shap_dir,'summarised_training_set_predictions.pkl'))

# Load average-event predictions from TimeSHAP directory
avg_event_preds = pd.read_pickle(os.path.join(shap_dir,'average_event_predictions.pkl'))

### II. Calculate testing set TimeSHAP values based on provided TimeSHAP partition row index
# Argument-induced bootstrapping functions
def main(array_task_id):

    # Initialize empty list to compile TimeSHAP dataframes
    compiled_ts_feature_data = []
    
    # Initialize empty list to compile missed transitions
    compiled_ts_transitions = []
    
    # Extract current significant clinical transitions based on `array_task_id`
    curr_transitions = timeshap_partitions[array_task_id]
    
    # Iterate through unique folds in current dataframe
    for curr_fold in tqdm(curr_transitions.FOLD.unique(),'Iterating through unique folds to calculate TimeSHAP'):
        
        # Define current fold token subdirectory
        token_fold_dir = os.path.join(tokens_dir,'fold'+str(curr_fold))

        # Load current token-indexed testing set
        testing_set = pd.read_pickle(os.path.join(token_fold_dir,'testing_indices.pkl'))

        # Filter testing set predictions based on `WindowIdx`
        testing_set = testing_set[testing_set.WindowIdx <= 84].reset_index(drop=True)

        # Load current token dictionary
        curr_vocab = cp.load(open(os.path.join(token_fold_dir,'token_dictionary.pkl'),"rb"))
        unknown_index = curr_vocab['<unk>']
        
        # Iterate through unique tuning indices in current dataframe
        for curr_tune_idx in tqdm(curr_transitions[curr_transitions.FOLD==curr_fold].TUNE_IDX.unique(),'Iterating through unique tuning indices in fold '+str(curr_fold)+' to calculate TimeSHAP'):
            
            # Extract current file and required hyperparameter information
            curr_file = ckpt_info.file[(ckpt_info.FOLD==curr_fold)&(ckpt_info.TUNE_IDX==curr_tune_idx)].values[0]
            curr_time_tokens = post_tuning_grid.TIME_TOKENS[(post_tuning_grid.TUNE_IDX==curr_tune_idx)&(post_tuning_grid.FOLD==curr_fold)].values[0]
        
            # Format time tokens of index sets based on current tuning configuration
            format_testing_set,time_tokens_mask = format_time_tokens(testing_set.copy(),curr_time_tokens,False)
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
            
            # Iterate through unique thresholds in current dataframe
            thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
            for curr_thresh in tqdm(curr_transitions[(curr_transitions.FOLD==curr_fold)&(curr_transitions.TUNE_IDX==curr_tune_idx)].Threshold.unique(),'Iterating through unique GOSE thresholds in fold '+str(curr_fold)+' and tuning index '+str(curr_tune_idx)+' to calculate TimeSHAP'):
                
                # Extract current threshold index
                curr_thresh_idx = thresh_labels.index(curr_thresh)
                
                # Initialize custom TimeSHAP model
                ts_GOSE_model = timeshap_GOSE_model(gose_model,curr_thresh_idx,unknown_index,cols_to_add)
                wrapped_gose_model = TorchModelWrapper(ts_GOSE_model)
                f_hs = lambda x, y=None: wrapped_gose_model.predict_last_hs(x, y)
                
                # Iterate through unique GUPIs in current dataframe
                for curr_GUPI in tqdm(curr_transitions[(curr_transitions.FOLD==curr_fold)&(curr_transitions.TUNE_IDX==curr_tune_idx)&(curr_transitions.Threshold==curr_thresh)].GUPI.unique(),'Iterating through unique GUPIs in fold '+str(curr_fold)+' and tuning index '+str(curr_tune_idx)+' and threshold '+curr_thresh+' to calculate TimeSHAP'):
                
                    # Filter testing set for current GUPI predictions
                    filt_testing_set = format_testing_set[format_testing_set.GUPI==curr_GUPI].reset_index(drop=True)
                
                    # Convert filtered testing set dataframe to multihot matrix
                    testing_multihot = df_to_multihot_matrix(filt_testing_set, len(curr_vocab), unknown_index, cols_to_add)
                    
                    # Iterate through unique window indices in current dataframe
                    for curr_wi in tqdm(curr_transitions[(curr_transitions.FOLD==curr_fold)&(curr_transitions.TUNE_IDX==curr_tune_idx)&(curr_transitions.Threshold==curr_thresh)&(curr_transitions.GUPI==curr_GUPI)].WindowIdx.unique(),'Iterating through unique window indices in fold '+str(curr_fold)+' and tuning index '+str(curr_tune_idx)+' and threshold '+curr_thresh+' and GUPI '+curr_GUPI+' to calculate TimeSHAP'):
                        
                        # Filter testing multihot matrix up to the window index of focus
                        filt_testing_multihot = np.expand_dims(testing_multihot[:curr_wi,:],axis=0)
                        
                        try:
                            
                            # Prune timepoints based on tolerance of 0.025
                            _,prun_idx = tsx.local_pruning(f_hs, filt_testing_multihot, {'tol': 0.025}, average_event, entity_uuid=None, entity_col=None, verbose=True)

                            # Calculate local feature-level TimeSHAP values after pruning
                            feature_dict = {'rs': 42, 'nsamples': 3200, 'feature_names': token_labels}
                            ts_feature_data = tsx.local_feat(f_hs, filt_testing_multihot, feature_dict, entity_uuid=None, entity_col=None, baseline=average_event, pruned_idx=filt_testing_multihot.shape[1]+prun_idx)

                            # Find features that exist within unpruned region
                            existing_features = np.asarray(token_labels)[filt_testing_multihot[:,filt_testing_multihot.shape[1]+prun_idx:,:].sum(1).squeeze(0) > 0]

                            # Filter feature-level TimeSHAP values to existing features
                            ts_feature_data = ts_feature_data[ts_feature_data.Feature.isin(existing_features)].reset_index(drop=True)

                            # Add metadata to TimeSHAP feature dataframe
                            ts_feature_data['FOLD'] = curr_fold
                            ts_feature_data['TUNE_IDX'] = curr_tune_idx
                            ts_feature_data['Threshold'] = curr_thresh
                            ts_feature_data['GUPI'] = curr_GUPI
                            ts_feature_data['WindowIdx'] = curr_wi
                            ts_feature_data = ts_feature_data.merge(curr_transitions,how='left')

                            # Append current TimeSHAP dataframe to compilation list
                            compiled_ts_feature_data.append(ts_feature_data)
                            
                        except:
                            
                            # Identify significant transitions for which TimeSHAP cannot be calculated
                            curr_missed_transition = curr_transitions[(curr_transitions.FOLD==curr_fold)&(curr_transitions.TUNE_IDX==curr_tune_idx)&(curr_transitions.Threshold==curr_thresh)&(curr_transitions.GUPI==curr_GUPI)&(curr_transitions.WindowIdx==curr_wi)].reset_index(drop=True)
                            
                            # Append to running list of missing transitions
                            compiled_ts_transitions.append(curr_missed_transition)

    # If missed transitions exist, compile and save
    if compiled_ts_transitions:
        
        # Compile list of missed transitions
        compiled_ts_transitions = pd.concat(compiled_ts_transitions,ignore_index=True)
        
        # Save compiled missed transition values into SHAP subdirectory
        compiled_ts_transitions.to_pickle(os.path.join(missed_transition_dir,'timeSHAP_values_partition_idx_'+str(array_task_id).zfill(4)+'.pkl'))
#         compiled_ts_transitions.to_pickle(os.path.join(missed_transition_dir,'missing_transitions_partition_idx_'+str(remaining_partition_indices[array_task_id]).zfill(4)+'.pkl'))
    
    # If TimeSHAP feature values exist, compile and save
    if compiled_ts_feature_data:
        
        # Compile list of TimeSHAP dataframes
        compiled_ts_feature_data = pd.concat(compiled_ts_feature_data,ignore_index=True)

        # Rename `Shapley Value` column
        compiled_ts_feature_data = compiled_ts_feature_data.rename(columns={'Shapley Value':'SHAP'})

        # Save compiled TimeSHAP values into SHAP subdirectory
        compiled_ts_feature_data.to_pickle(os.path.join(sub_shap_dir,'timeSHAP_values_partition_idx_'+str(array_task_id).zfill(4)+'.pkl'))
#         compiled_ts_feature_data.to_pickle(os.path.join(sub_shap_dir,'timeSHAP_values_partition_idx_'+str(remaining_partition_indices[array_task_id]).zfill(4)+'.pkl'))

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])
    main(array_task_id)