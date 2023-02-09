#### Master Script 6b: Calculate testing set predictions with dynamic tokens removed in parallel ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate static-only testing set predictions based on provided bootstrapping resample row index

### I. Initialisation
# Fundamental libraries
import os
import re
import sys
import time
import glob
import copy
import shutil
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
from torchtext.vocab import Vocab
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Custom methods
from classes.datasets import DYN_ALL_PREDICTOR_SET
from classes.calibration import TemperatureScaling, VectorScaling
from functions.model_building import format_time_tokens, collate_batch
from models.dynamic_APM import GOSE_model

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

# Load legacy cross-validation split information to extract testing resamples
legacy_cv_splits = pd.read_csv('../legacy_cross_validation_splits.csv')
study_GUPIs = legacy_cv_splits[['GUPI','GOSE']].drop_duplicates()

# Load and filter checkpoint file dataframe based on provided model version
ckpt_info = pd.read_pickle(os.path.join('/home/sb2406/rds/hpc-work/model_interpretations/',VERSION,'timeSHAP','ckpt_info.pkl'))
ckpt_info = ckpt_info[ckpt_info.TUNE_IDX==135].reset_index(drop=True)

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

### II. Calculate static-only testing set predictions based on provided bootstrapping resample row index
# Load the sensitivity analysis prediction grid
token_info_df = pd.read_pickle(os.path.join(model_dir,'sensitivity_analysis_prediction_grid.pkl'))

# Argument-induced bootstrapping functions
def main(array_task_id):

    # Load current token dictionary
    curr_vocab = cp.load(open(token_info_df.DICT_FILE[array_task_id],"rb"))

    # Create dataframe version of vocabulary
    curr_vocab_df = pd.DataFrame({'VocabIndex':list(range(len(curr_vocab))),'Token':curr_vocab.get_itos()})

    # Determine whether tokens are baseline
    curr_vocab_df['Baseline'] = curr_vocab_df['Token'].str.startswith('Baseline')

    # Create list of tokens non-static tokens that need to be masked
    nonstatic_mask = curr_vocab_df[~curr_vocab_df.Baseline].VocabIndex.to_list()

    # Load current token index dataframe
    curr_token_indices = pd.read_pickle(token_info_df.IDX_FILE[array_task_id])

    # Extract current tuning index
    curr_ti = token_info_df.TUNE_IDX[array_task_id]

    # Extract current token transformation parameters based on tuning index
    curr_time_tokens = post_tuning_grid.TIME_TOKENS[post_tuning_grid.TUNE_IDX==curr_ti].values[0]
    curr_output_activation = post_tuning_grid.OUTPUT_ACTIVATION[post_tuning_grid.TUNE_IDX==curr_ti].values[0]

    # Retrofit dataframe
    curr_token_indices = curr_token_indices.rename(columns={'VocabTimeFromAdmIndex':'VocabDaysSinceAdmIndex'})

    # Format time tokens of index sets based on current tuning configuration
    curr_token_indices,_ = format_time_tokens(curr_token_indices,curr_time_tokens,False)

    # Add GOSE scores to testing sets
    curr_token_indices = pd.merge(curr_token_indices,study_GUPIs,how='left',on='GUPI')

    # Create PyTorch Dataset objects
    test_Dataset = DYN_ALL_PREDICTOR_SET(curr_token_indices,curr_output_activation)

    # Create PyTorch DataLoader objects
    curr_test_DL = DataLoader(test_Dataset,
                                batch_size=len(test_Dataset),
                                shuffle=False,
                                collate_fn=collate_batch)

    # Load current, best-trained model from checkpoint
    best_model = GOSE_model.load_from_checkpoint(token_info_df.CKPT_FILE[array_task_id])
    best_model = copy.deepcopy(best_model)
    best_model.eval()

    # Add non-static token mask over embedding layer weights
    best_model.embedX.weight.detach()[nonstatic_mask,:] = 0.0
    best_model.embedW.weight.detach()[nonstatic_mask,:] = 0.0

    # Calculate uncalibrated testing set
    with torch.no_grad():
        for i, (curr_test_label_list, curr_test_idx_list, curr_test_bin_offsets, curr_test_gupi_offsets, curr_test_gupis) in enumerate(curr_test_DL):
            (test_yhat, out_test_gupi_offsets) = best_model(curr_test_idx_list, curr_test_bin_offsets, curr_test_gupi_offsets)
            curr_test_labels = torch.cat([curr_test_label_list],dim=0).cpu().numpy()
            if curr_output_activation == 'softmax': 
                curr_test_logits = torch.cat([test_yhat.detach()],dim=0).cpu().numpy()
                curr_test_probs = pd.DataFrame(F.softmax(torch.tensor(curr_test_logits)).cpu().numpy(),columns=['Pr(GOSE=1)','Pr(GOSE=2/3)','Pr(GOSE=4)','Pr(GOSE=5)','Pr(GOSE=6)','Pr(GOSE=7)','Pr(GOSE=8)'])
                curr_test_preds = pd.DataFrame(curr_test_logits,columns=['z_GOSE=1','z_GOSE=2/3','z_GOSE=4','z_GOSE=5','z_GOSE=6','z_GOSE=7','z_GOSE=8'])
                curr_test_preds = pd.concat([curr_test_preds,curr_test_probs], axis=1)
                curr_test_preds['TrueLabel'] = curr_test_labels
            else:
                raise ValueError("Invalid output layer type. Must be 'softmax' or 'sigmoid'")
            curr_test_preds.insert(loc=0, column='GUPI', value=curr_test_gupis)        
            curr_test_preds['TUNE_IDX'] = curr_ti
    curr_test_preds['WindowIdx'] = curr_test_preds.groupby('GUPI').cumcount(ascending=True)+1

    # Load current partition's uncalibrated validation set predictions
    curr_val_preds = pd.read_csv(os.path.join(token_info_df.CKPT_FILE[array_task_id].split('epoch=')[0],'uncalibrated_val_predictions.csv'))
    curr_val_preds['WindowIdx'] = curr_val_preds.groupby('GUPI').cumcount(ascending=True)+1

    # Extract names of important columns
    logit_cols = [col for col in curr_val_preds if col.startswith('z_GOSE=')]
    prob_cols = [col for col in curr_val_preds if col.startswith('Pr(GOSE=')]
    
    # Create lists to store calibrated predictions
    calibrated_val_preds = []
    calibrated_test_preds = []
    
    # Add predictions above window index limit to lists
    calibrated_val_preds.append(curr_val_preds[curr_val_preds.WindowIdx >= 4].reset_index(drop=True))
    calibrated_test_preds.append(curr_test_preds[curr_test_preds.WindowIdx >= 4].reset_index(drop=True))

    # Learn calibration parameters from validation set predictions
    for curr_wi in range(1,4):
        
        # Extract predictions of current window index
        curr_wi_val_preds = curr_val_preds[curr_val_preds.WindowIdx == curr_wi].reset_index(drop=True)
        curr_wi_test_preds = curr_test_preds[curr_test_preds.WindowIdx == curr_wi].reset_index(drop=True)
        
        # Extract current calibration configurations
        curr_optimization = 'nominal'
        curr_calibration = 'vector'
        
        if curr_calibration == 'vector':
            
            if curr_optimization == 'ordinal':
                thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
                for thresh in range(1,len(prob_cols)):
                    cols_gt = prob_cols[thresh:]
                    prob_gt = curr_wi_val_preds[cols_gt].sum(1).values
                    gt = (curr_wi_val_preds['TrueLabel'] >= thresh).astype(int).values
                    curr_wi_val_preds['Pr('+thresh_labels[thresh-1]+')'] = prob_gt
                    curr_wi_val_preds[thresh_labels[thresh-1]] = gt
                    
            scale_object = VectorScaling(curr_wi_val_preds)
            scale_object.set_vector(curr_optimization)
            with torch.no_grad():
                opt_vector = scale_object.vector.detach().data
                opt_biases = scale_object.biases.detach().data
            
            calib_test_logits = torch.matmul(torch.tensor(curr_wi_test_preds[logit_cols].values,dtype=torch.float32),torch.diag_embed(opt_vector.squeeze(1))) + opt_biases.squeeze(1)
            calib_test_probs = F.softmax(calib_test_logits)
            calib_test_preds = pd.DataFrame(torch.cat([calib_test_logits,calib_test_probs],1).numpy(),columns=logit_cols+prob_cols)
            calib_test_preds.insert(loc=0, column='GUPI', value=curr_wi_test_preds['GUPI'])
            calib_test_preds['TrueLabel'] = curr_wi_test_preds['TrueLabel']
            calib_test_preds['TUNE_IDX'] = curr_ti
            calib_test_preds['WindowIdx'] = curr_wi
            calibrated_test_preds.append(calib_test_preds)

    # Concatenate and sort calibrated predictions
    calibrated_test_preds = pd.concat(calibrated_test_preds,ignore_index=True).sort_values(by=['GUPI','WindowIdx'],ignore_index=True)

    # Add repeat and fold information of current partition
    calibrated_test_preds['REPEAT'] = token_info_df.REPEAT[array_task_id]
    calibrated_test_preds['FOLD'] = token_info_df.FOLD[array_task_id]

    # Define tuning configuration directory in which to save calibrated, static-only predictions
    tune_dir = token_info_df.CKPT_FILE[array_task_id].split('epoch=')[0]

    # Save calibrated, static-only predictions in the proper directory
    calibrated_test_preds.to_pickle(os.path.join(tune_dir,'calibrated_static_only_test_predictions.pkl'))

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])
    main(array_task_id)