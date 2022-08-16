#### Master Script 4a: Train dynamic all-predictor-based models ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Create grid of training combinations
# III. Train APM_deep model based on provided hyperparameter row index

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
from classes.calibration import TemperatureScaling, VectorScaling
from functions.model_building import collate_batch, format_tokens, format_time_tokens, T_scaling, vector_scaling
from functions.analysis import calc_ECE, calc_MCE, calc_val_ORC, calc_thresh_calibration
from models.dynamic_APM import GOSE_model

# Set version code
VERSION = 'v7-0'

# Initialise model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION
os.makedirs(model_dir,exist_ok=True)

# Initialise directory in which tokens are stored
tokens_dir = '/home/sb2406/rds/hpc-work/tokens'

# Load cross-validation split information
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Isolate partitions
partitions = cv_splits[['FOLD']].drop_duplicates().reset_index(drop=True)

# II. Create grid of training combinations
# If tuning grid doesn't exist, create it
# if not os.path.exists(os.path.join(model_dir,'tuning_grid.csv')):

#     # Create parameters for training token models
#     tuning_parameters = {'WINDOW_LIMIT':[12,24,84,'None'],
#                          'TIME_TOKENS':['None','DSA_only','TOD_only','Both'],
#                          'RNN_TYPE':['LSTM','GRU'],
#                          'LATENT_DIM':[32,64,128],
#                          'HIDDEN_DIM':[32,64,128],
#                          'TOKEN_CUTS':[20],
#                          'EMBED_DROPOUT':[.2],
#                          'RNN_LAYERS':[1],
#                          'NUM_EPOCHS':[30],
#                          'ES_PATIENCE':[10],
#                          'IMBALANCE_CORRECTION':['weights'],
#                          'OUTPUT_ACTIVATION':['softmax'],
#                          'LEARNING_RATE':[0.001],
#                          'BATCH_SIZE':[1]}
    
#     # Convert parameter dictionary to dataframe
#     tuning_grid = pd.DataFrame([row for row in itertools.product(*tuning_parameters.values())],columns=tuning_parameters.keys())
    
#     # Assign tuning indices
#     tuning_grid['TUNE_IDX'] = list(range(1,tuning_grid.shape[0]+1))
    
#     # Reorder tuning grid columns
#     tuning_grid = tuning_grid[['TUNE_IDX','WINDOW_LIMIT','TIME_TOKENS','LATENT_DIM','HIDDEN_DIM','TOKEN_CUTS','RNN_TYPE','EMBED_DROPOUT','RNN_LAYERS','NUM_EPOCHS','ES_PATIENCE','IMBALANCE_CORRECTION','OUTPUT_ACTIVATION','LEARNING_RATE','BATCH_SIZE']].reset_index(drop=True)
    
#     # Expand tuning grid per cross-validation folds
#     partitions['key'] = 1
#     tuning_grid['key'] = 1
#     tuning_grid = tuning_grid.merge(partitions,how='outer',on='key').drop(columns='key').reset_index(drop=True)

#     # Save tuning grid to model directory
#     tuning_grid.to_csv(os.path.join(model_dir,'tuning_grid.csv'),index=False)

# else:
    
#     # Load optimised tuning grid
#     tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))
tuning_grid = pd.read_pickle('remaining_tuning_grid.pkl')

### III. Train dynamic APM model based on provided hyperparameter row index
# Argument-induced training functions
def main(array_task_id):
    
    # Extract current tuning grid parameters related to cross-validation and token preparation
    curr_fold = tuning_grid.FOLD[array_task_id]
    curr_batch_size = tuning_grid.BATCH_SIZE[array_task_id]
    curr_window_limit = tuning_grid.WINDOW_LIMIT[array_task_id]
    curr_time_tokens = tuning_grid.TIME_TOKENS[array_task_id]
    curr_tune_idx = tuning_grid.TUNE_IDX[array_task_id]
    curr_window_dur = 2
    
    # Create a directory for the current fold
    fold_dir = os.path.join(model_dir,'fold'+str(curr_fold).zfill(int(np.log10(tuning_grid.FOLD.max()))+1))
    os.makedirs(fold_dir,exist_ok=True)
    
    # Create a directory for the current tuning index
    tune_dir = os.path.join(fold_dir,'tune'+str(curr_tune_idx).zfill(4))
    os.makedirs(tune_dir,exist_ok = True)
    
    # Initialize a variable to store the token subdirectory of the current fold
    token_fold_dir = os.path.join(tokens_dir,'fold'+str(curr_fold).zfill(int(np.log10(tuning_grid.FOLD.max()))+1))
    
    # Load current token-indexed training and testing sets
    training_set = pd.read_pickle(os.path.join(token_fold_dir,'training_indices.pkl'))
    validation_set = pd.read_pickle(os.path.join(token_fold_dir,'validation_indices.pkl'))
    testing_set = pd.read_pickle(os.path.join(token_fold_dir,'testing_indices.pkl'))
    
    # Define the limit of windows for model training (1 WINDOW = 1/12 HOURS)
    if curr_window_limit != 'None':
        training_set = training_set[training_set.WindowIdx <= int(curr_window_limit)].sort_values(by=['GUPI','WindowIdx'],ignore_index=True)
    
    # Format time tokens of index sets based on current tuning configuration
    training_set,time_tokens_mask = format_time_tokens(training_set,curr_time_tokens,True)
    validation_set,_ = format_time_tokens(validation_set,curr_time_tokens,False)
    testing_set,_ = format_time_tokens(testing_set,curr_time_tokens,False)

    # Load current token dictionary
    curr_vocab = cp.load(open(os.path.join(token_fold_dir,'token_dictionary.pkl'),"rb"))
        
    # Unique GUPI-GOSE combinations
    study_GUPIs = cv_splits[['GUPI','GOSE']].drop_duplicates()
    
    # Add GOSE scores to training and testing sets
    training_set = pd.merge(training_set,study_GUPIs,how='left',on='GUPI')
    validation_set = pd.merge(validation_set,study_GUPIs,how='left',on='GUPI')
    testing_set = pd.merge(testing_set,study_GUPIs,how='left',on='GUPI')

    # Create PyTorch Dataset objects
    train_Dataset = DYN_ALL_PREDICTOR_SET(training_set,tuning_grid.OUTPUT_ACTIVATION[array_task_id])
    val_Dataset = DYN_ALL_PREDICTOR_SET(validation_set,tuning_grid.OUTPUT_ACTIVATION[array_task_id])
    test_Dataset = DYN_ALL_PREDICTOR_SET(testing_set,tuning_grid.OUTPUT_ACTIVATION[array_task_id])

    # Create PyTorch DataLoader objects
    curr_train_DL = DataLoader(train_Dataset,
                               batch_size=int(curr_batch_size),
                               shuffle=True,
                               collate_fn=collate_batch)
    
    curr_val_DL = DataLoader(val_Dataset,
                             batch_size=len(val_Dataset), 
                             shuffle=False,
                             collate_fn=collate_batch)
    
    curr_test_DL = DataLoader(test_Dataset,
                              batch_size=len(test_Dataset),
                              shuffle=False,
                              collate_fn=collate_batch)
    
    # Initialize current model class based on hyperparameter selections
    model = GOSE_model(len(curr_vocab),
                       tuning_grid.LATENT_DIM[array_task_id],
                       tuning_grid.EMBED_DROPOUT[array_task_id],
                       tuning_grid.RNN_TYPE[array_task_id],
                       tuning_grid.HIDDEN_DIM[array_task_id],
                       tuning_grid.RNN_LAYERS[array_task_id],
                       tuning_grid.OUTPUT_ACTIVATION[array_task_id],
                       tuning_grid.LEARNING_RATE[array_task_id],
                       True,
                       train_Dataset.y,
                       time_tokens_mask+[0])
    
    early_stop_callback = EarlyStopping(
        monitor='val_ORC',
        patience=tuning_grid.ES_PATIENCE[array_task_id],
        mode='max'
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_ORC',
        dirpath=tune_dir,
        filename='{epoch:02d}-{val_ORC:.2f}',
        save_top_k=1,
        mode='max'
    )
      
    csv_logger = pl.loggers.CSVLogger(save_dir=fold_dir,name='tune'+str(curr_tune_idx).zfill(4))

    trainer = pl.Trainer(gpus = 1,
                         accelerator='gpu',
                         logger = csv_logger,
                         max_epochs = tuning_grid.NUM_EPOCHS[array_task_id],
                         enable_progress_bar = True,
                         enable_model_summary = True,
                         callbacks=[early_stop_callback,checkpoint_callback])
    
    trainer.fit(model=model,train_dataloaders=curr_train_DL,val_dataloaders=curr_val_DL)
    
    best_model = GOSE_model.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()
    
    ## Calculate and save uncalibrated validation set
    with torch.no_grad():
        for i, (curr_val_label_list, curr_val_idx_list, curr_val_bin_offsets, curr_val_gupi_offsets, curr_val_gupis) in enumerate(curr_val_DL):
            (val_yhat, out_val_gupi_offsets) = best_model(curr_val_idx_list, curr_val_bin_offsets, curr_val_gupi_offsets)
            curr_val_labels = torch.cat([curr_val_label_list],dim=0).cpu().numpy()
            if tuning_grid.OUTPUT_ACTIVATION[array_task_id] == 'softmax': 
                curr_val_logits = torch.cat([val_yhat.detach()],dim=0).cpu().numpy()
                curr_val_probs = pd.DataFrame(F.softmax(torch.tensor(curr_val_logits)).cpu().numpy(),columns=['Pr(GOSE=1)','Pr(GOSE=2/3)','Pr(GOSE=4)','Pr(GOSE=5)','Pr(GOSE=6)','Pr(GOSE=7)','Pr(GOSE=8)'])
                curr_val_preds = pd.DataFrame(curr_val_logits,columns=['z_GOSE=1','z_GOSE=2/3','z_GOSE=4','z_GOSE=5','z_GOSE=6','z_GOSE=7','z_GOSE=8'])
                curr_val_preds = pd.concat([curr_val_preds,curr_val_probs], axis=1)
                curr_val_preds['TrueLabel'] = curr_val_labels
            else:
                raise ValueError("Invalid output layer type. Must be 'softmax' or 'sigmoid'")
            curr_val_preds.insert(loc=0, column='GUPI', value=curr_val_gupis)        
            curr_val_preds['TUNE_IDX'] = curr_tune_idx
            curr_val_preds.to_csv(os.path.join(tune_dir,'uncalibrated_val_predictions.csv'),index=False)
    curr_val_preds['WindowIdx'] = curr_val_preds.groupby('GUPI').cumcount(ascending=True)+1

    ## Calculate and save uncalibrated testing set
    with torch.no_grad():
        for i, (curr_test_label_list, curr_test_idx_list, curr_test_bin_offsets, curr_test_gupi_offsets, curr_test_gupis) in enumerate(curr_test_DL):
            (test_yhat, out_test_gupi_offsets) = best_model(curr_test_idx_list, curr_test_bin_offsets, curr_test_gupi_offsets)
            curr_test_labels = torch.cat([curr_test_label_list],dim=0).cpu().numpy()
            if tuning_grid.OUTPUT_ACTIVATION[array_task_id] == 'softmax': 
                curr_test_logits = torch.cat([test_yhat.detach()],dim=0).cpu().numpy()
                curr_test_probs = pd.DataFrame(F.softmax(torch.tensor(curr_test_logits)).cpu().numpy(),columns=['Pr(GOSE=1)','Pr(GOSE=2/3)','Pr(GOSE=4)','Pr(GOSE=5)','Pr(GOSE=6)','Pr(GOSE=7)','Pr(GOSE=8)'])
                curr_test_preds = pd.DataFrame(curr_test_logits,columns=['z_GOSE=1','z_GOSE=2/3','z_GOSE=4','z_GOSE=5','z_GOSE=6','z_GOSE=7','z_GOSE=8'])
                curr_test_preds = pd.concat([curr_test_preds,curr_test_probs], axis=1)
                curr_test_preds['TrueLabel'] = curr_test_labels
            else:
                raise ValueError("Invalid output layer type. Must be 'softmax' or 'sigmoid'")
            curr_test_preds.insert(loc=0, column='GUPI', value=curr_test_gupis)        
            curr_test_preds['TUNE_IDX'] = curr_tune_idx
            curr_test_preds.to_csv(os.path.join(tune_dir,'uncalibrated_test_predictions.csv'),index=False)
    curr_test_preds['WindowIdx'] = curr_test_preds.groupby('GUPI').cumcount(ascending=True)+1

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
                prob_cols = [col for col in curr_wi_val_preds if col.startswith('Pr(GOSE=')]
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
                
            calib_val_logits = torch.matmul(torch.tensor(curr_wi_val_preds[logit_cols].values,dtype=torch.float32),torch.diag_embed(opt_vector.squeeze(1))) + opt_biases.squeeze(1)
            calib_val_probs = F.softmax(calib_val_logits)
            calib_val_preds = pd.DataFrame(torch.cat([calib_val_logits,calib_val_probs],1).numpy(),columns=logit_cols+prob_cols)
            calib_val_preds.insert(loc=0, column='GUPI', value=curr_wi_val_preds['GUPI'])
            calib_val_preds['TrueLabel'] = curr_wi_val_preds['TrueLabel']
            calib_val_preds['TUNE_IDX'] = curr_tune_idx
            calib_val_preds['WindowIdx'] = curr_wi
            calibrated_val_preds.append(calib_val_preds)
            
            calib_test_logits = torch.matmul(torch.tensor(curr_wi_test_preds[logit_cols].values,dtype=torch.float32),torch.diag_embed(opt_vector.squeeze(1))) + opt_biases.squeeze(1)
            calib_test_probs = F.softmax(calib_test_logits)
            calib_test_preds = pd.DataFrame(torch.cat([calib_test_logits,calib_test_probs],1).numpy(),columns=logit_cols+prob_cols)
            calib_test_preds.insert(loc=0, column='GUPI', value=curr_wi_test_preds['GUPI'])
            calib_test_preds['TrueLabel'] = curr_wi_test_preds['TrueLabel']
            calib_test_preds['TUNE_IDX'] = curr_tune_idx
            calib_test_preds['WindowIdx'] = curr_wi
            calibrated_test_preds.append(calib_test_preds)
    
    # Concatenate and sort calibrated predictions
    calibrated_val_preds = pd.concat(calibrated_val_preds,ignore_index=True).sort_values(by=['GUPI','WindowIdx'],ignore_index=True)
    calibrated_test_preds = pd.concat(calibrated_test_preds,ignore_index=True).sort_values(by=['GUPI','WindowIdx'],ignore_index=True)
    
    # Save calibrated predictions
    calibrated_val_preds.to_csv(os.path.join(tune_dir,'calibrated_val_predictions.csv'),index=False)
    calibrated_test_preds.to_csv(os.path.join(tune_dir,'calibrated_test_predictions.csv'),index=False)
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)