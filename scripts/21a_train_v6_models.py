#### Master Script 21a: Retrain dynamic all-predictor-based models to combat overfitting ####
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
from functions.model_building import collate_batch, format_tokens, format_time_tokens, T_scaling, vector_scaling
from models.dynamic_APM import GOSE_model

# Set version code
VERSION = 'v6-0'

# Initialise model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION
os.makedirs(model_dir,exist_ok=True)

# Load cross-validation split information
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Isolate partitions of first repeat
cv_splits = cv_splits[cv_splits.repeat == 1].reset_index(drop=True)

# II. Create grid of training combinations
# If bootstrapping resamples don't exist, create them
if not os.path.exists(os.path.join(model_dir,'tuning_grid.csv')):

    # Create parameters for training differential token models
    diff_tuning_parameters = {'STRATEGY':['diff'],
                              'WINDOW_LIMIT':[84,'None'],
                              'TIME_TOKENS':['None','TFA_only','TOD_only','Both'],
                              'LATENT_DIM':[32,64,128],
                              'HIDDEN_DIM':[32,64,128],
                              'TOKEN_CUTS':[20],
                              'RNN_TYPE':['GRU'],
                              'EMBED_DROPOUT':[.2],
                              'RNN_LAYERS':[1],
                              'NUM_EPOCHS':[30],
                              'ES_PATIENCE':[10],
                              'IMBALANCE_CORRECTION':['weights'],
                              'OUTPUT_ACTIVATION':['softmax'],
                              'LEARNING_RATE':[0.001],
                              'BATCH_SIZE':[1]}
    
    # Convert parameter dictionary to dataframe
    diff_tuning_grid = pd.DataFrame([row for row in itertools.product(*diff_tuning_parameters.values())],columns=diff_tuning_parameters.keys())
    
    # Create parameters for training absolute token models
    abs_tuning_parameters = {'STRATEGY':['abs'],
                             'WINDOW_LIMIT':[12,24,84],
                             'TIME_TOKENS':['None','TFA_only','TOD_only'],
                             'LATENT_DIM':[32,64,128],
                             'HIDDEN_DIM':[32,64,128],
                             'TOKEN_CUTS':[20],
                             'RNN_TYPE':['GRU'],
                             'EMBED_DROPOUT':[.2],
                             'RNN_LAYERS':[1],
                             'NUM_EPOCHS':[30],
                             'ES_PATIENCE':[10],
                             'IMBALANCE_CORRECTION':['weights'],
                             'OUTPUT_ACTIVATION':['softmax'],
                             'LEARNING_RATE':[0.001],
                             'BATCH_SIZE':[1]}
    
    # Convert parameter dictionary to dataframe
    abs_tuning_grid = pd.DataFrame([row for row in itertools.product(*abs_tuning_parameters.values())],columns=abs_tuning_parameters.keys())
    
    # Concatenate tuning grids and assign tuning indices
    tuning_grid = pd.concat([diff_tuning_grid,abs_tuning_grid],ignore_index=True)
    tuning_grid['TUNE_IDX'] = list(range(1,tuning_grid.shape[0]+1))
    
    # Reorder tuning grid columns
    tuning_grid = tuning_grid[['TUNE_IDX','STRATEGY','WINDOW_LIMIT','TIME_TOKENS','LATENT_DIM','HIDDEN_DIM','TOKEN_CUTS','RNN_TYPE','EMBED_DROPOUT','RNN_LAYERS','NUM_EPOCHS','ES_PATIENCE','IMBALANCE_CORRECTION','OUTPUT_ACTIVATION','LEARNING_RATE','BATCH_SIZE']].reset_index(drop=True)
    
    #Expand tuning grid by repeated cross-validation partitions
    uniq_splits = cv_splits[['repeat','fold']].drop_duplicates().reset_index(drop=True)
    uniq_splits['key'] = 1
    tuning_grid['key'] = 1
    tuning_grid = tuning_grid.merge(uniq_splits,how='outer',on='key').drop(columns='key').reset_index(drop=True)
    
    # Save tuning grid to model directory
    tuning_grid.to_csv(os.path.join(model_dir,'tuning_grid.csv'),index=False)

else:
    # Load optimised tuning grid
    tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))
    
### III. Train dynamic APM model based on provided hyperparameter row index
# Argument-induced training functions
def main(array_task_id):
    
    # Extract current repeat, fold, and adm/disch information
    curr_repeat = tuning_grid.repeat[array_task_id]
    curr_fold = tuning_grid.fold[array_task_id]
    curr_batch_size = tuning_grid.BATCH_SIZE[array_task_id]
    curr_window_limit = tuning_grid.WINDOW_LIMIT[array_task_id]
    curr_time_tokens = tuning_grid.TIME_TOKENS[array_task_id]
    curr_window_dur = 2
    curr_tune_idx = tuning_grid.TUNE_IDX[array_task_id]
    curr_strategy = tuning_grid.STRATEGY[array_task_id]
    
    # Create a directory for the current repeat
    repeat_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(2))
    os.makedirs(repeat_dir,exist_ok=True)
    
    # Create a directory for the current fold
    fold_dir = os.path.join(repeat_dir,'fold'+str(curr_fold).zfill(int(np.log10(tuning_grid.fold.max()))+1))
    os.makedirs(fold_dir,exist_ok=True)
    
    # Create a directory for the current tuning index
    tune_dir = os.path.join(fold_dir,'tune'+str(curr_tune_idx).zfill(4))
    os.makedirs(tune_dir,exist_ok = True)
    
    # Load current token-indexed training and testing sets
    training_set = pd.read_pickle('/home/sb2406/rds/hpc-work/tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/from_adm_strategy_'+curr_strategy+'_training_indices.pkl')
    validation_set = pd.read_pickle('/home/sb2406/rds/hpc-work/tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/from_adm_strategy_'+curr_strategy+'_validation_indices.pkl')
    testing_set = pd.read_pickle('/home/sb2406/rds/hpc-work/tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/from_adm_strategy_'+curr_strategy+'_testing_indices.pkl')

    # Define the limit of windows for model training (1 WINDOW = 1/12 HOURS)
    if curr_window_limit != 'None':
        training_set = training_set[training_set.WindowIdx <= int(curr_window_limit)].sort_values(by=['GUPI','WindowIdx'],ignore_index=True)
    
    # Format time tokens of index sets based on current tuning configuration
    training_set,time_tokens_mask = format_time_tokens(training_set,curr_time_tokens,True)
    validation_set,_ = format_time_tokens(validation_set,curr_time_tokens,False)
    testing_set,_ = format_time_tokens(testing_set,curr_time_tokens,False)

    # Load current token dictionary
    curr_vocab = cp.load(open('/home/sb2406/rds/hpc-work/tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/from_adm_strategy_'+curr_strategy+'_token_dictionary.pkl',"rb"))
        
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
        monitor='val_loss',
        patience=tuning_grid.ES_PATIENCE[array_task_id],
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=tune_dir,
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
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

#     ## VECTOR SCALING
#     # Calibrate validation predictions with vector-scaling
#     device = 'cuda:0'
#     vector = nn.Parameter((torch.ones(curr_val_logits.shape[1],1)).cuda())
#     nn.init.xavier_uniform_(vector)
#     biases = nn.Parameter((torch.zeros(curr_val_logits.shape[1],1)).cuda())
#     args = {'vector': vector,'biases':biases}
#     bal_weights = torch.from_numpy(compute_class_weight(class_weight='balanced',
#                                                         classes=np.sort(np.unique(curr_val_labels)),
#                                                         y=curr_val_labels)).type_as(val_yhat).to(device)
#     criterion = nn.CrossEntropyLoss(weight=bal_weights)
#     vectors = []
#     bias_list = []
#     losses = []

#     # Removing strong_wolfe line search results in jump after 50 epochs
#     vector_optimizer = optim.LBFGS([vector,biases], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')
    
#     # Define custom vector optimization function
#     def _vector_eval():
#         loss = criterion(vector_scaling(torch.tensor(curr_val_logits).type_as(val_yhat).to(device), args), torch.tensor(curr_val_labels).type_as(curr_val_label_list).to(device))
#         loss.backward()
#         vectors.append(vector.data)
#         bias_list.append(biases.data)
#         losses.append(loss)
#         return loss
    
#     # Optimize vector and bias value
#     vector_optimizer.step(_vector_eval)
    
#     # Extract optimal vector and optimal bias
#     vector_opt = vector.data
#     bias_opt = biases.data

#     # Scale and save vector-scaled validation set
#     vector_scaled_val_probs = pd.DataFrame(F.softmax(torch.matmul(torch.tensor(curr_val_logits).type_as(val_yhat),torch.diag_embed(vector_opt.squeeze(1))) + bias_opt.squeeze(1)).cpu().numpy(),columns=['Pr(GOSE=1)','Pr(GOSE=2/3)','Pr(GOSE=4)','Pr(GOSE=5)','Pr(GOSE=6)','Pr(GOSE=7)','Pr(GOSE=8)'])
#     vector_scaled_val_preds = pd.DataFrame((torch.matmul(torch.tensor(curr_val_logits).type_as(val_yhat),torch.diag_embed(vector_opt.squeeze(1))) + bias_opt.squeeze(1)).cpu().numpy(),columns=['z_GOSE=1','z_GOSE=2/3','z_GOSE=4','z_GOSE=5','z_GOSE=6','z_GOSE=7','z_GOSE=8'])
#     vector_scaled_val_preds = pd.concat([vector_scaled_val_preds,vector_scaled_val_probs], axis=1)
#     vector_scaled_val_preds['TrueLabel'] = curr_val_labels
#     vector_scaled_val_preds.insert(loc=0, column='GUPI', value=curr_val_gupis)        
#     vector_scaled_val_preds['TUNE_IDX'] = curr_tune_idx
#     vector_scaled_val_preds.to_csv(os.path.join(tune_dir,'vector_scaled_val_predictions.csv'),index=False)
    
#     # Scale and save vector-scaled testing set
#     vector_scaled_test_probs = pd.DataFrame(F.softmax(torch.matmul(torch.tensor(curr_test_logits).type_as(test_yhat),torch.diag_embed(vector_opt.squeeze(1))) + bias_opt.squeeze(1)).cpu().numpy(),columns=['Pr(GOSE=1)','Pr(GOSE=2/3)','Pr(GOSE=4)','Pr(GOSE=5)','Pr(GOSE=6)','Pr(GOSE=7)','Pr(GOSE=8)'])
#     vector_scaled_test_preds = pd.DataFrame((torch.matmul(torch.tensor(curr_test_logits).type_as(test_yhat),torch.diag_embed(vector_opt.squeeze(1))) + bias_opt.squeeze(1)).cpu().numpy(),columns=['z_GOSE=1','z_GOSE=2/3','z_GOSE=4','z_GOSE=5','z_GOSE=6','z_GOSE=7','z_GOSE=8'])
#     vector_scaled_test_preds = pd.concat([vector_scaled_test_preds,vector_scaled_test_probs], axis=1)
#     vector_scaled_test_preds['TrueLabel'] = curr_test_labels
#     vector_scaled_test_preds.insert(loc=0, column='GUPI', value=curr_test_gupis)        
#     vector_scaled_test_preds['TUNE_IDX'] = curr_tune_idx
#     vector_scaled_test_preds.to_csv(os.path.join(tune_dir,'vector_scaled_test_predictions.csv'),index=False)
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)
