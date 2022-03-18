#### Master Script 14a: Retrain dynamic all-predictor-based models to combat overfitting ####
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
from functions.model_building import collate_batch, format_tokens
from models.dynamic_APM import GOSE_model

# Set version code
VERSION = 'v4-0'

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

    # Load the optimised tuning grid from v1-0
    tuning_grid = pd.read_csv('/home/sb2406/rds/hpc-work/model_outputs/v1-0/tuning_grid.csv')
    tuning_grid = tuning_grid[tuning_grid.OUTPUT_ACTIVATION == 'softmax']

    # Change embedding layer dropout to 0.2
    tuning_grid.EMBED_DROPOUT = .2

    # Change number of maximum epochs to 30
    tuning_grid.NUM_EPOCHS = 30
    
    # Remove existing batch size and tuning index column
    tuning_grid = tuning_grid.drop(columns='BATCH_SIZE')
    tuning_grid = tuning_grid.drop(columns='tune_idx')
    
    # Add hyperparameter for window size (in hours)
    tuning_grid['key']=1

    # Identify unique tuning configuration combinations
    uniq_window_limit = pd.DataFrame({'WINDOW_LIMIT':[84,'None'],'key':1})
    uniq_batch_sizes = pd.DataFrame({'BATCH_SIZE':[1,4,32,128],'key':1})
    tuning_combos_df = pd.merge(uniq_window_limit,uniq_batch_sizes,how='outer',on='key')
    tuning_combos_df['tune_idx'] = list(range(1,(tuning_combos_df.shape[0]+1)))
    uniq_splits = cv_splits[['repeat','fold']].drop_duplicates().reset_index(drop=True)
    uniq_splits['key'] = 1
    cv_tune_combos = pd.merge(uniq_splits,tuning_combos_df,how='outer',on='key')
    
    # Combine tuning grid with unique adm/disch-CV combinations
    tuning_grid = pd.merge(tuning_grid,cv_tune_combos,how='outer',on='key').drop(columns='key').reset_index(drop=True)
    
    # Reorder tuning grid columns and save in model directory
    tuning_grid = tuning_grid[['tune_idx','WINDOW_LIMIT','BATCH_SIZE','RNN_TYPE','LATENT_DIM','EMBED_DROPOUT','HIDDEN_DIM','RNN_LAYERS','NUM_EPOCHS','ES_PATIENCE','IMBALANCE_CORRECTION','OUTPUT_ACTIVATION','LEARNING_RATE','repeat','fold']].reset_index(drop=True)
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
    curr_window_dur = 2
    curr_tune_idx = tuning_grid.tune_idx[array_task_id]
    
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
    training_set = pd.read_pickle('/home/sb2406/rds/hpc-work/tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/from_adm_training_indices.pkl')
    testing_set = pd.read_pickle('/home/sb2406/rds/hpc-work/tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/from_adm_testing_indices.pkl')
    
    # Remove set rows with missing index values
    training_set = training_set[~training_set.VocabIndex.isna()].reset_index(drop=True)
    testing_set = testing_set[~testing_set.VocabIndex.isna()].reset_index(drop=True)
    
    # Define the limit of windows for model training (1 WINDOW = 1/12 HOURS)
    if curr_window_limit == 'None':
        WINDOW_LIMIT = max(training_set.WindowIdx.max(),testing_set.WindowIdx.max())
    else:
        WINDOW_LIMIT = int(curr_window_limit)
    
    # Format tokens based on `curr_window_dur` and `WINDOW_LIMIT`
    training_set = format_tokens(training_set,WINDOW_LIMIT,'adm',curr_window_dur)
    testing_set = format_tokens(testing_set,WINDOW_LIMIT,'adm',curr_window_dur)
    
    # Load current token dictionary
    curr_vocab = cp.load(open('/home/sb2406/rds/hpc-work/tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/from_adm_token_dictionary.pkl',"rb"))
        
    # Unique GUPI-GOSE combinations
    study_GUPIs = cv_splits[['GUPI','GOSE']].drop_duplicates()
    
    # Add GOSE scores to training and testing sets
    training_set = pd.merge(training_set,study_GUPIs,how='left',on='GUPI')
    testing_set = pd.merge(testing_set,study_GUPIs,how='left',on='GUPI')
    
    # Set aside 15% of the training set for validation, independent of the final testing set
    full_training_GOSE = training_set[['GUPI','GOSE']].drop_duplicates(ignore_index=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15)
    for train_index, val_index in sss.split(full_training_GOSE.drop(columns='GOSE'),full_training_GOSE.GOSE):
        training_GUPIs, val_GUPIs = full_training_GOSE.GUPI[train_index], full_training_GOSE.GUPI[val_index]
        val_set = training_set[training_set.GUPI.isin(val_GUPIs)].reset_index(drop=True)
        training_set = training_set[training_set.GUPI.isin(training_GUPIs)].reset_index(drop=True)    
    cp.dump(sss, open(os.path.join(tune_dir,'val_set_splitter.pkl'), "wb"))
    
    # Create PyTorch Dataset objects
    train_Dataset = DYN_ALL_PREDICTOR_SET(training_set,tuning_grid.OUTPUT_ACTIVATION[array_task_id])
    val_Dataset = DYN_ALL_PREDICTOR_SET(val_set,tuning_grid.OUTPUT_ACTIVATION[array_task_id])
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
                       train_Dataset.y)
    
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
    
    # Save validation set probabilities
    for i, (curr_label_list, curr_idx_list, curr_bin_offsets, curr_gupi_offsets, curr_gupis) in enumerate(curr_val_DL):

        (yhat, out_gupi_offsets) = best_model(curr_idx_list, curr_bin_offsets, curr_gupi_offsets)
        curr_val_labels = torch.cat([curr_label_list],dim=0).cpu().numpy()
        
        if tuning_grid.OUTPUT_ACTIVATION[0] == 'softmax': 

            curr_val_probs = torch.cat([F.softmax(yhat.detach())],dim=0).cpu().numpy()
            curr_val_preds = pd.DataFrame(curr_val_probs,columns=['Pr(GOSE=1)','Pr(GOSE=2/3)','Pr(GOSE=4)','Pr(GOSE=5)','Pr(GOSE=6)','Pr(GOSE=7)','Pr(GOSE=8)'])
            curr_val_preds['TrueLabel'] = curr_val_labels

        elif tuning_grid.OUTPUT_ACTIVATION[0] == 'sigmoid': 

            curr_val_probs = torch.cat([F.sigmoid(yhat.detach())],dim=0).cpu().numpy()
            curr_val_probs = pd.DataFrame(curr_val_probs,columns=['Pr(GOSE>1)','Pr(GOSE>3)','Pr(GOSE>4)','Pr(GOSE>5)','Pr(GOSE>6)','Pr(GOSE>7)'])
            curr_val_labels = pd.DataFrame(curr_val_labels,columns=['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7'])
            curr_val_preds = pd.concat([curr_val_probs,curr_val_labels],axis = 1)

        else:
            raise ValueError("Invalid output layer type. Must be 'softmax' or 'sigmoid'")

        curr_val_preds.insert(loc=0, column='GUPI', value=curr_gupis)        
        curr_val_preds['tune_idx'] = curr_tune_idx

        curr_val_preds.to_csv(os.path.join(tune_dir,'val_predictions.csv'),index=False)
        
    best_model.eval()
        
    # Save testing set probabilities
    for i, (curr_label_list, curr_idx_list, curr_bin_offsets, curr_gupi_offsets, curr_gupis) in enumerate(curr_test_DL):

        (yhat, out_gupi_offsets) = best_model(curr_idx_list, curr_bin_offsets, curr_gupi_offsets)
        curr_test_labels = torch.cat([curr_label_list],dim=0).cpu().numpy()
        
        if tuning_grid.OUTPUT_ACTIVATION[0] == 'softmax': 

            curr_test_probs = torch.cat([F.softmax(yhat.detach())],dim=0).cpu().numpy()
            curr_test_preds = pd.DataFrame(curr_test_probs,columns=['Pr(GOSE=1)','Pr(GOSE=2/3)','Pr(GOSE=4)','Pr(GOSE=5)','Pr(GOSE=6)','Pr(GOSE=7)','Pr(GOSE=8)'])
            curr_test_preds['TrueLabel'] = curr_test_labels

        elif tuning_grid.OUTPUT_ACTIVATION[0] == 'sigmoid': 

            curr_test_probs = torch.cat([F.sigmoid(yhat.detach())],dim=0).cpu().numpy()
            curr_test_probs = pd.DataFrame(curr_test_probs,columns=['Pr(GOSE>1)','Pr(GOSE>3)','Pr(GOSE>4)','Pr(GOSE>5)','Pr(GOSE>6)','Pr(GOSE>7)'])
            curr_test_labels = pd.DataFrame(curr_test_labels,columns=['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7'])
            curr_test_preds = pd.concat([curr_test_probs,curr_test_labels],axis = 1)

        else:
            raise ValueError("Invalid output layer type. Must be 'softmax' or 'sigmoid'")

        curr_test_preds.insert(loc=0, column='GUPI', value=curr_gupis)        
        curr_test_preds['tune_idx'] = curr_tune_idx

        curr_test_preds.to_csv(os.path.join(tune_dir,'test_predictions.csv'),index=False)
        
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)