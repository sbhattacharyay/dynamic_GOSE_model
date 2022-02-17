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
from functions.model_building import collate_batch
from models.dynamic_APM import GOSE_model

# Set version code
VERSION = 'v1-0'

# Initialise model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION
os.makedirs(model_dir,exist_ok=True)

# Load cross-validation split information
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Load the optimised tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))
tuning_grid = tuning_grid[tuning_grid.OUTPUT_ACTIVATION == 'softmax']

# Define the limit of windows for model training (1 WINDOW = 1/12 HOURS)
WINDOW_LIMIT = 84

### II. Create grid of training combinations
# Identify unique adm/disch-CV combinations
uniq_splits = cv_splits[['repeat','fold']].drop_duplicates().reset_index(drop=True)
uniq_splits['key'] = 1
uniq_adm_disch = pd.DataFrame({'adm_or_disch':['adm','disch'],'key':1})
cv_ad_combos = pd.merge(uniq_splits,uniq_adm_disch,how='outer',on='key').drop(columns='key')

### III. Train
def main(array_task_id):
    
    # Extract current repeat, fold, and adm/disch information
    curr_repeat = cv_ad_combos.repeat[array_task_id]
    curr_fold = cv_ad_combos.fold[array_task_id]
    curr_adm_or_disch = cv_ad_combos.adm_or_disch[array_task_id]
    
    # Create a directory for the current repeat
    repeat_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(int(np.log10(cv_splits.repeat.max()))+1))
    os.makedirs(repeat_dir,exist_ok=True)
    
    # Create a directory for the current fold
    fold_dir = os.path.join(repeat_dir,'fold'+str(curr_fold).zfill(int(np.log10(cv_splits.fold.max()))+1))
    os.makedirs(fold_dir,exist_ok=True)
    
    # Create a directory for the current adm/disch
    adm_disch_dir = os.path.join(fold_dir,curr_adm_or_disch)
    os.makedirs(adm_disch_dir,exist_ok = True)
       
    # Load current token-indexed training and testing sets
    training_set = pd.read_pickle('/home/sb2406/rds/hpc-work/tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/from_'+curr_adm_or_disch+'_training_indices.pkl')
    testing_set = pd.read_pickle('/home/sb2406/rds/hpc-work/tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/from_'+curr_adm_or_disch+'_testing_indices.pkl')
    
    # Based on `curr_adm_or_disch` and WINDOW_LIMIT, refine dataset
    if curr_adm_or_disch == 'adm':
        
        # Extract tokens up to window limit after ICU admission
        training_set = training_set[training_set.WindowIdx <= WINDOW_LIMIT].sort_values(by=['GUPI','WindowIdx'],ignore_index=True)
        testing_set = testing_set[testing_set.WindowIdx <= WINDOW_LIMIT].sort_values(by=['GUPI','WindowIdx'],ignore_index=True)

    elif curr_adm_or_disch == 'disch':
        
        # Combine all tokens up to the window limit before ICU discharge
        comb_lim_training = training_set[training_set.WindowIdx >= WINDOW_LIMIT].groupby('GUPI')['VocabIndex'].apply(list).reset_index(name='VocabIndex')
        comb_lim_training['VocabIndex'] = comb_lim_training['VocabIndex'].apply(lambda x: list(np.unique([item for sublist in x for item in sublist])))
        full_lim_training = training_set[training_set.WindowIdx == WINDOW_LIMIT].drop(columns='VocabIndex').merge(comb_lim_training,on='GUPI',how='left')
        training_set = pd.concat([training_set[training_set.WindowIdx < WINDOW_LIMIT],full_lim_training],ignore_index=True).sort_values(by=['GUPI','WindowIdx'],ascending=[True,False],ignore_index=True)
        
        # Combine all tokens up to the window limit before ICU discharge
        comb_lim_testing = testing_set[testing_set.WindowIdx >= WINDOW_LIMIT].groupby('GUPI')['VocabIndex'].apply(list).reset_index(name='VocabIndex')
        comb_lim_testing['VocabIndex'] = comb_lim_testing['VocabIndex'].apply(lambda x: list(np.unique([item for sublist in x for item in sublist])))
        full_lim_testing = testing_set[testing_set.WindowIdx == WINDOW_LIMIT].drop(columns='VocabIndex').merge(comb_lim_testing,on='GUPI',how='left')
        testing_set = pd.concat([testing_set[testing_set.WindowIdx < WINDOW_LIMIT],full_lim_testing],ignore_index=True).sort_values(by=['GUPI','WindowIdx'],ascending=[True,False],ignore_index=True)
        
    else:
        raise ValueError('curr_adm_or_disch must be "adm" or "disch"')
    
    # Load current token dictionary
    curr_vocab = cp.load(open('/home/sb2406/rds/hpc-work/tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/from_'+curr_adm_or_disch+'_token_dictionary.pkl',"rb"))
        
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
    cp.dump(sss, open(os.path.join(adm_disch_dir,'val_set_splitter.pkl'), "wb"))
    
    # Create PyTorch Dataset objects
    train_Dataset = DYN_ALL_PREDICTOR_SET(training_set,tuning_grid.OUTPUT_ACTIVATION[0])
    val_Dataset = DYN_ALL_PREDICTOR_SET(val_set,tuning_grid.OUTPUT_ACTIVATION[0])
    test_Dataset = DYN_ALL_PREDICTOR_SET(testing_set,tuning_grid.OUTPUT_ACTIVATION[0])
    
    # Create PyTorch DataLoader objects
    curr_train_DL = DataLoader(train_Dataset,
                               batch_size=int(tuning_grid.BATCH_SIZE[0]),
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
                       tuning_grid.LATENT_DIM[0],
                       tuning_grid.EMBED_DROPOUT[0],
                       tuning_grid.RNN_TYPE[0],
                       tuning_grid.HIDDEN_DIM[0],
                       tuning_grid.RNN_LAYERS[0],
                       tuning_grid.OUTPUT_ACTIVATION[0],
                       tuning_grid.LEARNING_RATE[0],
                       True,
                       train_Dataset.y)
    
    early_stop_callback = EarlyStopping(
        monitor='val_ORC',
        patience=tuning_grid.ES_PATIENCE[0],
        mode='max'
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_ORC',
        dirpath=adm_disch_dir,
        filename='{epoch:02d}-{val_ORC:.2f}',
        save_top_k=1,
        mode='max'
    )
    
    csv_logger = pl.loggers.CSVLogger(save_dir=fold_dir,name=curr_adm_or_disch)

    trainer = pl.Trainer(gpus = -1,
                         accelerator='gpu',
                         strategy = 'ddp',
                         logger = csv_logger,
                         max_epochs = 20,
                         enable_progress_bar = True,
                         enable_model_summary = True,
                         callbacks=[early_stop_callback,checkpoint_callback])
    
    trainer.fit(model,curr_train_DL,curr_val_DL)
    
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
        curr_val_preds['adm_or_disch'] = curr_adm_or_disch

        curr_val_preds.to_csv(os.path.join(adm_disch_dir,'val_predictions.csv'),index=False)
        
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
        curr_test_preds['adm_or_disch'] = curr_adm_or_disch

        curr_test_preds.to_csv(os.path.join(adm_disch_dir,'test_predictions.csv'),index=False)
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)