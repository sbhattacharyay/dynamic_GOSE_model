#### Master Script 23a: Calculate LBM for dynAPM_DeepMN ####
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

# Import SHAP
import shap
from shap import DeepExplainer

# Custom methods
from classes.datasets import DYN_ALL_PREDICTOR_SET
from models.dynamic_APM import GOSE_model, shap_GOSE_model
from models.LBM import LBM_step1_model
from functions.model_building import format_shap, format_tokens, format_time_tokens

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

# Define a directory for the storage of LBM values
lbm_dir = os.path.join('/home/sb2406/rds/hpc-work/model_interpretations/'+VERSION,'LBM')
os.makedirs(lbm_dir,exist_ok=True)

# Define vector of GOSE thresholds
GOSE_thresholds = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']

### II. Find all top-performing model checkpoint files for SHAP calculation
# Either create or load APM checkpoint information for SHAP value 
if not os.path.exists(os.path.join(lbm_dir,'ckpt_info.pkl')):
    
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
    ckpt_info = ckpt_info.merge(test_splits,how='left',on=['REPEAT','FOLD']).drop(columns='GOSE')

    # Fix analysis to FIRST REPEAT
    ckpt_info = ckpt_info[ckpt_info.REPEAT == 1].reset_index(drop=True)
    ckpt_info['key'] = 1
    
    # Add threshold combinations to checkpoint info dataframe
    threshold_df = pd.DataFrame({'THRESHOLD_IDX':list(range(6)),'key':1})
    ckpt_info = ckpt_info.merge(threshold_df,how='left',on=['key']).drop(columns='key')
    
    # Sort checkpoint dataframe by GUPI, then REPEAT, FOLD
    ckpt_info = ckpt_info.sort_values(by=['GUPI','REPEAT','FOLD','THRESHOLD_IDX','TUNE_IDX','VERSION']).reset_index(drop=True)
    
    # Save model checkpoint information dataframe
    ckpt_info.to_pickle(os.path.join(lbm_dir,'ckpt_info.pkl'))
    
else:
    
    # Read model checkpoint information dataframe
    ckpt_info = pd.read_pickle(os.path.join(lbm_dir,'ckpt_info.pkl'))
    
### III. Calculate SHAP values based on given parameters
def main(array_task_id):

    # Extract current file, repeat, and fold information
    curr_file = ckpt_info.file[array_task_id]
    curr_tune_idx = ckpt_info.TUNE_IDX[array_task_id]
    curr_repeat = ckpt_info.REPEAT[array_task_id]
    curr_fold = ckpt_info.FOLD[array_task_id]
    curr_gupi = ckpt_info.GUPI[array_task_id]
    curr_threshold_idx = ckpt_info.THRESHOLD_IDX[array_task_id]
    
    # Define current fold directory based on current information
    tune_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold),'tune'+str(curr_tune_idx).zfill(4))
    
    # Define and create current LBM sub-directory
    curr_lbm_fold_dir = os.path.join(lbm_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold))
    curr_lbm_dir = os.path.join(curr_lbm_fold_dir)
    os.makedirs(curr_lbm_dir,exist_ok=True)

    # Filter out current tuning directory configuration hyperparameters
    curr_tune_hp = tuning_grid[(tuning_grid.TUNE_IDX == curr_tune_idx)&(tuning_grid.fold == curr_fold)].reset_index(drop=True)
    
    # Extract current testing set for current repeat and fold combination
    token_dir = os.path.join('/home/sb2406/rds/hpc-work/tokens','repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold))
    testing_set = pd.read_pickle(os.path.join(token_dir,'from_adm_strategy_'+curr_tune_hp.STRATEGY[0]+'_testing_indices.pkl'))
    
    # Extract testing set predictions of current GUPI
    testing_set = testing_set[testing_set.GUPI == curr_gupi].reset_index(drop=True)
    
    # Load current token dictionary
    curr_vocab = cp.load(open(os.path.join(token_dir,'from_adm_strategy_'+curr_tune_hp.STRATEGY[0]+'_token_dictionary.pkl'),"rb"))
    unknown_index = curr_vocab['<unk>']
    
    # Format time tokens of index sets based on current tuning configuration
    testing_set,_ = format_time_tokens(testing_set,curr_tune_hp.TIME_TOKENS[0],False)
    testing_set['SeqLength'] = testing_set.VocabIndex.apply(len)
    testing_set['Unknowns'] = testing_set.VocabIndex.apply(lambda x: x.count(unknown_index))    
    testing_set['DischWindowIdx'] = -(testing_set['WindowTotal'] - testing_set['WindowIdx'] + 1)
    
    # Number of columns to add
    cols_to_add = max(testing_set['Unknowns'].max(),1) - 1
    
    # Initialize empty dataframe for multihot encoding of testing set
    multihot_matrix = np.zeros([testing_set.shape[0],len(curr_vocab)+cols_to_add])

    # Encode testing set into multihot encoded matrix
    for i in tqdm(range(testing_set.shape[0])):
        curr_indices = np.array(testing_set.VocabIndex[i])
        if sum(curr_indices == unknown_index) > 1:
            zero_indices = np.where(curr_indices == unknown_index)[0]
            curr_indices[zero_indices[1:]] = [len(curr_vocab) + j for j in range(sum(curr_indices == unknown_index)-1)]
        multihot_matrix[i,curr_indices] = 1
    multihot_matrix = torch.tensor(multihot_matrix).float()
    
    # Create torch DataLoader object for multihot matrix
    patientDL = DataLoader(multihot_matrix,batch_size=len(multihot_matrix),shuffle=False)
    
    # Create vector of target column indices
    class_indices = list(range(7))
    gose_classes = np.sort(test_splits.GOSE.unique())
    
    # Load current pretrained model
    gose_model = GOSE_model.load_from_checkpoint(curr_file)
    gose_model.eval()
    
    # Initialize Step 1 LBM calculation PyTorch Lightning class
    curr_step1_LBM = LBM_step1_model(curr_threshold_idx,gose_model,cols_to_add,unknown_index,5,multihot_matrix.shape[0],.005,.5,.1)
    
    # Initialize callback and logging objects
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(monitor='lr_reductions',stopping_threshold=2,patience=5000,mode='max')
    thresh_dir = os.path.join(curr_lbm_dir,'GUPI_'+curr_gupi+'_THRESHOLDIDX_'+str(curr_threshold_idx))
    os.makedirs(thresh_dir,exist_ok=True)
    csv_logger = pl.loggers.CSVLogger(save_dir=curr_lbm_dir,name='GUPI_'+curr_gupi+'_THRESHOLDIDX_'+str(curr_threshold_idx))
    
    # Set training parameters
    trainer = pl.Trainer(max_epochs = 5000,
                         logger = csv_logger,
                         enable_progress_bar = True,
                         enable_model_summary = True,
                         enable_checkpointing=False,
                         callbacks=[lr_monitor,early_stop_callback])
    
    
    # Train LBM
    trainer.fit(model=curr_step1_LBM,train_dataloaders=patientDL)

    # Perform STEP 2 of LBM optimization
    with torch.no_grad():

        # Initialize step 2 LBM parameters
        eta = 0.5*np.ones_like(multihot_matrix)
        L_min = 100000.00
        reg_2_1 = .0001
        s_min = .05
        max_iter = 2

        for step2_iter in range(max_iter):
            for timestep in tqdm(range(multihot_matrix.shape[0])):

                # Derive unique values from current timestamp in continuous mask
                U_values = np.sort(curr_step1_LBM.get_m().detach()[timestep,:].unique().numpy())

                # Iterate through unique values from current timestamp in continuous mask
                for u in U_values:

                    # Create copy of the eta matrix
                    q_thresholds = eta.copy()

                    # Replace current timestamp values in the eta copy
                    q_thresholds[timestep,:] = u

                    # Create binary mask based on current values
                    curr_Mask = (curr_step1_LBM.get_m().detach().numpy() > q_thresholds).astype(int)

                    # Convert patient matrix to dt form
                    curr_dt_matrix = curr_step1_LBM.pt_to_dt_matrix(multihot_matrix)

                    # Mask the dt matrix and convert to patient matrix
                    curr_pt_matrix = curr_step1_LBM.dt_to_pt_matrix(curr_dt_matrix*torch.from_numpy(curr_Mask))

                    # Weigh embedding layers
                    comb_embedding = curr_step1_LBM.embedX.weight*torch.tile(torch.exp(curr_step1_LBM.embedW.weight),(1,curr_step1_LBM.embedX.weight.shape[1]))
                    curr_embedding_out = F.relu(torch.matmul(curr_pt_matrix,comb_embedding) / torch.tile(curr_pt_matrix.sum(1).unsqueeze(1),(1,comb_embedding.shape[1])))

                    # Obtain RNN output and transform to GOSE space
                    curr_rnn_out, _ = curr_step1_LBM.rnn_module(curr_embedding_out.unsqueeze(1))
                    curr_gose_out = curr_step1_LBM.hidden2gose(curr_rnn_out).squeeze(1)
                    curr_output_vector = F.softmax(curr_gose_out).cumsum(1)[:,curr_threshold_idx]

                    # Calculate current loss value
                    curr_loss = (torch.linalg.vector_norm(curr_output_vector,ord=1)) + (reg_2_1*torch.linalg.matrix_norm(1 - torch.t(torch.from_numpy(curr_Mask).to(torch.float32)),ord=1))

                    # If current loss is less than existing overall loss value, change the eta threshold matrix
                    if float(curr_loss) < L_min:

                        # Replace existing overall loss value
                        L_min = float(curr_loss)

                        # Change eta matrix appropriately
                        eta = q_thresholds.copy()

            # Calculate output vector after current iteration
            curr_Mask = (curr_step1_LBM.get_m().detach().numpy() > eta).astype(int)

            # Convert patient matrix to dt form
            curr_dt_matrix = curr_step1_LBM.pt_to_dt_matrix(multihot_matrix)

            # Mask the dt matrix and convert to patient matrix
            curr_pt_matrix = curr_step1_LBM.dt_to_pt_matrix(curr_dt_matrix*torch.from_numpy(curr_Mask))

            # Weigh embedding layers
            comb_embedding = curr_step1_LBM.embedX.weight*torch.tile(torch.exp(curr_step1_LBM.embedW.weight),(1,curr_step1_LBM.embedX.weight.shape[1]))
            curr_embedding_out = F.relu(torch.matmul(curr_pt_matrix,comb_embedding) / torch.tile(curr_pt_matrix.sum(1).unsqueeze(1),(1,comb_embedding.shape[1])))

            # Obtain RNN output and transform to GOSE space
            curr_rnn_out, _ = curr_step1_LBM.rnn_module(curr_embedding_out.unsqueeze(1))
            curr_gose_out = curr_step1_LBM.hidden2gose(curr_rnn_out).squeeze(1)
            curr_output_vector = F.softmax(curr_gose_out).cumsum(1)[:,curr_threshold_idx]

            # If all output values are below the given threshold, terminate step 2 threshold search
            if (curr_output_vector.numpy() < s_min).all():
                break

        # Extract token labels from vocabulary object
        token_labels = curr_vocab.get_itos() + [curr_vocab.get_itos()[unknown_index]+'_'+str(i+1).zfill(3) for i in range(cols_to_add)]
        token_labels[unknown_index] = token_labels[unknown_index]+'_000'

        # Convert optimal mask to attribution matrix
        LBM_attribution_df = pd.DataFrame(curr_Mask,columns=token_labels)
        LBM_attribution_df['WINDOW_IDX'] = list(range(1,curr_Mask.shape[0]+1))
        LBM_attribution_df = pd.melt(LBM_attribution_df,id_vars='WINDOW_IDX')
        LBM_attribution_df = LBM_attribution_df[LBM_attribution_df.value == 0].reset_index(drop=True).drop(columns='value').rename(columns={'variable':'TOKEN'})
        LBM_attribution_df['THRESHOLD'] = GOSE_thresholds[curr_threshold_idx]
        LBM_attribution_df['GUPI'] = curr_gupi
        LBM_attribution_df['TUNE_IDX'] = curr_tune_idx
        LBM_attribution_df['REPEAT'] = curr_repeat
        LBM_attribution_df['FOLD'] = curr_fold
        
        # Save attribution dataframe to current LBM directory
        LBM_attribution_df.to_pickle(os.path.join(thresh_dir,'LBM_dataframe.pkl'))
        
if __name__ == '__main__':

    array_task_id = int(sys.argv[1])    
    main(array_task_id)