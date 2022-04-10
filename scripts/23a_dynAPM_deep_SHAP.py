#### Master Script 23a: Calculate SHAP values for dynAPM_DeepMN ####
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
from pytorch_lightning.callbacks import ModelCheckpoint
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
from functions.model_building import format_shap, format_tokens

# Set version code
VERSION = 'v6-0'

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Load cross-validation split information to extract testing resamples
cv_splits = pd.read_csv('../cross_validation_splits.csv')
test_splits = cv_splits[cv_splits.set == 'test'].reset_index(drop=True)
test_splits = test_splits[test_splits.repeat == 1].reset_index(drop=True)
uniq_GUPIs = test_splits.GUPI.unique()

# Define a directory for the storage of SHAP values
shap_dir = os.path.join(model_dir,'SHAP_values')
os.makedirs(shap_dir,exist_ok=True)

### II. Find all top-performing model checkpoint files for SHAP calculation
# Either create or load APM checkpoint information for SHAP value 
if not os.path.exists(os.path.join(shap_dir,'ckpt_info.pkl')):
    
    # Find all model checkpoint files in APM output directory
    ckpt_files = []
    for path in Path(model_dir).rglob('*.ckpt'):
        ckpt_files.append(str(path.resolve()))

    # Categorize model checkpoint files based on name
    ckpt_info = pd.DataFrame({'file':ckpt_files,
                              'TUNE_IDX':[re.search('tune(.*)/', curr_file).group(1) for curr_file in ckpt_files],
                              'VERSION':[re.search('model_outputs/(.*)/repeat', curr_file).group(1) for curr_file in ckpt_files],
                              'repeat':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in ckpt_files],
                              'fold':[re.search('/fold(.*)/tune', curr_file).group(1) for curr_file in ckpt_files]
                             }).sort_values(by=['repeat','fold','TUNE_IDX','VERSION']).reset_index(drop=True)
    ckpt_info['ADM_OR_DISCH'] = ckpt_info['ADM_OR_DISCH'].str.rsplit(pat='/', n=1).apply(lambda x: x[1])
    ckpt_info['fold'] = ckpt_info['fold'].str.rsplit(pat='/', n=1).apply(lambda x: x[0]).astype(int)
    
    # Create combinations for each possible prediction point
    pred_points = pd.DataFrame({'prediction_point':list(range(1,85)),'key':1})
    
    # Merge output type information to checkpoint info dataframe
    ckpt_info['key'] = 1
    ckpt_info = pd.merge(ckpt_info,pred_points,how='outer',on='key').drop(columns='key')
    
    # Save model checkpoint information dataframe
    ckpt_info.to_pickle(os.path.join(shap_dir,'ckpt_info.pkl'))
    
else:
    
    # Read model checkpoint information dataframe
    ckpt_info = pd.read_pickle(os.path.join(shap_dir,'ckpt_info.pkl'))

ckpt_info = ckpt_info[ckpt_info.ADM_OR_DISCH == 'adm'].reset_index(drop=True)
    
### III. Calculate SHAP values based on given parameters
def main(array_task_id):

    # Extract current file, repeat, and fold information
    curr_file = ckpt_info.file[array_task_id]
    curr_repeat = ckpt_info.repeat[array_task_id]
    curr_fold = ckpt_info.fold[array_task_id]
    curr_TUNE_IDX = ckpt_info.TUNE_IDX[array_task_id]
    curr_ADM_OR_DISCH = ckpt_info.ADM_OR_DISCH[array_task_id]
    curr_pred_point = ckpt_info.prediction_point[array_task_id]
    
    # Define current fold directory based on current information
    tune_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold),curr_ADM_OR_DISCH,'tune'+curr_TUNE_IDX)
    
    # Extract current testing set for current repeat and fold combination
    testing_set = pd.read_pickle('/home/sb2406/rds/hpc-work/tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/from_'+curr_ADM_OR_DISCH+'_testing_indices.pkl')
    testing_set = format_tokens(testing_set,84,curr_ADM_OR_DISCH,2)
    testing_set['seq_len'] = testing_set.VocabIndex.apply(len)
    testing_set['unknowns'] = testing_set.VocabIndex.apply(lambda x: x.count(0))
    
    # Filter out patients whose predictions do not exceed the current prediciton point
    testing_set = testing_set[testing_set.WindowTotal >= curr_pred_point].reset_index(drop=True)
    
    # Filter rows under the current prediction point
    testing_set = testing_set[testing_set.WindowIdx <= curr_pred_point].reset_index(drop=True)

    # Extract current partition GUPIs and GOSEs
    curr_test_GUPIs = cv_splits[(cv_splits.repeat == curr_repeat) & (cv_splits.fold == curr_fold) & (cv_splits.GUPI.isin(testing_set.GUPI.unique()))].reset_index(drop=True)
    background_GUPIs = np.sort(resample(curr_test_GUPIs.GUPI.values,replace=False,n_samples=20))
    GUPI_idxs = np.where(np.isin(testing_set.GUPI.unique(),background_GUPIs))[0]
    
    # Find number of bins per patient in the testing set
    rows_per_patient = testing_set.groupby('GUPI',as_index=False).size().sort_values(by='GUPI',ignore_index=True)
    seq_lens = torch.tensor(testing_set['seq_len'])
    gupi_offsets = [0] + rows_per_patient['size'].to_list()
    gupi_offsets = torch.tensor(gupi_offsets[:-1]).cumsum(dim=0)

    # Number of columns to add
    cols_to_add = testing_set['unknowns'].max() - 1

    # Load current token dictionary
    curr_vocab = cp.load(open('/home/sb2406/rds/hpc-work/tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/from_'+curr_ADM_OR_DISCH+'_token_dictionary.pkl',"rb"))

    # Initialize empty dataframe for multihot encoding of testing set
    multihot_matrix = np.zeros([testing_set.shape[0],len(curr_vocab)+cols_to_add])

    # Encode testing set into multihot encoded matrix
    for i in range(testing_set.shape[0]):
        curr_indices = np.array(testing_set.VocabIndex[i])
        if sum(curr_indices == 0) > 1:
            zero_indices = np.where(curr_indices == 0)[0]
            curr_indices[zero_indices[1:]] = [len(curr_vocab) + i for i in range(sum(curr_indices == 0)-1)]
        multihot_matrix[i,curr_indices] = 1
    multihot_matrix = torch.tensor(multihot_matrix).float()
    
    # Convert multihot encoded matrix into RNN-compatible sequence
    multihot_rnn_sequence = []
    for curr_gupi_idx in torch.arange(0, len(gupi_offsets), dtype=torch.long):
        if curr_gupi_idx == (torch.LongTensor([len(gupi_offsets) - 1])[0]):
            curr_gupi_seq = torch.arange(gupi_offsets[curr_gupi_idx], multihot_matrix.shape[0], dtype=torch.long)
        else:
            curr_gupi_seq = torch.arange(gupi_offsets[curr_gupi_idx], gupi_offsets[curr_gupi_idx+1], dtype=torch.long) 
        multihot_rnn_sequence.append(multihot_matrix[curr_gupi_seq,:])
    multihot_rnn_sequence = pad_sequence(multihot_rnn_sequence,batch_first=True)
    
    # Load current pretrained model
    model = GOSE_model.load_from_checkpoint(curr_file)
    model.eval()
    
    # Extract learned weights from model checkpoint file
    vocab_embed_matrix = model.embedX.weight.detach().numpy()
    vocab_embed_matrix = np.append(vocab_embed_matrix,np.tile(np.expand_dims(vocab_embed_matrix[0,:], axis=0),(cols_to_add,1)),axis=0)
    vocab_embed_weights = np.exp(model.embedW.weight.detach().numpy())
    vocab_embed_weights = np.append(vocab_embed_weights,np.tile(np.expand_dims(vocab_embed_weights[0], axis=0),(cols_to_add,1)),axis=0)
    vocab_embed_matrix = torch.tensor(vocab_embed_matrix*vocab_embed_weights).float()
    
    # Load learnt weights and biases of RNN cell
    input_hidden_weights = model.rnn_module.weight_ih_l0.detach()
    hidden_hidden_weights = model.rnn_module.weight_hh_l0.detach()
    input_hidden_biases = model.rnn_module.bias_ih_l0.detach()
    hidden_hidden_biases = model.rnn_module.bias_hh_l0.detach()
    
    # Extract specific weights 
    hidden_size = int(input_hidden_weights.shape[0]/3)
    W_ir = input_hidden_weights[0:hidden_size,:]
    W_iz = input_hidden_weights[hidden_size:(2*hidden_size),:]
    W_in = input_hidden_weights[(2*hidden_size):(3*hidden_size),:]
    W_hr = hidden_hidden_weights[0:hidden_size,:]
    W_hz = hidden_hidden_weights[hidden_size:(2*hidden_size),:]
    W_hn = hidden_hidden_weights[(2*hidden_size):(3*hidden_size),:]
    
    # Extract specific biases
    b_ir = input_hidden_biases[0:hidden_size]
    b_iz = input_hidden_biases[hidden_size:(2*hidden_size)]
    b_in = input_hidden_biases[(2*hidden_size):(3*hidden_size)]
    b_hr = hidden_hidden_biases[0:hidden_size]
    b_hz = hidden_hidden_biases[hidden_size:(2*hidden_size)]
    b_hn = hidden_hidden_biases[(2*hidden_size):(3*hidden_size)]    
    
    # Load modified APM_deep instance based on trained weights and current output type
    shap_model = shap_GOSE_model(vocab_embed_matrix,
                                 W_ir,
                                 W_iz,
                                 W_in,
                                 W_hr,
                                 W_hz,
                                 W_hn,
                                 b_ir,
                                 b_iz, 
                                 b_in,
                                 b_hr,
                                 b_hz,
                                 b_hn, 
                                 model.hidden2gose,
                                 curr_pred_point,
                                 prob=True,
                                 thresh=False)

    # Initialize deep explainer explanation object
    e = shap.DeepExplainer(shap_model, multihot_rnn_sequence[GUPI_idxs,:,:])
    
    # Calculate SHAP values and save both explainer object and shap matrices 
    shap_values = e.shap_values(multihot_rnn_sequence)
#     cp.dump(e, open(os.path.join(tune_dir,'deep_explainer_from_'+curr_ADM_OR_DISCH+'_pred_point_'+str(curr_pred_point).zfill(2)+'.pkl'), "wb"))
#     cp.dump(shap_values, open(os.path.join(tune_dir,'shap_arrays_from_'+curr_ADM_OR_DISCH+'_pred_point_'+str(curr_pred_point).zfill(2)+'.pkl'), "wb"))
    
    # Define token labels
    token_labels = curr_vocab.get_itos() + [curr_vocab.get_itos()[0]+'_'+str(i+1).zfill(3) for i in range(cols_to_add)]
    token_labels[0] = token_labels[0]+'_000'
    
    # Convert each SHAP matrix to formatted dataframe and concatenate across labels
    shap_df = pd.concat([format_shap(curr_matrix,idx,token_labels,testing_set) for idx,curr_matrix in enumerate(shap_values)],ignore_index=True)
    shap_df['repeat'] = curr_repeat
    shap_df['fold'] = curr_fold
    shap_df['prediction_point'] = curr_pred_point
    
    # Convert multihot encoded matrix into formatted dataframe for token indicators
    indicator_df = pd.DataFrame(multihot_matrix.numpy(),columns=token_labels)
    indicator_df['GUPI'] = testing_set.GUPI
    indicator_df['WindowIdx'] = testing_set.WindowIdx
    indicator_df = indicator_df.melt(id_vars = ['GUPI','WindowIdx'], var_name = 'Token', value_name = 'Indicator')
    indicator_df['Indicator'] = indicator_df['Indicator'].astype(int)

    # Merge indicator dataframe with SHAP values
    shap_df = pd.merge(shap_df,indicator_df,how='left',on=['GUPI','WindowIdx','Token'])

    # Remove rows which correspond to non-existent or unknown tokens and save formatted dataframe
    shap_df = shap_df[shap_df.Indicator == 1]
    shap_df = shap_df[~shap_df.Token.str.startswith('<unk>_')].reset_index(drop=True)
    shap_df.to_pickle(os.path.join(tune_dir,'shap_dataframe_from_'+curr_ADM_OR_DISCH+'_pred_point_'+str(curr_pred_point).zfill(2)+'.pkl'))
    
#     # Calculate correlation among tokens if it does not already exist
#     if curr_output_type == 'logit':
        
#         corr_matrix = multihot_matrix.copy()
#         corr_matrix[corr_matrix == 0] = -1
#         corr_matrix = np.matmul(corr_matrix.transpose(),corr_matrix)
#         corr_matrix = corr_matrix/multihot_matrix.shape[0]
#         corr_matrix = np.triu(corr_matrix,1)
#         corr_matrix[np.tril_indices(corr_matrix.shape[0], 1)] = np.nan
        
#         corr_df = pd.DataFrame(corr_matrix,columns=token_labels)
#         corr_df['Token1'] = token_labels
#         corr_df = corr_df.melt(id_vars = 'Token1', var_name = 'Token2', value_name = 'correlation')
#         corr_df = corr_df.dropna().reset_index(drop=True)
#         corr_df.to_pickle(os.path.join(tune_dir,'correlation_dataframe.xz'),compression='xz')

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)