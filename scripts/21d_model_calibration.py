#### Master Script 21d: Assess calibration methods for top-performing configurations ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Create grid of calibration combinations
# III. Calibrate APM_deep model based on provided hyperparameter row index

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
from scipy.special import logit
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

# StatsModel methods
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

# TQDM for progress tracking
from tqdm import tqdm

# Custom methods
from classes.calibration import TemperatureScaling, VectorScaling
from classes.datasets import DYN_ALL_PREDICTOR_SET
from functions.model_building import collate_batch, format_tokens, format_time_tokens, T_scaling, vector_scaling
from functions.analysis import calc_ECE, calc_MCE, calc_ORC, calc_thresh_calibration
from models.dynamic_APM import GOSE_model

# Set version code
VERSION = 'v6-0'

# Initialise model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION
val_performance_dir = os.path.join(model_dir,'validation_performance')
calibration_dir = os.path.join(model_dir,'calibration_performance')
os.makedirs(calibration_dir,exist_ok=True)

# Load cross-validation split information
cv_splits = pd.read_csv('../cross_validation_splits.csv')

### II. Create grid of calibration combinations
# If bootstrapping resamples don't exist, create them
if not os.path.exists(os.path.join(calibration_dir,'calibration_grid.csv')):

    # Create parameters for training differential token models
    calibration_parameters = {'TUNE_IDX':[135,69],
                              'SCALING':['T','vector'],
                              'OPTIMIZATION':['nominal','ordinal'],
                              'WINDOW_IDX':list(range(1,85)),
                              'REPEAT':list(range(1,21)),
                              'FOLD':list(range(1,6))}
    
    # Convert parameter dictionary to dataframe
    calibration_grid = pd.DataFrame([row for row in itertools.product(*calibration_parameters.values())],columns=calibration_parameters.keys()).sort_values(by=['TUNE_IDX','SCALING','OPTIMIZATION','REPEAT','FOLD','WINDOW_IDX'],ignore_index=True)
    
    # Remove implausible rows
    calibration_grid = calibration_grid[(calibration_grid.TUNE_IDX != 69)|(calibration_grid.REPEAT==1)].reset_index(drop=True)
    
    # Save calibration grid to model directory
    calibration_grid.to_csv(os.path.join(calibration_dir,'calibration_grid.csv'),index=False)

else:
    # Load calibration grid
    calibration_grid = pd.read_csv(os.path.join(calibration_dir,'calibration_grid.csv'))
    
### III. Calibrate APM_deep model based on provided hyperparameter row index
# Argument-induced training functions
def main(array_task_id):
    
    # Extract current row informmation
    curr_tune_idx = calibration_grid.TUNE_IDX[array_task_id]
    curr_scaling = calibration_grid.SCALING[array_task_id]
    curr_optimization = calibration_grid.OPTIMIZATION[array_task_id]
    curr_window_idx = calibration_grid.WINDOW_IDX[array_task_id]
    curr_repeat = calibration_grid.REPEAT[array_task_id]
    curr_fold = calibration_grid.FOLD[array_task_id]
    
    # Define current tune directory
    tune_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold).zfill(int(np.log10(calibration_grid.FOLD.max()))+1),'tune'+str(curr_tune_idx).zfill(4))
    
    # Load uncalibrated validation set of current combination
    uncalib_val_preds = pd.read_csv(os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold),'tune'+str(curr_tune_idx).zfill(4),'uncalibrated_val_predictions.csv'))
    uncalib_val_preds['WindowIdx'] = uncalib_val_preds.groupby('GUPI').cumcount(ascending=True)+1

    # Filter out the predictions of the current window index
    uncalib_val_preds = uncalib_val_preds[uncalib_val_preds.WindowIdx == curr_window_idx].reset_index(drop=True)
        
    # Load uncalibrated testing set of current combination
    uncalib_test_preds = pd.read_csv(os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold),'tune'+str(curr_tune_idx).zfill(4),'uncalibrated_test_predictions.csv'))
    uncalib_test_preds['WindowIdx'] = uncalib_test_preds.groupby('GUPI').cumcount(ascending=True)+1

    # Filter out the predictions of the current window index
    uncalib_test_preds = uncalib_test_preds[uncalib_test_preds.WindowIdx == curr_window_idx].reset_index(drop=True)
    
    # Extract names of important columns
    logit_cols = [col for col in uncalib_val_preds if col.startswith('z_GOSE=')]
    prob_cols = [col for col in uncalib_val_preds if col.startswith('Pr(GOSE=')]

    # Calculate pre-calibration validation and testing calibration metrics
    pre_cal_val_ECE = calc_ECE(uncalib_val_preds)
    pre_cal_val_MCE = calc_MCE(uncalib_val_preds)
    pre_cal_test_ECE = calc_ECE(uncalib_test_preds)
    pre_cal_test_MCE = calc_MCE(uncalib_test_preds)
    
    # Calculate pre-calibration validation and testing calibration metrics at each threshold
    pre_cal_val_thresh_calib = calc_thresh_calibration(uncalib_val_preds)
    pre_cal_test_thresh_calib = calc_thresh_calibration(uncalib_test_preds)
    
    # Calculate pre-calibration discrimination performance
    pre_cal_val_ORC = calc_ORC(uncalib_val_preds[prob_cols],uncalib_val_preds['TrueLabel'],prob_cols)
    pre_cal_test_ORC = calc_ORC(uncalib_test_preds[prob_cols],uncalib_test_preds['TrueLabel'],prob_cols)

    # 
    if curr_scaling == 'T':
        scale_object = TemperatureScaling(uncalib_val_preds)
        scale_object.set_temperature(curr_optimization)
        with torch.no_grad():
            opt_temperature = scale_object.temperature.detach().item()
        if opt_temperature != opt_temperature:
            opt_temperature = 1    
        calib_val_logits = torch.tensor((uncalib_val_preds[logit_cols] / opt_temperature).values,dtype=torch.float32)
        calib_val_probs = F.softmax(calib_val_logits)
        calib_test_logits = torch.tensor((uncalib_test_preds[logit_cols] / opt_temperature).values,dtype=torch.float32)
        calib_test_probs = F.softmax(calib_test_logits)
        
    elif curr_scaling == 'vector':
        scale_object = VectorScaling(uncalib_val_preds)
        scale_object.set_vector(curr_optimization)
        with torch.no_grad():
            opt_vector = scale_object.vector.detach().data
            opt_biases = scale_object.biases.detach().data
        calib_val_logits = torch.matmul(torch.tensor(uncalib_val_preds[logit_cols].values,dtype=torch.float32),torch.diag_embed(opt_vector.squeeze(1))) + opt_biases.squeeze(1)
        calib_val_probs = F.softmax(calib_val_logits)
        calib_test_logits = torch.matmul(torch.tensor(uncalib_test_preds[logit_cols].values,dtype=torch.float32),torch.diag_embed(opt_vector.squeeze(1))) + opt_biases.squeeze(1)
        calib_test_probs = F.softmax(calib_test_logits)
        
    else:
        raise ValueError("Invalid scaling type. Must be 'T' or 'vector'")
        
    # 
    calib_val_preds = pd.DataFrame(torch.cat([calib_val_logits,calib_val_probs],1).numpy(),columns=logit_cols+prob_cols)
    calib_val_preds.insert(loc=0, column='GUPI', value=uncalib_val_preds['GUPI'])
    calib_val_preds['TrueLabel'] = uncalib_val_preds['TrueLabel']
    calib_val_preds['TUNE_IDX'] = curr_tune_idx
    calib_val_preds['WindowIdx'] = curr_window_idx
    calib_val_preds['REPEAT'] = curr_repeat
    calib_val_preds['FOLD'] = curr_fold
    calib_val_preds.to_pickle(os.path.join(tune_dir,'set_val_opt_'+curr_optimization+'_window_idx_'+str(curr_window_idx).zfill(2)+'_scaling_'+curr_scaling+'.pkl'))
    
    #
    post_cal_val_ECE = calc_ECE(calib_val_preds)
    post_cal_val_MCE = calc_MCE(calib_val_preds)
    
    # 
    calib_test_preds = pd.DataFrame(torch.cat([calib_test_logits,calib_test_probs],1).numpy(),columns=logit_cols+prob_cols)
    calib_test_preds.insert(loc=0, column='GUPI', value=uncalib_test_preds['GUPI'])
    calib_test_preds['TrueLabel'] = uncalib_test_preds['TrueLabel']
    calib_test_preds['TUNE_IDX'] = curr_tune_idx
    calib_test_preds['WindowIdx'] = curr_window_idx
    calib_test_preds['REPEAT'] = curr_repeat
    calib_test_preds['FOLD'] = curr_fold
    calib_test_preds.to_pickle(os.path.join(tune_dir,'set_test_opt_'+curr_optimization+'_window_idx_'+str(curr_window_idx).zfill(2)+'_scaling_'+curr_scaling+'.pkl'))

    #
    post_cal_test_ECE = calc_ECE(calib_test_preds)
    post_cal_test_MCE = calc_MCE(calib_test_preds)

    # Calculate post-calibration validation and testing calibration metrics at each threshold
    post_cal_val_thresh_calib = calc_thresh_calibration(calib_val_preds)
    post_cal_test_thresh_calib = calc_thresh_calibration(calib_test_preds)
    
    # Calculate post-calibration discrimination performance
    post_cal_val_ORC = calc_ORC(calib_val_preds[prob_cols],calib_val_preds['TrueLabel'],prob_cols)
    post_cal_test_ORC = calc_ORC(calib_test_preds[prob_cols],calib_test_preds['TrueLabel'],prob_cols)
    
    # Organize and merge threshold calibration metrics
    pre_cal_val_thresh_calib['SET'] = 'val'
    pre_cal_test_thresh_calib['SET'] = 'test'
    pre_cal_thresh_calib = pd.concat([pre_cal_val_thresh_calib,pre_cal_test_thresh_calib],ignore_index=True)
    pre_cal_thresh_calib['CALIBRATION'] = 'None'
    
    post_cal_val_thresh_calib['SET'] = 'val'
    post_cal_test_thresh_calib['SET'] = 'test'
    post_cal_thresh_calib = pd.concat([post_cal_val_thresh_calib,post_cal_test_thresh_calib],ignore_index=True)
    post_cal_thresh_calib['CALIBRATION'] = curr_scaling
    
    thresh_calib = pd.concat([pre_cal_thresh_calib,post_cal_thresh_calib],ignore_index=True).melt(id_vars=['THRESHOLD','SET','CALIBRATION'],var_name='METRIC',value_name='VALUE',ignore_index=True)
    ave_thresh_calib = thresh_calib.groupby(['SET','CALIBRATION','METRIC'],as_index=False)['VALUE'].mean()
    ave_thresh_calib.insert(loc=0, column='THRESHOLD', value='Average')
    thresh_calib = pd.concat([thresh_calib,ave_thresh_calib],ignore_index=True)
    
    # Add scalar metrics
    scalar_metrics = pd.DataFrame({'THRESHOLD':'Overall',
    'SET':['val','val','test','test','val','test','val','val','test','test','val','test'],
    'CALIBRATION':['None','None','None','None','None','None',curr_scaling,curr_scaling,curr_scaling,curr_scaling,curr_scaling,curr_scaling],
    'METRIC':['ECE','MCE','ECE','MCE','ORC','ORC','ECE','MCE','ECE','MCE','ORC','ORC'],
    'VALUE':[pre_cal_val_ECE,pre_cal_val_MCE,pre_cal_test_ECE,pre_cal_test_MCE,pre_cal_val_ORC,pre_cal_test_ORC,post_cal_val_ECE,post_cal_val_MCE,post_cal_test_ECE,post_cal_test_MCE,post_cal_val_ORC,post_cal_test_ORC]})
    
    # Concatenate metrics and sort
    metrics = pd.concat([thresh_calib,scalar_metrics],ignore_index=True).sort_values(by=['METRIC','THRESHOLD','SET','CALIBRATION'],ignore_index=True)
    metrics.insert(loc=0, column='TUNE_IDX', value=curr_tune_idx)
    metrics.insert(loc=1, column='OPTIMIZATION', value=curr_optimization)
    metrics.insert(loc=2, column='WINDOW_IDX', value=curr_window_idx)
    metrics.insert(loc=3, column='REPEAT', value=curr_repeat)
    metrics.insert(loc=4, column='FOLD', value=curr_fold)

    # 
    file_dir=os.path.join(calibration_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold).zfill(1))
    os.makedirs(file_dir,exist_ok=True)
    metrics.to_pickle(os.path.join(file_dir,'tune_idx_'+str(curr_tune_idx).zfill(4)+'_opt_'+curr_optimization+'_window_idx_'+str(curr_window_idx).zfill(2)+'_scaling_'+curr_scaling+'.pkl'))
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1]) + 30000    
    main(array_task_id)