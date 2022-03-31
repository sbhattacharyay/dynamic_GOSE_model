#### Master Script 21b: Calculate predictions of dynamic all-predictor-based models on each set ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile and save validation and testing set predictions across partitions

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
from shutil import rmtree
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

# StatsModel methods
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

# TQDM for progress tracking
from tqdm import tqdm

# Custom methods
from classes.datasets import DYN_ALL_PREDICTOR_SET
from functions.model_building import collate_batch, format_tokens, load_tune_predictions
from functions.analysis import calc_ORC
from models.dynamic_APM import GOSE_model

# Set version code
VERSION = 'v6-0'

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
study_GUPIs = cv_splits[['GUPI','GOSE']].drop_duplicates()

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Load the optimised tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

# Create directory for storing validaiton performance metrics
val_performance_dir = os.path.join(model_dir,'validation_performance')
os.makedirs(val_performance_dir,exist_ok=True)

### II. Create grid of model checkpoint files and sets
# If model checkpoint dataframe already exists, load it. Otherwise create it
if not os.path.exists(os.path.join(model_dir,'val_performance_grid.pkl')):

    # Search for all saved model checkpoint files
    uncalibrated_pred_files = []
    for path in Path(model_dir).rglob('uncalibrated_*'):
        uncalibrated_pred_files.append(str(path.resolve()))

    uncalibrated_pred_info = pd.DataFrame({'file':uncalibrated_pred_files,
                                           'TUNE_IDX':[int(re.search('/tune(.*)/uncalibrated_', curr_file).group(1)) for curr_file in uncalibrated_pred_files],
                                           'SET':[re.search('uncalibrated_(.*)_predictions.csv', curr_file).group(1) for curr_file in uncalibrated_pred_files],
                                           'VERSION':[re.search('_outputs/(.*)/repeat', curr_file).group(1) for curr_file in uncalibrated_pred_files],
                                           'repeat':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in uncalibrated_pred_files],
                                           'fold':[int(re.search('/fold(.*)/tune', curr_file).group(1)) for curr_file in uncalibrated_pred_files]
                                          }).sort_values(by=['TUNE_IDX','repeat','fold','SET','VERSION']).reset_index(drop=True)
    
    # Extract validation prediction files
    val_grid = uncalibrated_pred_info[uncalibrated_pred_info.SET == 'val'].reset_index(drop=True)
    
    # Save validation prediction grid
    val_grid.to_pickle(os.path.join(model_dir,'val_performance_grid.pkl'))
    
else:
    
    # Load validation performance grid
    val_grid = pd.read_pickle(os.path.join(model_dir,'val_performance_grid.pkl'))

### III. Calculate validation performance metrics based on the `array_task_id` and validation performance grid
# Argument-induced training functions
def main(array_task_id):
    
    # Extract validation performance grid information
    curr_file = val_grid.file[array_task_id]
    curr_tune_idx = val_grid.TUNE_IDX[array_task_id]
    curr_repeat = val_grid.repeat[array_task_id]
    curr_fold = val_grid.fold[array_task_id]
   
    # Create a directory for the current repeat
    repeat_dir = os.path.join(val_performance_dir,'repeat'+str(curr_repeat).zfill(2))
    os.makedirs(repeat_dir,exist_ok=True)
    
    # Create a directory for the current fold
    fold_dir = os.path.join(repeat_dir,'fold'+str(curr_fold).zfill(int(np.log10(tuning_grid.fold.max()))+1))
    os.makedirs(fold_dir,exist_ok=True)

    # Load predictions of current tuning configuration-cross-validation partition combination
    val_predictions = pd.read_csv(curr_file)
    val_predictions['WindowIdx'] = val_predictions.groupby('GUPI').cumcount(ascending=True)+1
    val_predictions['REPEAT'] = curr_repeat
    val_predictions['FOLD'] = curr_fold
    
    # Identify validation set columns that correspond to probability
    prob_cols = [col for col in val_predictions if col.startswith('Pr(GOSE=')]
    val_predictions['PredLabel'] = np.argmax(val_predictions[prob_cols].to_numpy(),axis=1)
    val_predictions['Confidence'] = val_predictions[prob_cols].max(axis=1)
    val_predictions['Hit'] = (val_predictions.PredLabel == val_predictions.TrueLabel).astype(int)

    # Filter out predictions before 7 days after admission
    val_predictions = val_predictions[val_predictions.WindowIdx <= 84].reset_index(drop=True)
    
    # Calculate overall accuracy and confidence
    overall_accuracy = val_predictions.Hit.sum()/val_predictions.shape[0]
    overall_confidence = val_predictions.Confidence.mean()
    overall_metrics = pd.DataFrame({'WindowIdx':'overall','Confidence':overall_confidence,'Accuracy':overall_accuracy},index=[0])
    
    # Calculate confidence and accuracy by window index
    idx_confidence = val_predictions.groupby('WindowIdx',as_index=False)['Confidence'].mean()
    idx_accuracy = val_predictions.groupby('WindowIdx',as_index=False)['Hit'].mean().rename(columns={'Hit':'Accuracy'})
    idx_metrics = idx_confidence.merge(idx_accuracy,how='left',on='WindowIdx')
    metrics = pd.concat([overall_metrics,idx_metrics],ignore_index=True)
    
    # Calculate overall calibration error
    confidence_linspace = np.linspace(val_predictions.Confidence.min(),val_predictions.Confidence.max(),200)
    smooth_accuracy = lowess(endog = val_predictions['Hit'], exog = val_predictions['Confidence'], it = 0, xvals = confidence_linspace)
    val_predictions['Smooth_Accuracy'] = val_predictions['Confidence'].apply(lambda x: smooth_accuracy[(np.abs(x - confidence_linspace)).argmin()])
    overall_ECE = (val_predictions['Smooth_Accuracy'] - val_predictions['Confidence']).abs().mean()
    overall_MCE = (val_predictions['Smooth_Accuracy'] - val_predictions['Confidence']).abs().max()
    overall_calibration = pd.DataFrame({'WindowIdx':'overall','ECE':overall_ECE,'MCE':overall_MCE},index=[0])
    
    # Iterate through window indices and calculate calibration errors
    wi_calibrations = []
    for curr_window_idx in range(1,85):
        curr_wi_preds = val_predictions[val_predictions.WindowIdx == curr_window_idx].reset_index(drop=True)
        curr_smooth_accuracy = lowess(endog = curr_wi_preds['Hit'], exog = curr_wi_preds['Confidence'], it = 0, xvals = confidence_linspace)
        curr_wi_preds['Smooth_Accuracy'] = curr_wi_preds['Confidence'].apply(lambda x: curr_smooth_accuracy[(np.abs(x - confidence_linspace)).argmin()])
        curr_ECE = (curr_wi_preds['Smooth_Accuracy'] - curr_wi_preds['Confidence']).abs().mean()
        curr_MCE = (curr_wi_preds['Smooth_Accuracy'] - curr_wi_preds['Confidence']).abs().max()
        wi_calibrations.append(pd.DataFrame({'WindowIdx':curr_window_idx,'ECE':curr_ECE,'MCE':curr_MCE},index=[0]))
    calibration = pd.concat([overall_calibration,pd.concat(wi_calibrations,ignore_index=True)],ignore_index=True)
    
    # Calculate overall ORC
    overall_ORC = calc_ORC(val_predictions[prob_cols],val_predictions['TrueLabel'],prob_cols)
    overall_ORC = pd.DataFrame({'WindowIdx':'overall','ORC':overall_ORC},index=[0])
    
    # Iterate through window indices and calculate ORCs
    wi_ORCs = []
    for curr_window_idx in range(1,85):
        curr_wi_preds = val_predictions[val_predictions.WindowIdx == curr_window_idx].reset_index(drop=True)
        curr_ORC = calc_ORC(curr_wi_preds[prob_cols],curr_wi_preds['TrueLabel'],prob_cols)
        wi_ORCs.append(pd.DataFrame({'WindowIdx':curr_window_idx,'ORC':curr_ORC},index=[0]))
    ORC = pd.concat([overall_ORC,pd.concat(wi_ORCs,ignore_index=True)],ignore_index=True)
    
    # Combine validation performance metrics
    val_performance_df = pd.merge(metrics,calibration,how='left',on='WindowIdx').merge(ORC,how='left',on='WindowIdx')
    val_performance_df.insert(loc=0, column='TUNE_IDX', value=curr_tune_idx)        
    val_performance_df.insert(loc=1, column='REPEAT', value=curr_repeat)        
    val_performance_df.insert(loc=2, column='FOLD', value=curr_fold)        

    # Save validation performance metrics for current tuning configuration
    val_performance_df.to_pickle(os.path.join(fold_dir,'tune_'+str(curr_tune_idx).zfill(4)+'_val_performance.pkl'))
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)