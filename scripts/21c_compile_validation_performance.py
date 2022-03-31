#### Master Script 21c: Compile validation performance metrics ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile all performance metrics
# III. Calculate confidence intervals on performance metrics

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
from pandas.api.types import CategoricalDtype
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

# Custom analysis functions
from functions.analysis import collect_metrics
from functions.model_building import collate_batch, format_tokens, format_time_tokens, T_scaling, vector_scaling

# Define directories in which performance metrics are saved
VERSION = 'v6-0'
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION
val_performance_dir = os.path.join(model_dir,'validation_performance')

# Load the optimised tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))
filt_tuning_grid = tuning_grid[['TUNE_IDX','STRATEGY','WINDOW_LIMIT','TIME_TOKENS','LATENT_DIM','HIDDEN_DIM']].drop_duplicates().reset_index(drop=True)

# Define number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()

### II. Compile all performance metrics
# Search for all performance metric files in the directory
val_metric_files = []
for path in Path(os.path.join(val_performance_dir)).rglob('*.pkl'):
    val_metric_files.append(str(path.resolve()))

# Compile all validation performance dataframes
val_performance_df = pd.concat([pd.read_pickle(f) for f in tqdm(val_metric_files)],ignore_index=True).sort_values(['TUNE_IDX','WindowIdx']).reset_index(drop=True)

# Calculate average of metrics across folds
ave_val_performance_df = val_performance_df.melt(id_vars=['TUNE_IDX','REPEAT','FOLD','WindowIdx'],value_vars=['Confidence','Accuracy','ECE','MCE','ORC'],ignore_index=True,var_name='METRIC',value_name='VALUE').groupby(['TUNE_IDX','WindowIdx','METRIC'],as_index=False)['VALUE'].mean()

# Observe overall discrimination performance
overall_ORC = ave_val_performance_df[(ave_val_performance_df.WindowIdx == 'overall')&(ave_val_performance_df.METRIC == 'ORC')].sort_values(by='VALUE',ascending=False).merge(filt_tuning_grid,how='left',on='TUNE_IDX')
abs_ORC = overall_ORC[overall_ORC.STRATEGY == 'abs'].reset_index(drop=True)
diff_ORC = overall_ORC[overall_ORC.STRATEGY == 'diff'].reset_index(drop=True)

# Observe lowest expected calibration error
overall_ECE = ave_val_performance_df[(ave_val_performance_df.WindowIdx == 'overall')&(ave_val_performance_df.METRIC == 'ECE')].sort_values(by='VALUE').merge(filt_tuning_grid,how='left',on='TUNE_IDX')
abs_ECE = overall_ECE[overall_ECE.STRATEGY == 'abs'].reset_index(drop=True)
diff_ECE = overall_ECE[overall_ECE.STRATEGY == 'diff'].reset_index(drop=True)

# Based on inspection of overall discrimination, identify top ABS and DIFF tuning indices
top_abs = 135
top_diff = 55

# Load raw predictions for top ABS and DIFF strategy models
val_grid = pd.read_pickle(os.path.join(model_dir,'val_performance_grid.pkl'))
abs_val_files = val_grid[val_grid.TUNE_IDX.isin([top_abs,top_diff])].reset_index(drop=True)

# Compile validation predictions of current grid selection
compiled_val_predictions = []

# Load each prediction file, add 'WindowIdx' and repeat/fold information
for curr_file_idx in range(abs_val_files.shape[0]):
    curr_preds = pd.read_csv(abs_val_files.file[curr_file_idx])
    curr_preds['WindowIdx'] = curr_preds.groupby('GUPI').cumcount(ascending=True)+1
    curr_preds['TUNE_IDX'] = abs_val_files.TUNE_IDX[curr_file_idx]
    curr_preds['REPEAT'] = abs_val_files.repeat[curr_file_idx]
    curr_preds['FOLD'] = abs_val_files.fold[curr_file_idx]
    compiled_val_predictions.append(curr_preds)
compiled_val_predictions = pd.concat(compiled_val_predictions,ignore_index=True)
prob_cols = [col for col in compiled_val_predictions if col.startswith('Pr(GOSE=')]
logit_cols = [col for col in compiled_val_predictions if col.startswith('z_GOSE=')]

# Filter out validation predictions corresponding to the top-performing configurations
abs_val_files = abs_val_files.rename(columns = {'repeat':'REPEAT','fold':'FOLD'})
top_val_performance_df = ave_val_performance_df[ave_val_performance_df.TUNE_IDX.isin([top_abs,top_diff])].reset_index(drop=True)
overall_top_performance = top_val_performance_df[top_val_performance_df.WindowIdx == 'overall'].reset_index(drop=True)

# Save uncalibrated validation performance of top-performing condfigurations
top_val_performance_df.to_csv(os.path.join(val_performance_dir,'top_config_val_performance.csv'),index=False)






# Curves of calibration error over window index
ece_curves = top_val_performance_df[(top_val_performance_df.METRIC == 'ORC')&(top_val_performance_df.WindowIdx != 'overall')].reset_index(drop=True)
plt.plot(ece_curves.WindowIdx[ece_curves.TUNE_IDX == 55],ece_curves.VALUE[ece_curves.TUNE_IDX == 55], label = "55")
plt.plot(ece_curves.WindowIdx[ece_curves.TUNE_IDX == 135],ece_curves.VALUE[ece_curves.TUNE_IDX == 135], label = "135")
plt.legend()
plt.show()

## TEMPERATURE SCALING
# Calibrate validation predictions with temperature-scaling
temperature = nn.Parameter((torch.ones(1)))
args = {'temperature': temperature}
for curr_file_idx in range(abs_val_files.shape[0]):
    
    bal_weights = torch.from_numpy(compute_class_weight(class_weight='balanced',
                                                    classes=np.sort(np.unique(compiled_val_predictions.TrueLabel)),
                                                    y=compiled_val_predictions.TrueLabel))
    
    pass

criterion = nn.CrossEntropyLoss(weight=bal_weights)
temps = []
losses = []

# Removing strong_wolfe line search results in jump after 50 epochs
temp_optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

# Define custom temperature optimization function
def _temp_eval():
    loss = criterion(T_scaling(torch.tensor(compiled_val_predictions[logit_cols].to_numpy()), args), torch.tensor(compiled_val_predictions.TrueLabel.to_numpy()))
    loss.backward()
    temps.append(temperature.item())
    losses.append(loss.data.item())
    return loss

# Optimize temperature value
temp_optimizer.step(_temp_eval)
t_opt = temperature.item()

# Scale and save temperature-scaled validation set
t_scaled_val_probs = pd.DataFrame(F.softmax(torch.div(torch.tensor(compiled_val_predictions[logit_cols].to_numpy()), t_opt)).numpy(),columns=prob_cols)
t_scaled_val_preds = pd.DataFrame(torch.div(torch.tensor(compiled_val_predictions[logit_cols].to_numpy()), t_opt).numpy(),columns=logit_cols)
t_scaled_val_preds = pd.concat([t_scaled_val_preds,t_scaled_val_probs], axis=1)
t_scaled_val_preds['TrueLabel'] = compiled_val_predictions['TrueLabel']
t_scaled_val_preds.insert(loc=0, column='GUPI', value=compiled_val_predictions['GUPI'])        
t_scaled_val_preds.insert(loc=1, column='T_OPT', value=t_opt)        
t_scaled_val_preds['TUNE_IDX'] = 143

t_scaled_val_preds['PredLabel'] = np.argmax(t_scaled_val_preds[prob_cols].to_numpy(),axis=1)
t_scaled_val_preds['Confidence'] = t_scaled_val_preds[prob_cols].max(axis=1)
t_scaled_val_preds['Hit'] = (t_scaled_val_preds.PredLabel == t_scaled_val_preds.TrueLabel).astype(int)

confidence_linspace = np.linspace(0,1,200)
smooth_accuracy = lowess(endog = t_scaled_val_preds['Hit'], exog = t_scaled_val_preds['Confidence'], it = 0, xvals = confidence_linspace)
t_scaled_val_preds['Smooth_Accuracy'] = t_scaled_val_preds['Confidence'].apply(lambda x: smooth_accuracy[(np.abs(x - confidence_linspace)).argmin()])
overall_ECE = (t_scaled_val_preds['Smooth_Accuracy'] - t_scaled_val_preds['Confidence']).abs().mean()
overall_MCE = (t_scaled_val_preds['Smooth_Accuracy'] - t_scaled_val_preds['Confidence']).abs().max()
overall_calibration = pd.DataFrame({'WindowIdx':'overall','ECE':overall_ECE,'MCE':overall_MCE},index=[0])


t_scaled_val_preds.to_csv(os.path.join(tune_dir,'t_scaled_val_predictions.csv'),index=False)

troli = val_performance_df[val_performance_df.WindowIdx != 'overall']
troli['WindowIdx'] = troli['WindowIdx'].astype(int)
troli = troli[troli.WindowIdx <= 36]
foli = troli.groupby(['TUNE_IDX','STRATEGY'],as_index=False)['ORC'].mean()
foli = foli.sort_values('ORC',ascending=False).reset_index(drop=True)

schmopi = val_performance_df[(val_performance_df.TUNE_IDX == 143)&(val_performance_df.WindowIdx != 'overall')]
schmopi['WindowIdx'] = schmopi['WindowIdx'].astype(int)
schmopi = schmopi[schmopi.WindowIdx <= 48]
plt.plot(schmopi.ORC)

abs_loli = val_performance_df[(val_performance_df.WindowIdx == 'overall')&(val_performance_df.STRATEGY == 'abs')].sort_values('ORC',ascending=False)
diff_loli = val_performance_df[(val_performance_df.WindowIdx == 'overall')&(val_performance_df.STRATEGY == 'diff')].sort_values('ORC',ascending=False)
check = val_performance_df[val_performance_df.WindowIdx != 'overall'].groupby('TUNE_IDX',as_index=False)['ORC'].mean().sort_values('ORC',ascending=False)