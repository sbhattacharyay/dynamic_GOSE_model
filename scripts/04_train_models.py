#### Master Script 4: Train dynamic all-predictor-based models ####
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
from classes.datasets import ALL_PREDICTOR_SET
from models.APM import APM_deep

# Set version code
VERSION = 'v1-0'

# Initialise model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION
os.makedirs(model_dir,exist_ok=True)

# Load cross-validation split information
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Load the tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))
tuning_grid = tuning_grid[tuning_grid.OUTPUT_ACTIVATION == 'softmax']

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
    training_set = pd.read_pickle('/home/sb2406/rds/hpc-work/tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/training_indices.pkl')
    testing_set = pd.read_pickle('/home/sb2406/rds/hpc-work/tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/testing_indices.pkl')

    # Load current token dictionary
    curr_vocab = cp.load(open('/home/sb2406/rds/hpc-work/APM_tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/token_dictionary.pkl',"rb"))
    
    # Unique GUPI-GOSE combinations
    study_GUPIs = cv_splits[['GUPI','GOSE']].drop_duplicates()
    
    # Add GOSE scores to training and testing sets
    training_set = pd.merge(training_set,study_GUPIs,how='left',on='GUPI')
    testing_set = pd.merge(testing_set,study_GUPIs,how='left',on='GUPI')
    
    # Set aside 15% of the training set for validation, independent of the final testing set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15)
    for train_index, val_index in sss.split(training_set.drop(columns='GOSE'),training_set.GOSE):
        training_set, val_set = training_set.loc[train_index].reset_index(drop=True), training_set.loc[val_index].reset_index(drop=True)
    
    cp.dump(sss, open(os.path.join(tune_dir,'val_set_splitter.pkl'), "wb"))

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)