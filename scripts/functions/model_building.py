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

def collate_batch(batch):
    (label_list, idx_list, bin_offsets, gupi_offsets, gupis) = ([], [], [0], [0], [])
    for (seq_lists, curr_GUPI, curr_label) in batch:
        gupi_offsets.append(len(seq_lists))
        for curr_bin in seq_lists:
            label_list.append(curr_label)
            gupis.append(curr_GUPI)
            processed_bin_seq = torch.tensor(curr_bin,
                    dtype=torch.int64)
            idx_list.append(processed_bin_seq)
            bin_offsets.append(processed_bin_seq.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    gupi_offsets = torch.tensor(gupi_offsets[:-1]).cumsum(dim=0)
    bin_offsets = torch.tensor(bin_offsets[:-1]).cumsum(dim=0)
    idx_list = torch.cat(idx_list)
    return (label_list, idx_list, bin_offsets, gupi_offsets, gupis)

def load_predictions(info_df, progress_bar=True, progress_bar_desc=''):
    
    compiled_predictions = []
        
    if progress_bar:
        iterator = tqdm(range(info_df.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(info_df.shape[0])
    
    # Load each prediction file, add 'WindowIdx' and repeat/fold information
    for curr_row in iterator:
        
        curr_preds = pd.read_csv(info_df.file[curr_row])
        curr_preds['repeat'] = info_df.repeat[curr_row]
        curr_preds['fold'] = info_df.fold[curr_row]
        
        if info_df.adm_or_disch[curr_row] == 'adm':
            curr_preds['WindowIdx'] = curr_preds.groupby('GUPI').cumcount(ascending=True)+1

        elif info_df.adm_or_disch[curr_row] == 'disch':
            curr_preds['WindowIdx'] = curr_preds.groupby('GUPI').cumcount(ascending=False)+1
        
        compiled_predictions.append(curr_preds)
        
    return pd.concat(compiled_predictions,ignore_index=True)