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
from pandas.api.types import is_integer_dtype, is_float_dtype
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

from tqdm import tqdm

# Function to remove dataframe elements that contain no strings
def no_digit_strings_to_na(x):
    if not any(char.isdigit() for char in str(x)):
        return np.nan
    else:
        return x
    
# Function to categorize tokens under certain conditions
def categorizer(x):
    if is_integer_dtype(x) & (len(x.unique()) <= 20):
        return x.astype(str).str.zfill(int(np.log10(x.max()))+1)
    elif is_float_dtype(x) & (len(x.unique()) <= 20):
        new_x = x.astype(str).str.replace('.','dec',regex=False)
        new_x[new_x.str.endswith('dec0')] = new_x[new_x.str.endswith('dec0')].str.replace('dec0','',regex=False)
        return new_x
    else:
        return x
    
def tokenize_categoricals(x):
    if (is_integer_dtype(x)) | (is_float_dtype(x)) | (x.name == 'GUPI'):
        return x
    else:
        new_x = x
        new_x = new_x.astype(str).str.upper().str.replace('[^0-9a-zA-Z.]+','',regex=True).replace('','NAN')
        new_x = new_x.name + '_' + new_x.astype(str)
        new_x[new_x.str.endswith('.0')] = new_x[new_x.str.endswith('.0')].str.replace('.0','',regex=False)
        return new_x
    
def get_ts_event_tokens(cat_df,curr_GUPI,curr_last_timestamp):
    filt_cat_df = cat_df[cat_df.GUPI == curr_GUPI]
    filt_cat_df['Timestamp'] = pd.to_datetime(filt_cat_df['Timestamp'],format = '%Y-%m-%d %H:%M:%S' )
    filt_cat_df = filt_cat_df[filt_cat_df['Timestamp'] <= curr_last_timestamp]
    
    if filt_cat_df.shape[0] == 0:
        return np.asarray([])
    
    filt_cat_df[filt_cat_df.columns[~filt_cat_df.columns.isin(['GUPI','Timestamp'])]] = filt_cat_df[filt_cat_df.columns[~filt_cat_df.columns.isin(['GUPI','Timestamp'])]].apply(lambda x: x.str.replace('\s+','',regex=True))
    
    filt_cat_df['TokenStub'] = filt_cat_df[filt_cat_df.columns[~filt_cat_df.columns.isin(['GUPI','Timestamp'])]].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    filt_cat_df = filt_cat_df.groupby(['GUPI'],as_index=False)['TokenStub'].apply(' '.join).reset_index(drop=True)
    
    return np.unique(' '.join(filt_cat_df['TokenStub']).split())

def get_date_event_tokens(cat_df,curr_GUPI,curr_last_timestamp):
    filt_cat_df = cat_df[cat_df.GUPI == curr_GUPI]
    filt_cat_df['Date'] = pd.to_datetime(filt_cat_df['Date'],format = '%Y-%m-%d')
    filt_cat_df = filt_cat_df[filt_cat_df['Date'] <= curr_last_timestamp]
    
    if filt_cat_df.shape[0] == 0:
        return np.asarray([])
    
    filt_cat_df[filt_cat_df.columns[~filt_cat_df.columns.isin(['GUPI','Date'])]] = filt_cat_df[filt_cat_df.columns[~filt_cat_df.columns.isin(['GUPI','Date'])]].apply(lambda x: x.str.replace('\s+','',regex=True))
    
    filt_cat_df['TokenStub'] = filt_cat_df[filt_cat_df.columns[~filt_cat_df.columns.isin(['GUPI','Date'])]].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    filt_cat_df = filt_cat_df.groupby(['GUPI'],as_index=False)['TokenStub'].apply(' '.join).reset_index(drop=True)
    
    return np.unique(' '.join(filt_cat_df['TokenStub']).split())

def get_ts_interval_tokens(cat_df,curr_GUPI,curr_last_timestamp):
    filt_cat_df = cat_df[cat_df.GUPI == curr_GUPI]
    filt_cat_df['Timestamp'] = pd.to_datetime(filt_cat_df['Timestamp'],format = '%Y-%m-%d %H:%M:%S' )
    filt_cat_df = filt_cat_df[filt_cat_df['Timestamp'] <= curr_last_timestamp]
    
    if filt_cat_df.shape[0] == 0:
        return np.asarray([])
    
    filt_cat_df[filt_cat_df.columns[~filt_cat_df.columns.isin(['GUPI','Timestamp'])]] = filt_cat_df[filt_cat_df.columns[~filt_cat_df.columns.isin(['GUPI','Timestamp'])]].apply(lambda x: x.str.replace('\s+','',regex=True))
    
    filt_cat_df['TokenStub'] = filt_cat_df[filt_cat_df.columns[~filt_cat_df.columns.isin(['GUPI','Timestamp'])]].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    filt_cat_df = filt_cat_df.groupby(['GUPI'],as_index=False)['TokenStub'].apply(' '.join).reset_index(drop=True)
    
    return np.unique(' '.join(filt_cat_df['TokenStub']).split())

def load_tokens(GUPIs,token_dir,progress_bar=True,progress_bar_desc=''):
    
    if progress_bar:
        iterator = tqdm(GUPIs,desc=progress_bar_desc)
    else:
        iterator = GUPIs
                
    return pd.concat([pd.read_csv(os.path.join(token_dir,g+'.csv')) for g in iterator],ignore_index=True)