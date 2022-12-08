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
from argparse import ArgumentParser
from collections import Counter, OrderedDict
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
def categorizer(x,threshold=20):
    if is_integer_dtype(x) & (len(x.unique()) <= threshold):
        new_x = x.astype(str).str.zfill(3)
        new_x[new_x == 'nan'] = np.nan
        return new_x
    elif is_float_dtype(x) & (len(x.unique()) <= threshold):
        new_x = x.astype(str).str.replace('.','dec',regex=False)
        new_x[new_x.str.endswith('dec0')] = new_x[new_x.str.endswith('dec0')].str.replace('dec0','',regex=False)
        new_x = new_x.str.zfill(3)
        new_x[new_x == 'nan'] = np.nan
        return new_x
    else:
        return x

# Function to clean categorical token rows
def clean_token_rows(tokens_df_slice,progress_bar=True,progress_bar_desc=''):    
    
    if progress_bar:
        iterator = tqdm(range(tokens_df_slice.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(tokens_df_slice.shape[0])
    
    for curr_row in iterator:
        curr_token_set = tokens_df_slice.TOKENS.iloc[curr_row]
        cleaned_token_set = ' '.join(np.sort(np.unique(curr_token_set.split())))
        tokens_df_slice.TOKENS.iloc[curr_row] = cleaned_token_set
        
    return tokens_df_slice

# Function to characterise tokens of each study window
def get_token_info(index_df,vocab_df,missing = True, progress_bar=True, progress_bar_desc=''):
    
    compiled_token_characteristics = []
    
    if progress_bar:
        iterator = tqdm(range(index_df.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(index_df.shape[0])
    
    for curr_row in iterator:
        
        curr_IndexList = index_df.VocabIndex[curr_row]
        
        if np.isnan(curr_IndexList).all():
            compiled_token_characteristics.append(pd.DataFrame({'GUPI':index_df.GUPI[curr_row],
                                                                'TimeStampStart':index_df.TimeStampStart[curr_row],
                                                                'TimeStampEnd':index_df.TimeStampEnd[curr_row],
                                                                'WindowIdx':index_df.WindowIdx[curr_row],
                                                                'WindowTotal':index_df.WindowTotal[curr_row],
                                                                'Set':index_df.Set[curr_row],
                                                                'TotalTokens':0,
                                                                'Numeric':0,
                                                                'Baseline':0,
                                                                'Discharge':0,
                                                                'Ordered':0,
                                                                'Binary':0,
                                                                'ICUIntervention':0,
                                                                'ClinicianInput':0},index=[0]))
        elif (len(curr_IndexList) == 1):
            filt_vocab = vocab_df[vocab_df.VocabIndex.isin(curr_IndexList)].reset_index(drop=True)
            
            if not missing:
                filt_vocab = filt_vocab[~filt_vocab.Missing].reset_index(drop=True)
            
            if (filt_vocab.shape[0] == 0):
                compiled_token_characteristics.append(pd.DataFrame({'GUPI':index_df.GUPI[curr_row],
                                                                    'TimeStampStart':index_df.TimeStampStart[curr_row],
                                                                    'TimeStampEnd':index_df.TimeStampEnd[curr_row],
                                                                    'WindowIdx':index_df.WindowIdx[curr_row],
                                                                    'WindowTotal':index_df.WindowTotal[curr_row],
                                                                    'Set':index_df.Set[curr_row],
                                                                    'TotalTokens':0,
                                                                    'Numeric':0,
                                                                    'Baseline':0,
                                                                    'Discharge':0,
                                                                    'Ordered':0,
                                                                    'Binary':0,
                                                                    'ICUIntervention':0,
                                                                    'ClinicianInput':0},index=[0]))
            else:
                compiled_token_characteristics.append(pd.DataFrame({'GUPI':index_df.GUPI[curr_row],
                                                                    'TimeStampStart':index_df.TimeStampStart[curr_row],
                                                                    'TimeStampEnd':index_df.TimeStampEnd[curr_row],
                                                                    'WindowIdx':index_df.WindowIdx[curr_row],
                                                                    'WindowTotal':index_df.WindowTotal[curr_row],
                                                                    'Set':index_df.Set[curr_row],
                                                                    'TotalTokens':filt_vocab.shape[0],
                                                                    'Numeric':int(filt_vocab.Numeric[0]),
                                                                    'Baseline':int(filt_vocab.Baseline[0]),
                                                                    'Discharge':int(filt_vocab.Discharge[0]),
                                                                    'Ordered':int(filt_vocab.Ordered[0]),
                                                                    'Binary':int(filt_vocab.Binary[0]),
                                                                    'ICUIntervention':int(filt_vocab.ICUIntervention[0]),
                                                                    'ClinicianInput':int(filt_vocab.ClinicianInput[0])},index=[0]))
            
        else:
            filt_vocab = vocab_df[vocab_df.VocabIndex.isin(curr_IndexList)].reset_index(drop=True)

            if not missing:
                filt_vocab = filt_vocab[~filt_vocab.Missing].reset_index(drop=True)

            compiled_token_characteristics.append(pd.DataFrame({'GUPI':index_df.GUPI[curr_row],
                                                                'TimeStampStart':index_df.TimeStampStart[curr_row],
                                                                'TimeStampEnd':index_df.TimeStampEnd[curr_row],
                                                                'WindowIdx':index_df.WindowIdx[curr_row],
                                                                'WindowTotal':index_df.WindowTotal[curr_row],
                                                                'Set':index_df.Set[curr_row],
                                                                'TotalTokens':filt_vocab.shape[0],
                                                                'Numeric':filt_vocab.Numeric.sum(),
                                                                'Baseline':filt_vocab.Baseline.sum(),
                                                                'Discharge':filt_vocab.Discharge.sum(),
                                                                'Ordered':filt_vocab.Ordered.sum(),
                                                                'Binary':filt_vocab.Binary.sum(),
                                                                'ICUIntervention':filt_vocab.ICUIntervention.sum(),
                                                                'ClinicianInput':filt_vocab.ClinicianInput.sum()},index=[0]))
    return pd.concat(compiled_token_characteristics,ignore_index=True)

# Function to characterise tokens of each study window in legacy (v6-0)
def get_legacy_token_info(index_df,vocab_df,missing = True, progress_bar=True, progress_bar_desc=''):
    
    compiled_token_characteristics = []
    
    if progress_bar:
        iterator = tqdm(range(index_df.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(index_df.shape[0])
    
    for curr_row in iterator:
        
        curr_IndexList = index_df.VocabIndex[curr_row]
        
        if np.isnan(curr_IndexList).all():
            compiled_token_characteristics.append(pd.DataFrame({'GUPI':index_df.GUPI[curr_row],
                                                                'TimeStampStart':index_df.TimeStampStart[curr_row],
                                                                'TimeStampEnd':index_df.TimeStampEnd[curr_row],
                                                                'WindowIdx':index_df.WindowIdx[curr_row],
                                                                'WindowTotal':index_df.WindowTotal[curr_row],
                                                                'Set':index_df.Set[curr_row],
                                                                'TotalTokens':0,
                                                                'Numeric':0,
                                                                'Baseline':0,
                                                                'Ordered':0,
                                                                'Binary':0,
                                                                'ICUIntervention':0,
                                                                'ClinicianInput':0},index=[0]))
        elif (len(curr_IndexList) == 1):
            filt_vocab = vocab_df[vocab_df.VocabIndex.isin(curr_IndexList)].reset_index(drop=True)
            
            if not missing:
                filt_vocab = filt_vocab[~filt_vocab.Missing].reset_index(drop=True)
            
            if (filt_vocab.shape[0] == 0):
                compiled_token_characteristics.append(pd.DataFrame({'GUPI':index_df.GUPI[curr_row],
                                                                    'TimeStampStart':index_df.TimeStampStart[curr_row],
                                                                    'TimeStampEnd':index_df.TimeStampEnd[curr_row],
                                                                    'WindowIdx':index_df.WindowIdx[curr_row],
                                                                    'WindowTotal':index_df.WindowTotal[curr_row],
                                                                    'Set':index_df.Set[curr_row],
                                                                    'TotalTokens':0,
                                                                    'Numeric':0,
                                                                    'Baseline':0,
                                                                    'Ordered':0,
                                                                    'Binary':0,
                                                                    'ICUIntervention':0,
                                                                    'ClinicianInput':0},index=[0]))
            else:
                compiled_token_characteristics.append(pd.DataFrame({'GUPI':index_df.GUPI[curr_row],
                                                                    'TimeStampStart':index_df.TimeStampStart[curr_row],
                                                                    'TimeStampEnd':index_df.TimeStampEnd[curr_row],
                                                                    'WindowIdx':index_df.WindowIdx[curr_row],
                                                                    'WindowTotal':index_df.WindowTotal[curr_row],
                                                                    'Set':index_df.Set[curr_row],
                                                                    'TotalTokens':filt_vocab.shape[0],
                                                                    'Numeric':int(filt_vocab.Numeric[0]),
                                                                    'Baseline':int(filt_vocab.Baseline[0]),
                                                                    'Ordered':int(filt_vocab.Ordered[0]),
                                                                    'Binary':int(filt_vocab.Binary[0]),
                                                                    'ICUIntervention':int(filt_vocab.ICUIntervention[0]),
                                                                    'ClinicianInput':int(filt_vocab.ClinicianInput[0])},index=[0]))
            
        else:
            filt_vocab = vocab_df[vocab_df.VocabIndex.isin(curr_IndexList)].reset_index(drop=True)

            if not missing:
                filt_vocab = filt_vocab[~filt_vocab.Missing].reset_index(drop=True)

            compiled_token_characteristics.append(pd.DataFrame({'GUPI':index_df.GUPI[curr_row],
                                                                'TimeStampStart':index_df.TimeStampStart[curr_row],
                                                                'TimeStampEnd':index_df.TimeStampEnd[curr_row],
                                                                'WindowIdx':index_df.WindowIdx[curr_row],
                                                                'WindowTotal':index_df.WindowTotal[curr_row],
                                                                'Set':index_df.Set[curr_row],
                                                                'TotalTokens':filt_vocab.shape[0],
                                                                'Numeric':filt_vocab.Numeric.sum(),
                                                                'Baseline':filt_vocab.Baseline.sum(),
                                                                'Ordered':filt_vocab.Ordered.sum(),
                                                                'Binary':filt_vocab.Binary.sum(),
                                                                'ICUIntervention':filt_vocab.ICUIntervention.sum(),
                                                                'ClinicianInput':filt_vocab.ClinicianInput.sum()},index=[0]))
    return pd.concat(compiled_token_characteristics,ignore_index=True)

# Function to count specific token incidences of each study window
def count_token_incidences(index_df,curr_vocab,vocab_df,missing = True, progress_bar=True, progress_bar_desc=''):
    
    compiled_token_incidences = []
    
    if progress_bar:
        iterator = tqdm(index_df.GUPI.unique(),desc=progress_bar_desc)
    else:
        iterator = index_df.GUPI.unique()
        
    for curr_GUPI in iterator:
        
        filt_index_df = index_df[index_df.GUPI==curr_GUPI].reset_index(drop=True)
        full_index_list = list(itertools.chain.from_iterable(filt_index_df.VocabIndex.tolist()))
        full_vocab_list = curr_vocab.lookup_tokens(full_index_list)
        filt_vocab_df = vocab_df[vocab_df.VocabIndex.isin(np.unique(full_index_list))].reset_index(drop=True)
        
        if not missing:
            filt_vocab_df = filt_vocab_df[~filt_vocab_df.Missing].reset_index(drop=True)
            
        token_freqs = pd.DataFrame.from_dict(OrderedDict(Counter(full_vocab_list).most_common()),orient='index').reset_index().rename(columns={'index':'Token',0:'Count'})
        token_freqs = token_freqs[token_freqs.Token.isin(filt_vocab_df.Token)].reset_index(drop=True)
        token_freqs['GUPI'] = curr_GUPI
        
        compiled_token_incidences.append(token_freqs)
    
    return pd.concat(compiled_token_incidences,ignore_index=True)
        
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

def load_tokens(GUPIs,adm_or_disch,token_dir,progress_bar=True,progress_bar_desc=''):
    
    if progress_bar:
        iterator = tqdm(GUPIs,desc=progress_bar_desc)
    else:
        iterator = GUPIs
                
    return pd.concat([pd.read_csv(os.path.join(token_dir,'from_'+adm_or_disch+'_'+g+'.csv')) for g in iterator],ignore_index=True)

def convert_tokens(tokens_df,vocabulary,progress_bar=True,progress_bar_desc=''):
    
    if progress_bar:
        iterator = tqdm(range(tokens_df.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(tokens_df.shape[0])
        
    tokens_df['VocabIndex'] = [vocabulary.lookup_indices(tokens_df.Token[curr_row].split(' ')) for curr_row in iterator]
    
    tokens_df = tokens_df.drop(columns='Token')
    
    return tokens_df

def del_files(x,y):
    for f in x:
        os.remove(f)