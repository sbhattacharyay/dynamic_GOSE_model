#### Master Script 25a: Compute TimeSHAP for dynAPM_DeepMN ####
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
import math
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

# Import TimeSHAP methods
import timeshap
from timeshap.wrappers import TorchModelWrapper

# Import TimeSHAP-specific themes
import altair as alt
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
from timeshap.plot import timeshap_theme
alt.themes.register("timeshap_theme", timeshap_theme)
alt.themes.enable("timeshap_theme")

# Custom methods
from classes.datasets import DYN_ALL_PREDICTOR_SET
from models.dynamic_APM import GOSE_model, timeshap_GOSE_model, derive_embedded_vector
from functions.model_building import format_shap, format_tokens, format_time_tokens

# Set version code
VERSION = 'v6-0'

############################################################
# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Load the current version tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

# Load cross-validation split information to extract testing resamples
cv_splits = pd.read_csv('../cross_validation_splits.csv')
test_splits = cv_splits[cv_splits.set == 'test'].rename(columns={'repeat':'REPEAT','fold':'FOLD','set':'SET'}).reset_index(drop=True)
uniq_GUPIs = test_splits.GUPI.unique()

# Define vector of GOSE thresholds
GOSE_thresholds = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
ckpt_info = pd.read_pickle(os.path.join('/home/sb2406/rds/hpc-work/model_interpretations/'+VERSION,'LBM','ckpt_info.pkl'))

array_task_id=0
# Extract current file, repeat, and fold information
curr_file = ckpt_info.file[array_task_id]
curr_tune_idx = ckpt_info.TUNE_IDX[array_task_id]
curr_repeat = ckpt_info.REPEAT[array_task_id]
curr_fold = ckpt_info.FOLD[array_task_id]
curr_gupi = ckpt_info.GUPI[array_task_id]
curr_threshold_idx = ckpt_info.THRESHOLD_IDX[array_task_id]

# Define current fold directory based on current information
tune_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold),'tune'+str(curr_tune_idx).zfill(4))

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

# Embed multihot matrix
embedded_matrix = derive_embedded_vector(gose_model,unknown_index,cols_to_add,multihot_matrix)

# Initialize custom TimeSHAP model
ts_GOSE_model = timeshap_GOSE_model(gose_model,curr_threshold_idx)
wrapped_gose_model = TorchModelWrapper(ts_GOSE_model)
f_hs = lambda x, y=None: wrapped_gose_model.predict_last_hs(x, y)

from timeshap.utils import calc_avg_event
average_event = np.zeros((1,128))

from timeshap.utils import get_avg_score_with_avg_event
avg_score_over_len = get_avg_score_with_avg_event(f_hs, average_event, top=embedded_matrix.shape[0]+1)

from timeshap.explainer import local_report
plot_feats = dict(zip([str(i) for i in range(1,129)], [str(i) for i in range(1,129)]))
pruning_dict = {'tol': 0.025}
event_dict = {'rs': 42, 'nsamples': 32000}
feature_dict = {'rs': 42, 'nsamples': 32000, 'feature_names': [str(i) for i in range(1,129)], 'plot_features': plot_feats}
cell_dict = {'rs': 42, 'nsamples': 32000, 'top_x_feats': 2, 'top_x_events': 2}
local_report(f_hs, embedded_matrix.unsqueeze(0).numpy(), pruning_dict, event_dict, feature_dict, cell_dict=cell_dict, entity_uuid='trial', entity_col='all_id', baseline=average_event)

################## SAMPLE
data_directories = next(os.walk("/home/sb2406/python_venv/bin/timeshap/notebooks/AReM/AReM"))[1]

all_csvs = []
for folder in data_directories:
    if folder in ['bending1', 'bending2']:
        continue
    folder_csvs = next(os.walk(f"/home/sb2406/python_venv/bin/timeshap/notebooks/AReM/AReM/{folder}"))[2]
    for data_csv in folder_csvs:
        if data_csv == 'dataset8.csv' and folder == 'sitting':
            # this dataset only has 479 instances
            # it is possible to use it, but would require padding logic
            continue
        loaded_data = pd.read_csv(f"/home/sb2406/python_venv/bin/timeshap/notebooks/AReM/AReM/{folder}/{data_csv}", skiprows=4)
        print(f"{folder}/{data_csv} ------ {loaded_data.shape}")
        
        csv_id = re.findall(r'\d+', data_csv)[0]
        loaded_data['id'] = csv_id
        loaded_data['all_id'] = f"{folder}_{csv_id}"
        loaded_data['activity'] = folder
        all_csvs.append(loaded_data)

all_data = pd.concat(all_csvs)
raw_model_features = ['avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23']
all_data.columns = ['timestamp', 'avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23', 'id', 'all_id', 'activity']

# choose ids to use for test
ids_for_test = np.random.choice(all_data['id'].unique(), size=4, replace=False)

d_train =  all_data[~all_data['id'].isin(ids_for_test)]
d_test = all_data[all_data['id'].isin(ids_for_test)]

class NumericalNormalizer:
    def __init__(self, fields: list):
        self.metrics = {}
        self.fields = fields

    def fit(self, df: pd.DataFrame ) -> list:
        means = df[self.fields].mean()
        std = df[self.fields].std()
        for field in self.fields:
            field_mean = means[field]
            field_stddev = std[field]
            self.metrics[field] = {'mean': field_mean, 'std': field_stddev}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Transform to zero-mean and unit variance.
        for field in self.fields:
            f_mean = self.metrics[field]['mean']
            f_stddev = self.metrics[field]['std']
            # OUTLIER CLIPPING to [avg-3*std, avg+3*avg]
            df[field] = df[field].apply(lambda x: f_mean - 3 * f_stddev if x < f_mean - 3 * f_stddev else x)
            df[field] = df[field].apply(lambda x: f_mean + 3 * f_stddev if x > f_mean + 3 * f_stddev else x)
            if f_stddev > 1e-5:
                df[f'p_{field}_normalized'] = df[field].apply(lambda x: ((x - f_mean)/f_stddev))
            else:
                df[f'p_{field}_normalized'] = df[field].apply(lambda x: x * 0)
        return df
    
normalizor = NumericalNormalizer(raw_model_features)
normalizor.fit(d_train)
d_train_normalized = normalizor.transform(d_train)
d_test_normalized = normalizor.transform(d_test)

model_features = [f"p_{x}_normalized" for x in raw_model_features]
time_feat = 'timestamp'
label_feat = 'activity'
sequence_id_feat = 'all_id'

chosen_activity = 'cycling'

d_train_normalized['label'] = d_train_normalized['activity'].apply(lambda x: int(x == chosen_activity))
d_test_normalized['label'] = d_test_normalized['activity'].apply(lambda x: int(x == chosen_activity))

def df_to_Tensor(df, model_feats, label_feat, group_by_feat, timestamp_Feat):
    sequence_length = len(df[timestamp_Feat].unique())
    
    data_tensor = np.zeros((len(df[group_by_feat].unique()), sequence_length, len(model_feats)))
    labels_tensor = np.zeros((len(df[group_by_feat].unique()), 1))
    
    for i, name in enumerate(df[group_by_feat].unique()):
        name_data = df[df[group_by_feat] == name]
        sorted_data = name_data.sort_values(timestamp_Feat)
        
        data_x = sorted_data[model_feats].values
        labels = sorted_data[label_feat].values
        assert labels.sum() == 0 or labels.sum() == len(labels)
        data_tensor[i, :, :] = data_x
        labels_tensor[i, :] = labels[0]
    data_tensor = torch.from_numpy(data_tensor).type(torch.FloatTensor)
    labels_tensor = torch.from_numpy(labels_tensor).type(torch.FloatTensor)
    
    return data_tensor, labels_tensor

train_data, train_labels = df_to_Tensor(d_train_normalized, model_features, 'label', sequence_id_feat, time_feat)
test_data, test_labels = df_to_Tensor(d_test_normalized, model_features, 'label', sequence_id_feat, time_feat)

import torch.nn as nn
class ExplainedRNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 cfg: dict,
                 ):
        super(ExplainedRNN, self).__init__()
        self.hidden_dim = cfg.get('hidden_dim', 32)
        torch.manual_seed(cfg.get('random_seed', 42))

        self.recurrent_block = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_dim,
            batch_first=True,
            num_layers=2,
            )
        
        self.classifier_block = nn.Linear(self.hidden_dim, 1)
        self.output_activation_func = nn.Sigmoid()

    def forward(self,
                x: torch.Tensor,
                hidden_states: tuple = None,
                ):
        
        print(x.shape)
        
        if hidden_states is None:
            output, hidden = self.recurrent_block(x)
        else:
            output, hidden = self.recurrent_block(x, hidden_states)

        # -1 on hidden, to select the last layer of the stacked gru
        assert torch.equal(output[:,-1,:], hidden[-1, :, :])
        
        y = self.classifier_block(hidden[-1, :, :])
        y = self.output_activation_func(y)
        return y, hidden
    
import torch.optim as optim

model = ExplainedRNN(len(model_features), {})
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

learning_rate = 0.005
EPOCHS = 8

import tqdm
import copy
for epoch in tqdm.trange(EPOCHS):
    train_data_local = copy.deepcopy(train_data)
    train_labels_local = copy.deepcopy(train_labels)
    
    y_pred, hidden_states = model(train_data_local)
    train_loss = loss_function(y_pred, train_labels_local)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    with torch.no_grad():
        test_data_local = copy.deepcopy(test_data)
        test_labels_local = copy.deepcopy(test_labels)
        test_preds, _ = model(test_data_local)
        test_loss = loss_function(test_preds, test_labels_local)
        print(f"Train loss: {train_loss.item()} --- Test loss {test_loss.item()} ")
        
from timeshap.wrappers import TorchModelWrapper
model_wrapped = TorchModelWrapper(model)
f_hs = lambda x, y=None: model_wrapped.predict_last_hs(x, y)

from timeshap.utils import calc_avg_event
average_event = calc_avg_event(d_train_normalized, numerical_feats=model_features, categorical_feats=[])

from timeshap.utils import get_avg_score_with_avg_event
avg_score_over_len = get_avg_score_with_avg_event(f_hs, average_event, top=480)

