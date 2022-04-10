## SOURCE: https://github.com/gpleiss/temperature_scaling

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

# Torch methods
import torch
from torch import nn, optim
from torch.nn import functional as F

# SciKit-Learn methods
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

class TemperatureScaling(nn.Module):
    """
    """
    def __init__(self, val_preds):
        super(TemperatureScaling, self).__init__()
        self.val_preds = val_preds
        self.true_labels = torch.tensor(self.val_preds.TrueLabel.values,dtype=torch.int64)
        logit_cols = [col for col in self.val_preds if col.startswith('z_GOSE=')]
        self.logits = torch.tensor(self.val_preds[logit_cols].values,dtype=torch.float32)
        self.temperature = nn.Parameter(torch.ones(1) * 1.25)

    def forward(self, input):
        
        return self.temperature_scale(self.logits)
    
    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, optimization):
        """
        optimization (string): Either 'nominal' or 'ordinal'
        """
        if optimization == 'nominal':
            bal_weights = torch.from_numpy(compute_class_weight(class_weight='balanced',
                                                                classes=np.sort(np.unique(self.val_preds.TrueLabel)),
                                                                y=self.val_preds.TrueLabel)).type(torch.float32)
            criterion = nn.CrossEntropyLoss(weight=bal_weights)
            
        elif optimization == 'ordinal':
            thresh_cols = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
            thresh_label_matrix = self.val_preds[thresh_cols]            
            bal_weights = torch.from_numpy(((thresh_label_matrix.shape[0]-thresh_label_matrix.sum())/thresh_label_matrix.sum()).values).type(torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=bal_weights)
            
        else:
            raise ValueError("Invalid optimization type. Must be 'nominal' or 'ordinal'")
            
        # First, calculate loss before calibration
        if optimization == 'nominal':
            pre_calib_loss = criterion(self.logits, self.true_labels).item()
            print('Before calibration - Cross-Entropy Loss: %.3f' % (pre_calib_loss))
            
        elif optimization == 'ordinal':
            probs = F.softmax(self.logits)
            thresh_probs = []
            thresh_labels = []
            for thresh_idx in range(1,probs.shape[1]):
                prob_gt = probs[:,thresh_idx:].sum(1)
                thresh_probs.append(prob_gt)
                thresh_labels.append((self.true_labels >= thresh_idx).type_as(prob_gt))
            thresh_logits = torch.logit(torch.stack(thresh_probs,dim=1))
            thresh_labels = torch.stack(thresh_labels,dim=1)
            pre_calib_loss = criterion(thresh_logits, thresh_labels).item()
            print('Before calibration - Binary Entropy Loss: %.3f' % (pre_calib_loss))
            
        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=200)

        def eval():
            optimizer.zero_grad()
            if optimization == 'nominal':
                loss = criterion(self.temperature_scale(self.logits), self.true_labels)
            elif optimization == 'ordinal':
                probs = F.softmax(self.temperature_scale(self.logits))
                thresh_probs = []
                thresh_labels = []
                for thresh_idx in range(1,probs.shape[1]):
                    prob_gt = probs[:,thresh_idx:].sum(1)
                    thresh_probs.append(prob_gt)
                    thresh_labels.append((self.true_labels >= thresh_idx).type_as(prob_gt))
                thresh_logits = torch.logit(torch.stack(thresh_probs,dim=1))
                thresh_labels = torch.stack(thresh_labels,dim=1)
                loss = criterion(thresh_logits, thresh_labels)
            loss.backward()
            return loss
        optimizer.step(eval)
        
        # Calculate loss after temperature scaling
        if optimization == 'nominal':
            post_calib_loss = criterion(self.temperature_scale(self.logits), self.true_labels).item()
            print('After calibration - Cross-Entropy Loss: %.3f' % (post_calib_loss))
        
        elif optimization == 'ordinal':
            probs = F.softmax(self.temperature_scale(self.logits))
            thresh_probs = []
            thresh_labels = []
            for thresh_idx in range(1,probs.shape[1]):
                prob_gt = probs[:,thresh_idx:].sum(1)
                thresh_probs.append(prob_gt)
                thresh_labels.append((self.true_labels >= thresh_idx).type_as(prob_gt))
            thresh_logits = torch.logit(torch.stack(thresh_probs,dim=1))
            thresh_labels = torch.stack(thresh_labels,dim=1)
            post_calib_loss = criterion(thresh_logits, thresh_labels).item()
            print('After calibration - Binary Entropy Loss: %.3f' % (post_calib_loss))
            
        return self
    
class VectorScaling(nn.Module):
    """
    """
    def __init__(self, val_preds):
        super(VectorScaling, self).__init__()
        self.val_preds = val_preds
        self.true_labels = torch.tensor(self.val_preds.TrueLabel.values,dtype=torch.int64)
        logit_cols = [col for col in self.val_preds if col.startswith('z_GOSE=')]
        self.logits = torch.tensor(self.val_preds[logit_cols].values,dtype=torch.float32)
        self.vector = nn.Parameter(torch.ones(self.logits.shape[1],1))
        nn.init.xavier_uniform_(self.vector)
        self.biases = nn.Parameter((torch.zeros(self.logits.shape[1],1)))
        self.args = {'vector': self.vector,'biases':self.biases}
        
    def forward(self, input):
        
        return self.vector_scale(self.logits)
    
    def vector_scale(self, logits):
        """
        Perform vector scaling on logits
        """
        return torch.matmul(logits,torch.diag_embed(self.vector.squeeze(1))) + self.biases.squeeze(1)

    # This function probably should live outside of this class, but whatever
    def set_vector(self, optimization):
        """
        optimization (string): Either 'nominal' or 'ordinal'
        """
        if optimization == 'nominal':
            bal_weights = torch.from_numpy(compute_class_weight(class_weight='balanced',
                                                                classes=np.sort(np.unique(self.val_preds.TrueLabel)),
                                                                y=self.val_preds.TrueLabel)).type(torch.float32)
            criterion = nn.CrossEntropyLoss(weight=bal_weights)
            
        elif optimization == 'ordinal':
            thresh_cols = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
            thresh_label_matrix = self.val_preds[thresh_cols]            
            bal_weights = torch.from_numpy(((thresh_label_matrix.shape[0]-thresh_label_matrix.sum())/thresh_label_matrix.sum()).values).type(torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=bal_weights)
            
        else:
            raise ValueError("Invalid optimization type. Must be 'nominal' or 'ordinal'")
            
        # First, calculate loss before calibration
        if optimization == 'nominal':
            pre_calib_loss = criterion(self.logits, self.true_labels).item()
            print('Before calibration - Cross-Entropy Loss: %.3f' % (pre_calib_loss))
            
        elif optimization == 'ordinal':
            probs = F.softmax(self.logits)
            thresh_probs = []
            thresh_labels = []
            for thresh_idx in range(1,probs.shape[1]):
                prob_gt = probs[:,thresh_idx:].sum(1)
                thresh_probs.append(prob_gt)
                thresh_labels.append((self.true_labels >= thresh_idx).type_as(prob_gt))
            thresh_logits = torch.logit(torch.stack(thresh_probs,dim=1))
            thresh_labels = torch.stack(thresh_labels,dim=1)
            pre_calib_loss = criterion(thresh_logits, thresh_labels).item()
            print('Before calibration - Binary Entropy Loss: %.3f' % (pre_calib_loss))
            
        # Next: optimize the vector w.r.t. NLL
        optimizer = optim.LBFGS([self.vector,self.biases], lr=0.01, max_iter=200)
        
        def eval():
            optimizer.zero_grad()
            if optimization == 'nominal':
                loss = criterion(self.vector_scale(self.logits), self.true_labels)
            elif optimization == 'ordinal':
                probs = F.softmax(self.vector_scale(self.logits))
                thresh_probs = []
                thresh_labels = []
                for thresh_idx in range(1,probs.shape[1]):
                    prob_gt = probs[:,thresh_idx:].sum(1)
                    thresh_probs.append(prob_gt)
                    thresh_labels.append((self.true_labels >= thresh_idx).type_as(prob_gt))
                thresh_logits = torch.logit(torch.stack(thresh_probs,dim=1))
                thresh_labels = torch.stack(thresh_labels,dim=1)
                loss = criterion(thresh_logits, thresh_labels)
            loss.backward()
            return loss
        optimizer.step(eval)
        
        # Calculate loss after vector scaling
        if optimization == 'nominal':
            post_calib_loss = criterion(self.vector_scale(self.logits), self.true_labels).item()
            print('After calibration - Cross-Entropy Loss: %.3f' % (post_calib_loss))
        
        elif optimization == 'ordinal':
            probs = F.softmax(self.vector_scale(self.logits))
            thresh_probs = []
            thresh_labels = []
            for thresh_idx in range(1,probs.shape[1]):
                prob_gt = probs[:,thresh_idx:].sum(1)
                thresh_probs.append(prob_gt)
                thresh_labels.append((self.true_labels >= thresh_idx).type_as(prob_gt))
            thresh_logits = torch.logit(torch.stack(thresh_probs,dim=1))
            thresh_labels = torch.stack(thresh_labels,dim=1)
            post_calib_loss = criterion(thresh_logits, thresh_labels).item()
            print('After calibration - Binary Entropy Loss: %.3f' % (post_calib_loss))
            
        return self