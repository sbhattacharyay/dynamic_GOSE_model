# Fundamental libraries
import os
import sys
import time
import glob
import copy
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
warnings.filterwarnings(action="ignore")

# PyTorch, PyTorch.Text, and Lightning-PyTorch methods
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# SciKit-Learn methods
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# Define model to train LBM step 1 mask
class LBM_step1_model(pl.LightningModule):
    def __init__(self,threshold_idx,gose_model,cols_to_add,unknown_index,A,T,reg_1,reg_2,lr):
        """
        Args:
            gose_model (LightningModule): trained dynAPM model from which to extract prediction layers
            threshold_idx (int): index of the GOSE threshold to focus on for LBM
            cols_to_add (int): Number of rows to add to embedding layer to account for unknown indices
            unknown_index (int): Embedding layer index corresponding to '<unk>' token
            A (float): constant for sigmoid transformation of z to mask
            T (int): number of total timesteps in current patient matrix
            reg_1 (float): Regularization parameter for mask values
            reg_2 (float): Regularization parameter for binary cross entropy loss
            lr (float): Initial learning rate
        """
        super(LBM_step1_model, self).__init__()
        
        # Save all hyperparameters except GOSE Model
        self.save_hyperparameters(ignore='gose_model')

        # Extract trained initial embedding layer and modify for LBM calculation 
        self.embedX = copy.deepcopy(gose_model).embedX
        self.embedX.weight.requires_grad = False
        self.embedX.weight = nn.Parameter(torch.cat((self.embedX.weight,torch.tile(self.embedX.weight[unknown_index,:],(cols_to_add,1))),dim=0),requires_grad=False)
        
        # Extract trained weighting embedding layer and modify for LBM calculation 
        self.embedW = copy.deepcopy(gose_model).embedW
        self.embedW.weight.requires_grad = False
        self.embedW.weight = nn.Parameter(torch.cat((self.embedW.weight,torch.tile(self.embedW.weight[unknown_index,:],(cols_to_add,1))),dim=0),requires_grad=False)
        
        # Extract trained RNN module and modify for LBM calculation
        self.rnn_module = copy.deepcopy(gose_model).rnn_module
        for name, param in self.rnn_module.named_parameters():
            param.requires_grad = False
        
        # Extract trained output layer and modify for LBM calculation
        self.hidden2gose = copy.deepcopy(gose_model).hidden2gose
        for name, param in self.hidden2gose.named_parameters():
            param.requires_grad = False
            
        # Initialize z-parameter
        self.A = A
        self.T = T
        self.z = nn.Parameter(torch.ones(self.T,self.embedX.weight.shape[0]),requires_grad=True)
        
        # Save regularization parameters
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        
        # Save initial learning rate
        self.lr = lr
        
        # Save threshold idx of focus
        self.threshold_idx = threshold_idx
        
        # Define a variable to store the number of learning rate reductions
        self.lr_reductions = 0
        
        # Define variable to save number of bad epochs
        self.bad_epoch_count = 0
        
        # Initialize an artificially curr_train_loss
        self.curr_train_loss = 1000000
        
    # Custom function to convert patient matrix to dt matrix
    def pt_to_dt_matrix(self,pt_matrix):
        return torch.diff(pt_matrix,prepend=torch.zeros(1,pt_matrix.shape[1]),dim=0)
    
    # Custom function to convert dt matrix to patient matrix
    def dt_to_pt_matrix(self,dt_matrix):
        return torch.cumsum(dt_matrix,dim=0)
    
    # Custom function to calculate mask from current z-parameter
    def get_m(self):
        return F.sigmoid(self.A*self.z)
    
    # Custom function to calculate loss
    def loss_fn(self,output_vector):
        
        # Get the current mask
        curr_m = self.get_m()
        
        # Return loss value
        return (torch.linalg.vector_norm(output_vector,ord=1)) + (self.reg_1*torch.linalg.matrix_norm(1 - torch.t(curr_m),ord=1)) + (self.reg_2*F.binary_cross_entropy(curr_m, (curr_m > 0.5).to(dtype=torch.float)))
    
    # Forward pass of LBM model
    def forward(self,batch):
        
        # Convert patient matrix to dt form
        curr_dt_matrix = self.pt_to_dt_matrix(batch)
        
        # Get the current mask
        curr_m = self.get_m()
        
        # Mask the dt matrix and convert to patient matrix
        curr_pt_matrix = self.dt_to_pt_matrix(curr_dt_matrix*curr_m)
        
        # Weigh embedding layers
        comb_embedding = self.embedX.weight*torch.tile(torch.exp(self.embedW.weight),(1,self.embedX.weight.shape[1]))
        curr_embedding_out = F.relu(torch.matmul(curr_pt_matrix,comb_embedding) / torch.tile(curr_pt_matrix.sum(1).unsqueeze(1),(1,comb_embedding.shape[1])))
        
        # Obtain RNN output and transform to GOSE space
        curr_rnn_out, _ = self.rnn_module(curr_embedding_out.unsqueeze(1))
        curr_gose_out = self.hidden2gose(curr_rnn_out).squeeze(1)
        
        # Calculate predicted probability outputs per GOSE
        return F.softmax(curr_gose_out).cumsum(1)[:,self.threshold_idx]
        
    # Function to calculate loss at each training step
    def training_step(self, batch, batch_idx):
        
        # Collect current predicted probability outputs for the selected class index
        prob_outputs = self(batch)
        
        # Calculate the current loss
        loss = self.loss_fn(prob_outputs)
        
        # Logging and returning loss value
        return {"loss": loss, "prob_outputs": prob_outputs}
    
    # End of training epoch
    def training_epoch_end(self, training_step_outputs):
        
        # Compile and take mean of loss
        self.curr_train_loss = torch.tensor([x["loss"].detach() for x in training_step_outputs]).cpu().numpy().mean()
        self.curr_train_prob_outputs = torch.vstack([x["prob_outputs"].detach() for x in training_step_outputs])
        if self.scheduler.is_better(float(self.curr_train_loss), self.scheduler.best):
            self.bad_epoch_count = 0
        else:
            self.bad_epoch_count += 1
        if self.bad_epoch_count > 5:
            self.bad_epoch_count = 0
            self.lr_reductions += 1
        self.log('train_loss', self.curr_train_loss, prog_bar=False, logger=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('lr_reductions', self.lr_reductions, prog_bar=False, logger=True, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.parameters()),lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',threshold=0.0001,threshold_mode='abs',patience=5,factor=0.1,verbose=True)
        #return ({'optimizer': optimizer,'lr_scheduler':{'scheduler':self.scheduler,'monitor': 'val_loss','frequency':1}})
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=None,using_native_amp=None,using_lbfgs=None):
        
        if batch_idx == 0: # to call the scheduler after each validation                
            self.scheduler.step(self.curr_train_loss)
            print(f'train_loss: {self.curr_train_loss}, best: {self.scheduler.best}, num_bad_epochs: {self.scheduler.num_bad_epochs}') # for debugging
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.log('lr_reductions', self.lr_reductions, prog_bar=False, logger=True, sync_dist=True)