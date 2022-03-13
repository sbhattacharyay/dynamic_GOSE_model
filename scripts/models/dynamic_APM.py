# Fundamental libraries
import os
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

# Define dynamic GOSE prediction model
class GOSE_model(pl.LightningModule):
    def __init__(self,n_tokens,latent_dim,embed_dropout,rnn_type,hidden_dim,rnn_layers,output_activation,learning_rate,class_weights,targets):
        """
        Args:
            n_tokens (int): Size of vocabulary
            latent_dim (int): Number of dimensions to which tokens are embedded
            embed_dropout (float): Probability of dropout layer on embedding vectors
            rnn_type (string, 'LSTM' or 'GRU'): Identify RNN architecture type
            hidden_dim (int): Number of dimensions in the RNN hidden state (output dimensionality of RNN)
            rnn_layers (int): Number of recurrent layers
            output_activation (string, 'softmax' or 'sigmoid'): Identifies output layer type based on output encoding
            learning_rate (float): Learning rate for ADAM optimizer
            class_weights (boolean): identifies whether loss should be weighted against class frequency
            targets (NumPy array): if class_weights == True, provides the class labels of the training set
        """
        super(GOSE_model, self).__init__()
        
        self.save_hyperparameters()
        
        self.n_tokens = n_tokens
        self.latent_dim = latent_dim
        self.dropout = embed_dropout
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.targets = targets
        
        self.embedX = nn.Embedding(n_tokens, latent_dim)
        self.embedW = nn.Embedding(n_tokens, 1)
        self.embed_Dropout = nn.Dropout(p = embed_dropout)
        
        if rnn_type == 'LSTM':
            self.rnn_module = nn.LSTM(input_size = latent_dim, hidden_size = hidden_dim, num_layers = rnn_layers)
        elif rnn_type == 'GRU':
            self.rnn_module = nn.GRU(input_size = latent_dim, hidden_size = hidden_dim, num_layers = rnn_layers)
        else:
            raise ValueError("Invalid RNN type. Must be 'LSTM' or 'GRU'")
            
        if self.output_activation == 'softmax': 
            self.hidden2gose = nn.Linear(hidden_dim,7)
        elif self.output_activation == 'sigmoid': 
            self.hidden2gose = nn.Linear(hidden_dim,6)
        else:
            raise ValueError("Invalid output layer type. Must be 'softmax' or 'sigmoid'")
            
    def forward(self,idx_list, bin_offsets, gupi_offsets):
        
        embeddedX = self.embedX(idx_list)
        
        # Constrain weights to be positive with exponentiation
        w = torch.exp(self.embedW(idx_list))
        
        # Iterate through infdividual bins and calculate weighted averages per bin
        embed_output = []
        for curr_bin_idx in torch.arange(0, len(bin_offsets), dtype=torch.long):
            if curr_bin_idx == (torch.LongTensor([len(bin_offsets) - 1])[0]):
                curr_bin_seq = torch.arange(bin_offsets[curr_bin_idx], embeddedX.shape[0], dtype=torch.long)
            else:
                curr_bin_seq = torch.arange(bin_offsets[curr_bin_idx], bin_offsets[curr_bin_idx+1], dtype=torch.long)
            embeddedX_avg = (embeddedX[curr_bin_seq,:] * w[curr_bin_seq]).sum(dim=0, keepdim=True) / (len(curr_bin_seq) + 1e-6)
            embed_output += [embeddedX_avg]
        embed_output = torch.cat(embed_output, dim=0)
        embed_output = self.embed_Dropout(F.relu(embed_output))
        
        # Iterate through unique patients to run sequences through LSTM and prediction networks
        rnn_linear_outputs = []
        for curr_gupi_idx in torch.arange(0, len(gupi_offsets), dtype=torch.long):
            if curr_gupi_idx == (torch.LongTensor([len(gupi_offsets) - 1])[0]):
                curr_gupi_seq = torch.arange(gupi_offsets[curr_gupi_idx], embed_output.shape[0], dtype=torch.long)
            else:
                curr_gupi_seq = torch.arange(gupi_offsets[curr_gupi_idx], gupi_offsets[curr_gupi_idx+1], dtype=torch.long)    
            curr_rnn_out, _ = self.rnn_module(embed_output[curr_gupi_seq,:].unsqueeze(1))
            curr_gose_out = self.hidden2gose(curr_rnn_out).squeeze(1)
            if self.output_activation == 'softmax': 
                rnn_linear_outputs += [curr_gose_out]
            elif self.output_activation == 'sigmoid': 
                mod_out = -F.relu(curr_gose_out.clone()[:,1:6])
                curr_gose_out[:,1:6] = mod_out
                rnn_linear_outputs += [curr_gose_out.cumsum(dim=1)]
            else:
                raise ValueError("Invalid output activation type. Must be 'softmax' or 'sigmoid'")
        rnn_linear_outputs = torch.cat(rnn_linear_outputs, dim=0)
        return rnn_linear_outputs, gupi_offsets
    
    def training_step(self, batch, batch_idx):
        
        # Get information from current batch
        curr_label_list, curr_idx_list, curr_bin_offsets, curr_gupi_offsets, curr_gupis = batch
        
        # Collect current model state outputs for the batch
        (yhat, out_gupi_offsets) = self(curr_idx_list, curr_bin_offsets, curr_gupi_offsets)
        
        # Calculate loss based on the output activation type
        if self.output_activation == 'softmax': 
            
            if self.class_weights:
                bal_weights = torch.from_numpy(compute_class_weight(class_weight='balanced',
                                                                    classes=np.sort(np.unique(self.targets)),
                                                                    y=self.targets)).type_as(yhat)
                loss = F.cross_entropy(yhat, curr_label_list, weight = bal_weights)
            else:
                loss = F.cross_entropy(yhat, curr_label_list)
                
        elif self.output_activation == 'sigmoid': 
            
            if self.class_weights:
                bal_weights = torch.from_numpy((self.targets.shape[0]
                                                - np.sum(self.targets, axis=0))
                                               / np.sum(self.targets,
                                                        axis=0)).type_as(yhat)
                
                loss = F.binary_cross_entropy_with_logits(yhat, curr_label_list.type_as(yhat), pos_weight = bal_weights)
            else:
                loss = F.binary_cross_entropy_with_logits(yhat, curr_label_list.type_as(yhat))
                
        else:
            raise ValueError("Invalid output layer type. Must be 'softmax' or 'sigmoid'")
        
        return {"loss": loss, "yhat": yhat, "curr_label_list": curr_label_list}
    
    def training_epoch_end(self, training_step_outputs):
        
        comp_loss = torch.tensor([x["loss"].detach() for x in training_step_outputs]).cpu().numpy().mean()
        comp_yhats = torch.vstack([x["yhat"].detach() for x in training_step_outputs])
        comp_label_list = torch.cat([x["curr_label_list"].detach() for x in training_step_outputs])
        
        curr_train_labels = torch.cat([comp_label_list],dim=0).cpu().numpy()
        
        if self.output_activation == 'softmax': 
            curr_train_probs = torch.cat([F.softmax(comp_yhats)],dim=0).cpu().numpy()
            aucs = []
            for ix, (a, b) in enumerate(itertools.combinations(np.sort(np.unique(curr_train_labels)), 2)):
                a_mask = curr_train_labels == a
                b_mask = curr_train_labels == b
                ab_mask = np.logical_or(a_mask,b_mask)
                condit_probs = curr_train_probs[ab_mask,b]/(curr_train_probs[ab_mask,a]+curr_train_probs[ab_mask,b]) 
                condit_probs = np.nan_to_num(condit_probs,nan=.5,posinf=1,neginf=0)
                condit_labels = b_mask[ab_mask].astype(int)
                aucs.append(roc_auc_score(condit_labels,condit_probs))            
            train_ORC = np.mean(aucs)
        elif self.output_activation == 'sigmoid': 
            curr_train_probs = torch.cat([F.sigmoid(comp_yhats)],dim=0).cpu().numpy()
            curr_train_labels = curr_train_labels.sum(1).astype(int)
            
            train_probs = np.empty([curr_train_probs.shape[0], curr_train_probs.shape[1]+1])
            train_probs[:,0] = 1 - curr_train_probs[:,0]
            train_probs[:,-1] = curr_train_probs[:,-1]
            
            for col_idx in range(1,(curr_train_probs.shape[1])):
                train_probs[:,col_idx] = curr_train_probs[:,col_idx-1] - curr_train_probs[:,col_idx]                
            
            aucs = []
            for ix, (a, b) in enumerate(itertools.combinations(np.sort(np.unique(curr_train_labels)), 2)):
                a_mask = curr_train_labels == a
                b_mask = curr_train_labels == b
                ab_mask = np.logical_or(a_mask,b_mask)
                condit_probs = train_probs[ab_mask,b]/(train_probs[ab_mask,a]+train_probs[ab_mask,b]) 
                condit_probs = np.nan_to_num(condit_probs,nan=.5,posinf=1,neginf=0)
                condit_labels = b_mask[ab_mask].astype(int)
                aucs.append(roc_auc_score(condit_labels,condit_probs))            
            train_ORC = np.mean(aucs)
                    
        self.log('train_ORC', train_ORC, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('train_loss', comp_loss, prog_bar=False, logger=True, sync_dist=True, on_step=False, on_epoch=True)
    
    def validation_step(self, batch, batch_idx):
        
        # Get information from current batch
        curr_label_list, curr_idx_list, curr_bin_offsets, curr_gupi_offsets, curr_gupis = batch
        
        # Collect current model state outputs for the batch
        (yhat, out_gupi_offsets) = self(curr_idx_list, curr_bin_offsets, curr_gupi_offsets)
        
        curr_val_labels = torch.cat([curr_label_list],dim=0).cpu().numpy()
                
        # Calculate loss based on the output activation type
        if self.output_activation == 'softmax': 
            
            curr_val_probs = torch.cat([F.softmax(yhat)],dim=0).cpu().numpy()
            
            val_loss = F.cross_entropy(yhat, curr_label_list)
            
            aucs = []
            for ix, (a, b) in enumerate(itertools.combinations(np.sort(np.unique(curr_val_labels)), 2)):
                a_mask = curr_val_labels == a
                b_mask = curr_val_labels == b
                ab_mask = np.logical_or(a_mask,b_mask)
                condit_probs = curr_val_probs[ab_mask,b]/(curr_val_probs[ab_mask,a]+curr_val_probs[ab_mask,b]) 
                condit_probs = np.nan_to_num(condit_probs,nan=.5,posinf=1,neginf=0)
                condit_labels = b_mask[ab_mask].astype(int)
                aucs.append(roc_auc_score(condit_labels,condit_probs))            
            val_ORC = np.mean(aucs)
                            
        elif self.output_activation == 'sigmoid': 
            
            curr_val_probs = torch.cat([F.sigmoid(yhat)],dim=0).cpu().numpy()
            curr_val_labels = curr_val_labels.sum(1).astype(int)
            
            val_probs = np.empty([curr_val_probs.shape[0], curr_val_probs.shape[1]+1])
            val_probs[:,0] = 1 - curr_val_probs[:,0]
            val_probs[:,-1] = curr_val_probs[:,-1]
            
            for col_idx in range(1,(curr_val_probs.shape[1])):
                val_probs[:,col_idx] = curr_val_probs[:,col_idx-1] - curr_val_probs[:,col_idx]                
            
            val_loss = F.binary_cross_entropy_with_logits(yhat, curr_label_list.type_as(yhat))
                
            aucs = []
            for ix, (a, b) in enumerate(itertools.combinations(np.sort(np.unique(curr_val_labels)), 2)):
                a_mask = curr_val_labels == a
                b_mask = curr_val_labels == b
                ab_mask = np.logical_or(a_mask,b_mask)
                condit_probs = val_probs[ab_mask,b]/(val_probs[ab_mask,a]+val_probs[ab_mask,b]) 
                condit_probs = np.nan_to_num(condit_probs,nan=.5,posinf=1,neginf=0)
                condit_labels = b_mask[ab_mask].astype(int)
                aucs.append(roc_auc_score(condit_labels,condit_probs))            
            val_ORC = np.mean(aucs)
                        
        else:
            raise ValueError("Invalid output layer type. Must be 'softmax' or 'sigmoid'")
        
        self.log('val_ORC', val_ORC, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss', val_loss, prog_bar=False, logger=True, sync_dist=True)

        return val_loss
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),lr=self.learning_rate)
        return optimizer