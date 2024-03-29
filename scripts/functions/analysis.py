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
from pandas.api.types import CategoricalDtype
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
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

# Function to calculate ORC on validation set predictions
def calc_val_ORC(pred_df,window_indices,progress_bar = True,progress_bar_desc = ''):
    orcs = []
    if progress_bar:
        iterator = tqdm(pred_df.TUNE_IDX.unique(),desc=progress_bar_desc)
    else:
        iterator = pred_df.TUNE_IDX.unique()
    for curr_tune_idx in iterator:
        for curr_wi in window_indices:
            filt_is_preds = pred_df[(pred_df.WindowIdx == curr_wi)&(pred_df.TUNE_IDX == curr_tune_idx)].reset_index(drop=True)
            aucs = []
            for ix, (a, b) in enumerate(itertools.combinations(np.sort(filt_is_preds.TrueLabel.unique()), 2)):
                filt_prob_matrix = filt_is_preds[filt_is_preds.TrueLabel.isin([a,b])].reset_index(drop=True)
                filt_prob_matrix['ConditLabel'] = (filt_prob_matrix.TrueLabel == b).astype(int)
                aucs.append(roc_auc_score(filt_prob_matrix['ConditLabel'],filt_prob_matrix['ExpectedValue']))
            orcs.append(pd.DataFrame({'TUNE_IDX':curr_tune_idx,
                                      'WINDOW_IDX':curr_wi,
                                      'METRIC':'ORC',
                                      'VALUE':np.mean(aucs)},index=[0]))
    return pd.concat(orcs,ignore_index=True)

# Function to calculate ORC on testing set predictions
def calc_test_ORC(pred_df,window_indices,progress_bar = True,progress_bar_desc = ''):
    orcs = []
    if progress_bar:
        iterator = tqdm(pred_df.TUNE_IDX.unique(),desc=progress_bar_desc)
    else:
        iterator = pred_df.TUNE_IDX.unique()
    for curr_tune_idx in iterator:
        for curr_wi in window_indices:
            filt_is_preds = pred_df[(pred_df.WindowIdx == curr_wi)&(pred_df.TUNE_IDX == curr_tune_idx)].reset_index(drop=True)
            aucs = []
            for ix, (a, b) in enumerate(itertools.combinations(np.sort(filt_is_preds.TrueLabel.unique()), 2)):
                filt_prob_matrix = filt_is_preds[filt_is_preds.TrueLabel.isin([a,b])].reset_index(drop=True)
                filt_prob_matrix['ConditLabel'] = (filt_prob_matrix.TrueLabel == b).astype(int)
                aucs.append(roc_auc_score(filt_prob_matrix['ConditLabel'],filt_prob_matrix['ExpectedValue']))
            orcs.append(pd.DataFrame({'TUNE_IDX':curr_tune_idx,
                                      'WINDOW_IDX':curr_wi,
                                      'METRIC':'ORC',
                                      'VALUE':np.mean(aucs)},index=[0]))
    return pd.concat(orcs,ignore_index=True)

# Function to calculate threshold-level calibration metrics
def calc_val_thresh_calibration(pred_df,window_indices,progress_bar = True,progress_bar_desc = ''):
    
    prob_cols = [col for col in pred_df if col.startswith('Pr(GOSE=')]
    thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
    calib_metrics = []
    
    if progress_bar:
        iterator = tqdm(pred_df.TUNE_IDX.unique(),desc=progress_bar_desc)
    else:
        iterator = pred_df.TUNE_IDX.unique()
    
    for thresh in range(1,len(prob_cols)):
        cols_gt = prob_cols[thresh:]
        prob_gt = pred_df[cols_gt].sum(1).values
        gt = (pred_df['TrueLabel'] >= thresh).astype(int).values
        pred_df['Pr('+thresh_labels[thresh-1]+')'] = prob_gt
        pred_df[thresh_labels[thresh-1]] = gt
    
    for curr_tune_idx in iterator:
        for curr_wi in window_indices:
            filt_is_preds = pred_df[(pred_df.WindowIdx == curr_wi)&(pred_df.TUNE_IDX == curr_tune_idx)].reset_index(drop=True)
            for thresh in thresh_labels:
                thresh_prob_name = 'Pr('+thresh+')'
                logit_gt = np.nan_to_num(logit(filt_is_preds[thresh_prob_name]),neginf=-100,posinf=100)
                calib_glm = Logit(filt_is_preds[thresh], add_constant(logit_gt))
                calib_glm_res = calib_glm.fit(disp=False)
#                 thresh_calib_linspace = np.linspace(filt_is_preds[thresh_prob_name].min(),filt_is_preds[thresh_prob_name].max(),200)
#                 TrueProb = lowess(endog = filt_is_preds[thresh], exog = filt_is_preds[thresh_prob_name], it = 0, xvals = thresh_calib_linspace)
#                 filt_is_preds['TruePr('+thresh+')'] = filt_is_preds[thresh_prob_name].apply(lambda x: TrueProb[(np.abs(x - thresh_calib_linspace)).argmin()])
#                 ICI = (filt_is_preds['TruePr('+thresh+')'] - filt_is_preds[thresh_prob_name]).abs().mean()
#                 Emax = (filt_is_preds['TruePr('+thresh+')'] - filt_is_preds[thresh_prob_name]).abs().max()
                calib_metrics.append(pd.DataFrame({'TUNE_IDX':curr_tune_idx,
                                                   'WINDOW_IDX':curr_wi,
                                                   'THRESHOLD':thresh,
                                                   'METRIC':'CALIB_SLOPE',
                                                   'VALUE':calib_glm_res.params[1]},
                                                 index=[0]))
    calib_metrics = pd.concat(calib_metrics,ignore_index = True).reset_index(drop=True)
    return calib_metrics

# Function to calculate threshold-level calibration metrics
def calc_test_thresh_calibration(pred_df,window_indices,progress_bar = True,progress_bar_desc = ''):
    
    prob_cols = [col for col in pred_df if col.startswith('Pr(GOSE=')]
    thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
    calib_metrics = []
    
    if progress_bar:
        iterator = tqdm(pred_df.TUNE_IDX.unique(),desc=progress_bar_desc)
    else:
        iterator = pred_df.TUNE_IDX.unique()
    
    for thresh in range(1,len(prob_cols)):
        cols_gt = prob_cols[thresh:]
        prob_gt = pred_df[cols_gt].sum(1).values
        gt = (pred_df['TrueLabel'] >= thresh).astype(int).values
        pred_df['Pr('+thresh_labels[thresh-1]+')'] = prob_gt
        pred_df[thresh_labels[thresh-1]] = gt
    
    for curr_tune_idx in iterator:
        for curr_wi in window_indices:
            filt_is_preds = pred_df[(pred_df.WindowIdx == curr_wi)&(pred_df.TUNE_IDX == curr_tune_idx)].reset_index(drop=True)
            for thresh in thresh_labels:
                thresh_prob_name = 'Pr('+thresh+')'
                try:
                    logit_gt = np.nan_to_num(logit(filt_is_preds[thresh_prob_name]),neginf=-100,posinf=100)
                    calib_glm = Logit(filt_is_preds[thresh], add_constant(logit_gt))
                    calib_glm_res = calib_glm.fit(disp=False)
                    curr_calib_slope = calib_glm_res.params[1]
                except:
                    curr_calib_slope = np.nan
                thresh_calib_linspace = np.linspace(filt_is_preds[thresh_prob_name].min(),filt_is_preds[thresh_prob_name].max(),200)
                TrueProb = lowess(endog = filt_is_preds[thresh], exog = filt_is_preds[thresh_prob_name], it = 0, xvals = thresh_calib_linspace)
                filt_is_preds['TruePr('+thresh+')'] = filt_is_preds[thresh_prob_name].apply(lambda x: TrueProb[(np.abs(x - thresh_calib_linspace)).argmin()])
                ICI = (filt_is_preds['TruePr('+thresh+')'] - filt_is_preds[thresh_prob_name]).abs().mean()
                Emax = (filt_is_preds['TruePr('+thresh+')'] - filt_is_preds[thresh_prob_name]).abs().max()
                calib_metrics.append(pd.DataFrame({'TUNE_IDX':curr_tune_idx,
                                                   'WINDOW_IDX':curr_wi,
                                                   'THRESHOLD':thresh,
                                                   'METRIC':['CALIB_SLOPE','ICI','Emax'],
                                                   'VALUE':[curr_calib_slope,ICI,Emax]}))
    calib_metrics = pd.concat(calib_metrics,ignore_index = True).reset_index(drop=True)
    return calib_metrics

# Function to calculate Somers_D on testing set predictions
def calc_test_Somers_D(pred_df,window_indices,progress_bar = True,progress_bar_desc = ''):
    somers_d = []
    if progress_bar:
        iterator = tqdm(pred_df.TUNE_IDX.unique(),desc=progress_bar_desc)
    else:
        iterator = pred_df.TUNE_IDX.unique()
    for curr_tune_idx in iterator:
        for curr_wi in window_indices:
            filt_is_preds = pred_df[(pred_df.WindowIdx == curr_wi)&(pred_df.TUNE_IDX == curr_tune_idx)].reset_index(drop=True)
            aucs = []
            prevalence = []
            for ix, (a, b) in enumerate(itertools.combinations(np.sort(filt_is_preds.TrueLabel.unique()), 2)):
                filt_prob_matrix = filt_is_preds[filt_is_preds.TrueLabel.isin([a,b])].reset_index(drop=True)
                filt_prob_matrix['ConditLabel'] = (filt_prob_matrix.TrueLabel == b).astype(int)
                prevalence.append((filt_prob_matrix.TrueLabel == a).sum()*(filt_prob_matrix.TrueLabel == b).sum())
                aucs.append(roc_auc_score(filt_prob_matrix['ConditLabel'],filt_prob_matrix['ExpectedValue']))
            somers_d.append(pd.DataFrame({'TUNE_IDX':curr_tune_idx,
                                          'WINDOW_IDX':curr_wi,
                                          'METRIC':'Somers D',
                                          'VALUE':2*(np.sum(np.multiply(aucs,prevalence))/np.sum(prevalence))-1},index=[0]))
    return pd.concat(somers_d,ignore_index=True)

# Function to calculate threshold-level calibration curves
def calc_test_thresh_calib_curves(pred_df,window_indices,progress_bar = True,progress_bar_desc = ''):
    
    prob_cols = [col for col in pred_df if col.startswith('Pr(GOSE=')]
    thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
    thresh_calib_linspace = np.linspace(0,1,200)
    calib_curves = []
    
    if progress_bar:
        iterator = tqdm(pred_df.TUNE_IDX.unique(),desc=progress_bar_desc)
    else:
        iterator = pred_df.TUNE_IDX.unique()
    
    for thresh in range(1,len(prob_cols)):
        cols_gt = prob_cols[thresh:]
        prob_gt = pred_df[cols_gt].sum(1).values
        gt = (pred_df['TrueLabel'] >= thresh).astype(int).values
        pred_df['Pr('+thresh_labels[thresh-1]+')'] = prob_gt
        pred_df[thresh_labels[thresh-1]] = gt
    
    for curr_tune_idx in iterator:
        for curr_wi in window_indices:
            filt_is_preds = pred_df[(pred_df.WindowIdx == curr_wi)&(pred_df.TUNE_IDX == curr_tune_idx)].reset_index(drop=True)
            for thresh in thresh_labels:
                thresh_prob_name = 'Pr('+thresh+')'
                TrueProb = lowess(endog = filt_is_preds[thresh], exog = filt_is_preds[thresh_prob_name], it = 0, xvals = thresh_calib_linspace)
                calib_curves.append(pd.DataFrame({'TUNE_IDX':curr_tune_idx,
                                                  'WINDOW_IDX':curr_wi,
                                                  'THRESHOLD':thresh,
                                                  'PREDPROB':thresh_calib_linspace,
                                                  'TRUEPROB':TrueProb}))
    calib_curves = pd.concat(calib_curves,ignore_index = True).reset_index(drop=True)
    return calib_curves

# Function to load and compile test performance metrics for DeepIMPACT models
def collect_metrics(metric_file_info,progress_bar = True, progress_bar_desc = ''):
    output_df = []
    if progress_bar:
        iterator = tqdm(metric_file_info.file,desc=progress_bar_desc)
    else:
        iterator = metric_file_info.file
    return pd.concat([pd.read_csv(f) for f in iterator],ignore_index=True)

# Function to load and compile test calibration curves for DeepIMPACT models
def collect_calib_curves(curve_file_info,progress_bar = True, progress_bar_desc = ''):
    output_df = []
    if progress_bar:
        iterator = tqdm(curve_file_info.file,desc=progress_bar_desc)
    else:
        iterator = curve_file_info.file
    return pd.concat([pd.read_pickle(f) for f in iterator],ignore_index=True)

# Function to calculate ECE
def calc_ECE(preds):
    prob_cols = [col for col in preds if col.startswith('Pr(GOSE=')]
    preds['PredLabel'] = np.argmax(preds[prob_cols].to_numpy(),axis=1)
    preds['Confidence'] = preds[prob_cols].max(axis=1)
    preds['Hit'] = (preds.PredLabel == preds.TrueLabel).astype(int)
    confidence_linspace = np.linspace(preds.Confidence.min(),preds.Confidence.max(),200)
    smooth_accuracy = lowess(endog = preds['Hit'], exog = preds['Confidence'], it = 0, xvals = confidence_linspace)
    preds['Smooth_Accuracy'] = preds['Confidence'].apply(lambda x: smooth_accuracy[(np.abs(x - confidence_linspace)).argmin()])
    overall_ECE = (preds['Smooth_Accuracy'] - preds['Confidence']).abs().mean()
    return overall_ECE

# Function to calculate MCE
def calc_MCE(preds):
    prob_cols = [col for col in preds if col.startswith('Pr(GOSE=')]
    preds['PredLabel'] = np.argmax(preds[prob_cols].to_numpy(),axis=1)
    preds['Confidence'] = preds[prob_cols].max(axis=1)
    preds['Hit'] = (preds.PredLabel == preds.TrueLabel).astype(int)
    confidence_linspace = np.linspace(preds.Confidence.min(),preds.Confidence.max(),200)
    smooth_accuracy = lowess(endog = preds['Hit'], exog = preds['Confidence'], it = 0, xvals = confidence_linspace)
    preds['Smooth_Accuracy'] = preds['Confidence'].apply(lambda x: smooth_accuracy[(np.abs(x - confidence_linspace)).argmin()])
    overall_MCE = (preds['Smooth_Accuracy'] - preds['Confidence']).abs().max()
    return overall_MCE

# Function to collect LBM values
def collect_LBM(LBM_file_info,progress_bar = True, progress_bar_desc = ''):
    output_df = []
    if progress_bar:
        iterator = tqdm(range(LBM_file_info.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(LBM_file_info.shape[0])
        
    for curr_idx in iterator:
        curr_LBM_df = pd.read_pickle(LBM_file_info.file[curr_idx])
        curr_LBM_vector = curr_LBM_df.groupby(['TOKEN','THRESHOLD','GUPI','TUNE_IDX','REPEAT','FOLD'],as_index=False).WINDOW_IDX.count().rename(columns={'WINDOW_IDX':'INCIDENCE'})
        curr_LBM_vector['ATTRIBUTION'] = curr_LBM_vector['INCIDENCE']/LBM_file_info.WindowTotal[curr_idx]
        output_df.append(curr_LBM_vector)
    output_df = pd.concat(output_df,ignore_index=True)
    agg_output_df = output_df.groupby(['TOKEN','THRESHOLD','TUNE_IDX'],as_index=False).ATTRIBUTION.aggregate({'SUM_ATTRIBUTION':'sum','COUNT':'count'})
    return agg_output_df

# Function to calculate Kendall's Tau values during robustness checks
def calc_tau(df):
    tau,_ = stats.kendalltau(df.OrderIdx,df.SHAP)
    return(tau)