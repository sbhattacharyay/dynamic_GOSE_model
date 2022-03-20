#### Master Script 18a: Assess model performance ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate performance metrics on resamples

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
from scipy.special import logit
from argparse import ArgumentParser
from pandas.api.types import CategoricalDtype
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, recall_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder, StandardScaler, minmax_scale
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

# StatsModel methods
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

# TQDM for progress tracking
from tqdm import tqdm

# Create directory to store model performance metrics
VERSION = 'v5-0'
perf_dir = '../model_performance/'+VERSION
os.makedirs(perf_dir,exist_ok=True)

# Establish number of resamples for bootstrapping
NUM_RESAMP = 1000

# Load cross-validation information to get GOSE and GUPIs
cv_splits = pd.read_csv('../cross_validation_splits.csv')
study_GUPI_GOSE = cv_splits[['GUPI','GOSE']].drop_duplicates()

# If bootstrapping resamples don't exist, create them
if not os.path.exists(os.path.join(perf_dir,'bs_resamples.pkl')):
    
    # Make stratified resamples for bootstrapping metrics
    bs_rs_GUPIs = [resample(study_GUPI_GOSE.GUPI.values,replace=True,n_samples=study_GUPI_GOSE.shape[0],stratify=study_GUPI_GOSE.GOSE.values) for _ in range(NUM_RESAMP)]
    bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in bs_rs_GUPIs]

    # Create Data Frame to store bootstrapping resmaples 
    bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':bs_rs_GUPIs})
    
    # Save bootstrapping resample dataframe
    bs_resamples.to_pickle(os.path.join(perf_dir,'bs_resamples.pkl'))
    
# Otherwise, load the pre-defined bootstrapping resamples
else:
    bs_resamples = pd.read_pickle(os.path.join(perf_dir,'bs_resamples.pkl'))

# Define model version directory
model_dir = '/home/sb2406/rds/hpc-work/model_outputs/'+VERSION

# Load tuning grid of current model version
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))
tuning_grid = tuning_grid[['tune_idx','WINDOW_LIMIT','STRATEGY']].drop_duplicates()
set_grid = pd.DataFrame({'set':['train','val','test'],'key':1})
tuning_grid['key'] = 1
bs_resamples['key'] = 1
rs_model_combos = pd.merge(bs_resamples,tuning_grid,how='outer',on='key').merge(set_grid,how='outer',on='key').drop(columns='key')

### II. Calculate performance metrics on resamples
# Define metric calculation function
def main(array_task_id):

    # Get resample information for current trial
    curr_gupis = rs_model_combos.GUPIs[array_task_id]
    curr_adm_or_disch = 'adm'
    curr_rs_idx = rs_model_combos.RESAMPLE_IDX[array_task_id]
    curr_tune_idx = rs_model_combos.tune_idx[array_task_id]
    curr_set = rs_model_combos.set[array_task_id]
    
    # Create directory to save current combination outputs
    metric_dir = os.path.join(perf_dir,'tune'+str(curr_tune_idx).zfill(4),'resample'+str(curr_rs_idx).zfill(4))
    os.makedirs(metric_dir,exist_ok=True)
    
    # Load compiled set predictions
    compiled_set_preds = pd.read_csv(os.path.join(model_dir,'compiled_'+curr_set+'_predictions_from_'+curr_adm_or_disch+'.csv'),index_col=0)
    
    # Filter out predictions of current tuning index
    compiled_set_preds = compiled_set_preds[compiled_set_preds.tune_idx == curr_tune_idx].reset_index(drop=True)
    
    # Define sequence of window indices for model assessment
    window_indices = list(range(1,85))
    
    # Calculate cumulative probabilities at each threshold
    prob_cols = [col for col in compiled_set_preds if col.startswith('Pr(GOSE=')]
    thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']

    for thresh in range(1,len(prob_cols)):
        cols_gt = prob_cols[thresh:]
        prob_gt = compiled_set_preds[cols_gt].sum(1).values
        gt = (compiled_set_preds['TrueLabel'] >= thresh).astype(int).values

        compiled_set_preds['Pr('+thresh_labels[thresh-1]+')'] = prob_gt
        compiled_set_preds[thresh_labels[thresh-1]] = gt
    
    # Predict highest level of functional recovery based on conservative decision rule
    prob_labels = [col for col in compiled_set_preds if col.startswith('Pr(GOSE=')]
    set_prob_matrix = compiled_set_preds[prob_labels]
    set_prob_matrix.columns = [i for i in range(set_prob_matrix.shape[1])]
    compiled_set_preds['PredLabel'] = (set_prob_matrix.cumsum(axis=1) > .5).idxmax(axis=1)
    
    ### Filter in-sample predictions for current bootstrapping resample
    curr_is_preds = compiled_set_preds[compiled_set_preds.GUPI.isin(curr_gupis)].reset_index(drop=True)
    
    ### ORC
    orcs = []
    for curr_wi in window_indices:
        filt_is_preds = curr_is_preds[curr_is_preds.WindowIdx == curr_wi].reset_index(drop=True)
        aucs = []
        for ix, (a, b) in enumerate(itertools.combinations(np.sort(filt_is_preds.TrueLabel.unique()), 2)):
            filt_rs_preds = filt_is_preds[filt_is_preds.TrueLabel.isin([a,b])].reset_index(drop=True)
            filt_rs_preds['ConditProb'] = filt_rs_preds[prob_labels[b]]/(filt_rs_preds[prob_labels[a]] + filt_rs_preds[prob_labels[b]])
            filt_rs_preds['ConditProb'] = np.nan_to_num(filt_rs_preds['ConditProb'],nan=.5,posinf=1,neginf=0)
            filt_rs_preds['ConditLabel'] = (filt_rs_preds.TrueLabel == b).astype(int)
            aucs.append(roc_auc_score(filt_rs_preds['ConditLabel'],filt_rs_preds['ConditProb']))
        curr_orc = np.mean(aucs)
        orcs.append(pd.DataFrame({'ADM_OR_DISCH':curr_adm_or_disch,'TUNE_IDX':curr_tune_idx,'RESAMPLE_IDX':curr_rs_idx,'WINDOW_IDX':curr_wi,'METRIC':'ORC','VALUE':curr_orc},index=[0]))
    orcs = pd.concat(orcs,ignore_index=True)
    
    ### Somers D
    somers_d = []
    for curr_wi in window_indices:
        filt_is_preds = curr_is_preds[curr_is_preds.WindowIdx == curr_wi].reset_index(drop=True)
        aucs = []
        prevalence = []
        for ix, (a, b) in enumerate(itertools.combinations(np.sort(filt_is_preds.TrueLabel.unique()), 2)):
            filt_rs_preds = filt_is_preds[filt_is_preds.TrueLabel.isin([a,b])].reset_index(drop=True)
            filt_rs_preds['ConditProb'] = filt_rs_preds[prob_labels[b]]/(filt_rs_preds[prob_labels[a]] + filt_rs_preds[prob_labels[b]])
            filt_rs_preds['ConditProb'] = np.nan_to_num(filt_rs_preds['ConditProb'],nan=.5,posinf=1,neginf=0)
            filt_rs_preds['ConditLabel'] = (filt_rs_preds.TrueLabel == b).astype(int)
            prevalence.append((filt_rs_preds.TrueLabel == a).sum()*(filt_rs_preds.TrueLabel == b).sum())
            aucs.append(roc_auc_score(filt_rs_preds['ConditLabel'],filt_rs_preds['ConditProb']))
        curr_somers_d = 2*(np.sum(np.multiply(aucs,prevalence))/np.sum(prevalence))-1
        somers_d.append(pd.DataFrame({'ADM_OR_DISCH':curr_adm_or_disch,'TUNE_IDX':curr_tune_idx,'RESAMPLE_IDX':curr_rs_idx,'WINDOW_IDX':curr_wi,'METRIC':'Somers D','VALUE':curr_somers_d},index=[0]))
    somers_d = pd.concat(somers_d,ignore_index=True)
    
    ### Entropy
    curr_is_preds['Entropy'] = stats.entropy(curr_is_preds[prob_cols],axis=1,base=2)
    entropy = curr_is_preds.groupby('WindowIdx',as_index=False)['Entropy'].mean().rename(columns={'Entropy':'VALUE','WindowIdx':'WINDOW_IDX'})
    entropy['ADM_OR_DISCH'] = curr_adm_or_disch
    entropy['TUNE_IDX'] = curr_tune_idx
    entropy['RESAMPLE_IDX'] = curr_rs_idx
    entropy['METRIC'] = 'Entropy'
    entropy = entropy[entropy.WINDOW_IDX.isin(window_indices)]
    
    ### Compile overall metrics into a single dataframe
    overall_metrics = pd.concat([orcs,somers_d,entropy],ignore_index=True)
    overall_metrics.to_csv(os.path.join(metric_dir,'overall_metrics_'+curr_set+'_set.csv'),index=False)
    
    ### Threshold-level AUC and ROC
    thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
    thresh_aucs = []
    thresh_rocs = []
    for curr_wi in window_indices:
        filt_is_preds = curr_is_preds[curr_is_preds.WindowIdx == curr_wi].reset_index(drop=True)
        thresh_prob_labels = [col for col in filt_is_preds if col.startswith('Pr(GOSE>')]
        thresh_aucs.append(pd.DataFrame({'ADM_OR_DISCH':curr_adm_or_disch,
                                         'TUNE_IDX':curr_tune_idx,
                                         'RESAMPLE_IDX':curr_rs_idx,
                                         'WINDOW_IDX':curr_wi,
                                         'THRESHOLD':thresh_labels,
                                         'METRIC':'AUC',
                                         'VALUE':roc_auc_score(filt_is_preds[thresh_labels],filt_is_preds[thresh_prob_labels],average=None)}))
#         for thresh in thresh_labels:
#             prob_name = 'Pr('+thresh+')'
#             (fpr, tpr, _) = roc_curve(filt_is_preds[thresh], filt_is_preds[prob_name])
#             interp_tpr = np.interp(np.linspace(0,1,200),fpr,tpr)
#             roc_axes = pd.DataFrame({'WINDOW_IDX':curr_wi,'THRESHOLD':thresh,'FPR':np.linspace(0,1,200),'TPR':interp_tpr})
#             thresh_rocs.append(roc_axes)
#     thresh_rocs = pd.concat(thresh_rocs,ignore_index = True)
#     thresh_rocs['ADM_OR_DISCH']=curr_adm_or_disch
#     thresh_rocs['TUNE_IDX']=curr_tune_idx
#     thresh_rocs['RESAMPLE_IDX']=curr_rs_idx
#     thresh_rocs.to_csv(os.path.join(metric_dir,'ROCs.csv'),index=False)
    thresh_aucs = pd.concat(thresh_aucs,ignore_index = True)
    
    ### Threshold-level calibration curves and associated metrics
    thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
    calib_metrics = []
    calib_curves = []
    for curr_wi in window_indices:
        filt_is_preds = curr_is_preds[curr_is_preds.WindowIdx == curr_wi].reset_index(drop=True)
        for thresh in thresh_labels:
            thresh_prob_name = 'Pr('+thresh+')'
            logit_gt = np.nan_to_num(logit(filt_is_preds[thresh_prob_name]),neginf=-100,posinf=100)
            calib_glm = Logit(filt_is_preds[thresh], add_constant(logit_gt))
            calib_glm_res = calib_glm.fit(disp=False)
            calib_metrics.append(pd.DataFrame({'ADM_OR_DISCH':curr_adm_or_disch,
                                               'TUNE_IDX':curr_tune_idx,
                                               'RESAMPLE_IDX':curr_rs_idx,
                                               'WINDOW_IDX':curr_wi,
                                               'THRESHOLD':thresh,
                                               'METRIC':['Calib_Slope'],
                                               'VALUE':calib_glm_res.params[1]}))
            TrueProb = lowess(endog = filt_is_preds[thresh], exog = filt_is_preds[thresh_prob_name], it = 0, xvals = np.linspace(0,1,200))
#             calib_curves.append(pd.DataFrame({'ADM_OR_DISCH':curr_adm_or_disch,
#                                               'TUNE_IDX':curr_tune_idx,
#                                               'RESAMPLE_IDX':curr_rs_idx,
#                                               'WINDOW_IDX':curr_wi,
#                                               'THRESHOLD':thresh,
#                                               'PREDPROB':np.linspace(0,1,200),
#                                               'TRUEPROB':TrueProb}))
    calib_metrics = pd.concat(calib_metrics,ignore_index = True).reset_index(drop=True)
#     calib_curves = pd.concat(calib_curves,ignore_index = True)
#     calib_curves.to_csv(os.path.join(metric_dir,'calibration_curves.csv'),index=False)
    
    #### Compile and save threshold-level metrics
    thresh_level_metrics = pd.concat([thresh_aucs,calib_metrics],ignore_index=True)
    thresh_level_metrics.to_csv(os.path.join(metric_dir,'threshold_metrics_'+curr_set+'_set.csv'),index=False)

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1]) + 20000
    main(array_task_id)