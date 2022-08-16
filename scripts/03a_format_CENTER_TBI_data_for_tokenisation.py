#### Master Script 3a: Format CENTER-TBI data for tokenisation ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Format baseline predictors in CENTER-TBI
# III. Format discharge predictors in CENTER-TBI
# IV. Format date-intervalled predictors in CENTER-TBI
# V. Format time-intervalled predictors in CENTER-TBI
# VI. Format dated single-event predictors in CENTER-TBI
# VII. Format timestamped single-event predictors in CENTER-TBI

### I. Initialisation
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
from tqdm import tqdm
from scipy import stats
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings(action="ignore")
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype

# Custom methods
from functions.token_preparation import categorizer

# Load cross-validation splits of study population
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Load CENTER-TBI ICU admission and discharge timestamps
CENTER_TBI_ICU_datetime = pd.read_csv('/home/sb2406/rds/hpc-work/timestamps/ICU_adm_disch_timestamps.csv')
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# Create directory to store formatted predictors
form_pred_dir = os.path.join('/home/sb2406/rds/hpc-work/CENTER-TBI','FormattedPredictors')
os.makedirs(form_pred_dir,exist_ok=True)

### II. Format baseline predictors in CENTER-TBI
# Load demographic, history, injury characeristic baseline predictors
demo_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
demo_info = demo_info[demo_info.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)
demo_names = list(demo_info.columns)

# Load inspection table and gather names of baseline variables
demo_baseline_variable_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/DemoInjHospMedHx/inspection_table.xlsx',sheet_name='baseline_variables',na_values = ["NA","NaN"," ", ""])
demo_baseline_names = demo_baseline_variable_desc.name[demo_baseline_variable_desc.name.isin(demo_names)].to_list()
baseline_demo_info = demo_info[['GUPI']+demo_baseline_names].dropna(subset=demo_baseline_names,how='all').reset_index(drop=True)
baseline_demo_info.columns = baseline_demo_info.columns.str.replace('_','')
demo_baseline_names = [x.replace('_', '') for x in demo_baseline_names]

## Get names of variables used to calculate new baseline variables
demo_new_variable_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/DemoInjHospMedHx/inspection_table.xlsx',sheet_name='new_variable_calculation',na_values = ["NA","NaN"," ", ""])
demo_new_baseline_names = demo_new_variable_desc.name[demo_new_variable_desc.name.isin(demo_names)].to_list()

# Convert AIS to markers of injury
AIS_new_baseline_names = [n for n in demo_new_baseline_names if 'AIS' in n]
replacement_AIS_names = [n.replace('AIS','Indicator') for n in AIS_new_baseline_names]
new_baseline_AIS_info = demo_info[['GUPI']+AIS_new_baseline_names].dropna(subset=AIS_new_baseline_names,how='all').reset_index(drop=True)
new_baseline_AIS_info[AIS_new_baseline_names] = (new_baseline_AIS_info[AIS_new_baseline_names] != 0.0).astype(int)
new_baseline_AIS_info = new_baseline_AIS_info.rename(columns=dict(zip(AIS_new_baseline_names,replacement_AIS_names)))

# Add injury markers to baseline variable dataframe
baseline_demo_info = baseline_demo_info.merge(new_baseline_AIS_info,how='left',on='GUPI')

## Calculate time from injury to ICU admission
inj_time_new_baseline_names = [n for n in demo_new_baseline_names if 'AIS' not in n]
new_baseline_inj_time_info = demo_info[['GUPI']+inj_time_new_baseline_names].dropna(subset=inj_time_new_baseline_names,how='all').reset_index(drop=True)
new_baseline_inj_time_info['InjTimestamp'] = new_baseline_inj_time_info[['DateInj', 'TimeInj']].astype(str).agg(' '.join, axis=1)
new_baseline_inj_time_info['InjTimestamp'][new_baseline_inj_time_info.DateInj.isna() | new_baseline_inj_time_info.TimeInj.isna()] = np.nan
new_baseline_inj_time_info['InjTimestamp'] = pd.to_datetime(new_baseline_inj_time_info['InjTimestamp'],format = '%Y-%m-%d %H:%M:%S')
new_baseline_inj_time_info = new_baseline_inj_time_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUAdmTimeStamp']],how='left',on='GUPI')
new_baseline_inj_time_info['InjToICUAdmHours'] = (new_baseline_inj_time_info['ICUAdmTimeStamp'] - new_baseline_inj_time_info['InjTimestamp']).astype('timedelta64[s]')/3600

# If injury-to-admission delay is "less than -10 hours," add 24 hours to delay value
new_baseline_inj_time_info.InjToICUAdmHours[(new_baseline_inj_time_info.InjToICUAdmHours <= -10)] = new_baseline_inj_time_info.InjToICUAdmHours[(new_baseline_inj_time_info.InjToICUAdmHours <= -10)]+24

# If injury-to-admission delay is still "less than 0 hours," convert value to 0
new_baseline_inj_time_info.InjToICUAdmHours[(new_baseline_inj_time_info.InjToICUAdmHours < 0)] = 0

# Add injury-to-admission delay to baseline variable dataframe 
baseline_demo_info = baseline_demo_info.merge(new_baseline_inj_time_info[['GUPI','InjToICUAdmHours']],how='left',on='GUPI')

## Split baseline variables by categorical and numeric predictors
# Categorize predictors meeting criteria
baseline_demo_info = baseline_demo_info.apply(categorizer)

# Convert ED Coagulopathy volume variables to numerics
ed_coag_volumes = [n for n in baseline_demo_info if n.startswith('EDCoagulopathyVolume')]
for edcoag in ed_coag_volumes:
    curr_edcoag = baseline_demo_info[edcoag].astype(str)
    curr_edcoag[curr_edcoag.str.isalpha()] = np.nan
    curr_edcoag = curr_edcoag.str.replace(',','.').str.replace('[^0-9\\.]','',regex=True)
    curr_edcoag[curr_edcoag == ''] = np.nan
    baseline_demo_info[edcoag] = curr_edcoag.astype(float)

# Prepend 'Baseline' to all predictor names in `baseline_demo_info` dataframe
baseline_name_cols = baseline_demo_info.columns.difference(['GUPI']).tolist()
new_baseline_names = ['Baseline'+n for n in baseline_name_cols]
baseline_demo_info = baseline_demo_info[['GUPI']+baseline_name_cols].dropna(subset=baseline_name_cols,how='all').rename(columns=dict(zip(baseline_name_cols,new_baseline_names))).reset_index(drop=True)

# Extract baseline numeric and categorical names
numeric_baseline_names = np.sort(baseline_demo_info.select_dtypes(include=['number']).columns.values)
categorical_baseline_names = np.sort(baseline_demo_info.select_dtypes(exclude=['number']).drop(columns='GUPI').columns.values)

# Remove formatting from all categorical variable string values
categorical_baseline_predictors = baseline_demo_info[np.insert(categorical_baseline_names,0,'GUPI')].dropna(axis=1,how='all').dropna(subset=categorical_baseline_names,how='all').reset_index(drop=True)
format_cols = categorical_baseline_predictors.columns.difference(['GUPI','SiteCode'])
categorical_baseline_predictors[format_cols] = categorical_baseline_predictors[format_cols].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True))
concat_cols = categorical_baseline_predictors.columns.difference(['GUPI'])
categorical_baseline_predictors[concat_cols] = categorical_baseline_predictors[concat_cols].fillna('NAN').apply(lambda x: x.name + '_' + x)

# Concatenate tokens per patient
categorical_baseline_predictors = categorical_baseline_predictors.melt(id_vars='GUPI').drop_duplicates(subset=['GUPI','value'],ignore_index=True).groupby('GUPI',as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Load and format injury description information
AIS_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/AIS/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
AIS_info = AIS_info[AIS_info.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)
AIS_names = list(AIS_info.columns)

# Load inspection table and gather names of permitted variables
AIS_variable_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/AIS/inspection_table.xlsx',sheet_name='variables',na_values = ["NA","NaN"," ", ""])
AIS_names = AIS_variable_desc.name[AIS_variable_desc.name.isin(AIS_names)].to_list()
baseline_AIS_info = AIS_info[['GUPI']+AIS_names].dropna(subset=AIS_names,how='all').reset_index(drop=True)

# Categorize coded variables
baseline_AIS_info = baseline_AIS_info.apply(categorizer,args=(100,))

# Remove formatting from strings
AIS_format_cols = baseline_AIS_info.columns.difference(['GUPI'])
baseline_AIS_info[AIS_format_cols] = baseline_AIS_info[AIS_format_cols].fillna('nan').apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True))

# For `InjDesOther,` remove all numeric values
baseline_AIS_info['InjDesOther'] = baseline_AIS_info['InjDesOther'].str.replace('[^a-zA-Z]','').str.replace(r'^\s*$','NAN',regex=True).fillna('NAN')
baseline_AIS_info[AIS_format_cols] = baseline_AIS_info[AIS_format_cols].apply(lambda x: 'Baseline' + x.name + '_' + x)

# Concatenate tokens per patient and add to categorical baseline predictors
baseline_AIS_info = baseline_AIS_info.melt(id_vars='GUPI').drop_duplicates(subset=['GUPI','value'],ignore_index=True).groupby('GUPI',as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'TokenStub'})
categorical_baseline_predictors = categorical_baseline_predictors.merge(baseline_AIS_info,how='left',on='GUPI')
categorical_baseline_predictors.TokenStub = categorical_baseline_predictors.TokenStub.fillna('BaselineInjDescription_NAN')
categorical_baseline_predictors['Token'] = categorical_baseline_predictors['Token'] + ' ' + categorical_baseline_predictors['TokenStub']
categorical_baseline_predictors = categorical_baseline_predictors.drop(columns='TokenStub')

## Load and format prior medication information
PriorMeds_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/PriorMeds/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
PriorMeds_info = PriorMeds_info[PriorMeds_info.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)

# Filter out rows with no information
PriorMeds_names = PriorMeds_info.columns.difference(['GUPI'])
PriorMeds_info = PriorMeds_info.dropna(subset=PriorMeds_names,how='all').reset_index(drop=True)

# Categorize coded variables
PriorMeds_info = PriorMeds_info.apply(categorizer,args=(100,))

# Remove formatting from strings
PriorMeds_format_cols = PriorMeds_info.columns.difference(['GUPI'])
PriorMeds_info[PriorMeds_format_cols] = PriorMeds_info[PriorMeds_format_cols].fillna('nan').apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True))

# For `PmMedicationName,` remove all numeric values and units
PriorMeds_info['PmMedicationName'] = PriorMeds_info['PmMedicationName'].str.replace('[^a-zA-Z]','').str.replace(r'^\s*$','NAN',regex=True)
PriorMeds_info['PmMedicationName'] = PriorMeds_info['PmMedicationName'].str.replace('MG','').str.replace(r'^\s*$','NAN',regex=True).fillna('NAN')
PriorMeds_info[PriorMeds_format_cols] = PriorMeds_info[PriorMeds_format_cols].apply(lambda x: 'Baseline' + x.name + '_' + x)

# Concatenate tokens per patient and add to categorical baseline predictors
PriorMeds_info = PriorMeds_info.melt(id_vars='GUPI').drop_duplicates(subset=['GUPI','value'],ignore_index=True).groupby('GUPI',as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'TokenStub'})
categorical_baseline_predictors = categorical_baseline_predictors.merge(PriorMeds_info,how='left',on='GUPI')
categorical_baseline_predictors.TokenStub = categorical_baseline_predictors.TokenStub.fillna('BaselinePriorMeds_NAN')
categorical_baseline_predictors['Token'] = categorical_baseline_predictors['Token'] + ' ' + categorical_baseline_predictors['TokenStub']
categorical_baseline_predictors = categorical_baseline_predictors.drop(columns='TokenStub')

## Load and format labs taken and processed in the ER
ER_labs = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/Labs/data.csv',na_values = ["NA","NaN"," ", "","-"])

# Filter study set patients and ER labs
ER_labs = ER_labs[(ER_labs.GUPI.isin(cv_splits.GUPI))&(ER_labs.LabsLocation=='ER')].dropna(axis=1,how='all').drop(columns=['DLDate','DLTime','LabsCompleteStatus','LabsLocation']).reset_index(drop=True)
ER_labs_names = [n for n in ER_labs if n != 'GUPI']
new_ER_labs_names = [n.replace('DL','BaselineER') for n in ER_labs_names]
ER_labs = ER_labs.dropna(subset=ER_labs_names,how='all').rename(columns=dict(zip(ER_labs_names,new_ER_labs_names))).reset_index(drop=True)

# Categorize coded variables
ER_labs = ER_labs.apply(categorizer)

# Extract ER lab numeric and categorical names
numeric_ER_lab_names = np.sort(ER_labs.select_dtypes(include=['number']).columns.values)
numeric_ER_labs = ER_labs[np.insert(numeric_ER_lab_names,0,'GUPI')].dropna(subset=numeric_ER_lab_names,how='all').reset_index(drop=True)

categorical_ER_lab_names = np.sort(ER_labs.select_dtypes(exclude=['number']).drop(columns='GUPI').columns.values)
categorical_ER_labs = ER_labs[np.insert(categorical_ER_lab_names,0,'GUPI')].dropna(subset=categorical_ER_lab_names,how='all').reset_index(drop=True).fillna('nan')
categorical_ER_labs[categorical_ER_lab_names] = categorical_ER_labs[categorical_ER_lab_names].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

# Concatenate tokens per patient and add to categorical baseline predictors
categorical_ER_labs = categorical_ER_labs.melt(id_vars='GUPI').drop_duplicates(subset=['GUPI','value'],ignore_index=True).groupby('GUPI',as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'TokenStub'})
categorical_baseline_predictors = categorical_baseline_predictors.merge(categorical_ER_labs,how='left',on='GUPI')
categorical_baseline_predictors.TokenStub = categorical_baseline_predictors.TokenStub.fillna('BaselineERLabsCategoricals_NAN')
categorical_baseline_predictors['Token'] = categorical_baseline_predictors['Token'] + ' ' + categorical_baseline_predictors['TokenStub']
categorical_baseline_predictors = categorical_baseline_predictors.drop(columns='TokenStub')

## Load and format imaging data taken and processed in the ER
ER_imaging = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/Imaging/data.csv',na_values = ["NA","NaN"," ", "","-"])

# Filter study set patients and ER images
ER_imaging = ER_imaging[(ER_imaging.GUPI.isin(cv_splits.GUPI))&((ER_imaging.CTPatientLocation=='ED')|(ER_imaging.MRIPatientLocation=='ED'))].drop(columns=['ExperimentDate','ExperimentTime','AcquisitionDate','AcquisitionTime']).dropna(axis=1,how='all').reset_index(drop=True)

# Calculate new variable to indicate lesion after CT
imaging_new_variable_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/Imaging/inspection_table.xlsx',sheet_name='new_variable_calculation',na_values = ["NA","NaN"," ", ""])
new_var_imaging_name = imaging_new_variable_desc.name[0]
ER_imaging['CTLesionDetected'] = np.nan
ER_imaging.CTLesionDetected[ER_imaging[new_var_imaging_name] == 0] = 0
ER_imaging.CTLesionDetected[(ER_imaging[new_var_imaging_name] != 0)&(ER_imaging[new_var_imaging_name] != 88)&(~ER_imaging[new_var_imaging_name].isna())] = 1
ER_imaging.CTLesionDetected[ER_imaging[new_var_imaging_name] == 88] = 2

# Load inspection table and gather names of permitted variables
ER_imaging_names = [n for n in ER_imaging if n not in ['GUPI','CRFForm']]
imaging_variable_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/Imaging/inspection_table.xlsx',sheet_name='variables',na_values = ["NA","NaN"," ", ""])
imaging_names = imaging_variable_desc.name[imaging_variable_desc.name.isin(ER_imaging_names)].to_list()+['CTLesionDetected']
new_imaging_names = ['BaselineER'+n for n in imaging_names]
ER_imaging = ER_imaging[['GUPI']+imaging_names].dropna(subset=imaging_names,how='all').rename(columns=dict(zip(imaging_names,new_imaging_names))).reset_index(drop=True)
ER_imaging.columns = ER_imaging.columns.str.replace('_','')
new_imaging_names = [x.replace('_', '') for x in new_imaging_names]

# First, separate out CT and MRI dataframes
ER_CT_imaging = ER_imaging[ER_imaging.BaselineERXsiType == 'xnat:ctSessionData'].dropna(axis=1,how='all').reset_index(drop=True)
ER_MR_imaging = ER_imaging[ER_imaging.BaselineERXsiType == 'xnat:mrSessionData'].dropna(axis=1,how='all').reset_index(drop=True)

# Split CT dataframe into categorical and numeric
ER_CT_imaging = ER_CT_imaging.apply(categorizer)

numeric_ER_CT_imaging_names = np.sort(ER_CT_imaging.select_dtypes(include=['number']).columns.values)
numeric_ER_CT_imaging = ER_CT_imaging[np.insert(numeric_ER_CT_imaging_names,0,'GUPI')].dropna(subset=numeric_ER_CT_imaging_names,how='all').reset_index(drop=True)

categorical_ER_CT_imaging_names = np.sort(ER_CT_imaging.select_dtypes(exclude=['number']).drop(columns='GUPI').columns.values)
categorical_ER_CT_imaging = ER_CT_imaging[np.insert(categorical_ER_CT_imaging_names,0,'GUPI')].dropna(subset=categorical_ER_CT_imaging_names,how='all').reset_index(drop=True).fillna('nan')
categorical_ER_CT_imaging[categorical_ER_CT_imaging_names] = categorical_ER_CT_imaging[categorical_ER_CT_imaging_names].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

# Concatenate tokens per patient and add to categorical baseline predictors
categorical_ER_CT_imaging = categorical_ER_CT_imaging.melt(id_vars='GUPI').drop_duplicates(subset=['GUPI','value'],ignore_index=True).groupby('GUPI',as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'TokenStub'})
categorical_baseline_predictors = categorical_baseline_predictors.merge(categorical_ER_CT_imaging,how='left',on='GUPI')
categorical_baseline_predictors.TokenStub = categorical_baseline_predictors.TokenStub.fillna('BaselineERCTImagingCategoricals_NAN')
categorical_baseline_predictors['Token'] = categorical_baseline_predictors['Token'] + ' ' + categorical_baseline_predictors['TokenStub']
categorical_baseline_predictors = categorical_baseline_predictors.drop(columns='TokenStub')

# Do the same for MRI dataframe (no numerical values due to small sample)
ER_MR_imaging = ER_MR_imaging.apply(categorizer)

ER_MR_imaging_names = np.sort(ER_MR_imaging.select_dtypes(exclude=['number']).drop(columns='GUPI').columns.values)
ER_MR_imaging = ER_MR_imaging[np.insert(ER_MR_imaging_names,0,'GUPI')].dropna(subset=ER_MR_imaging_names,how='all').reset_index(drop=True).fillna('nan')
ER_MR_imaging[ER_MR_imaging_names] = ER_MR_imaging[ER_MR_imaging_names].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

ER_MR_imaging = ER_MR_imaging.melt(id_vars='GUPI').drop_duplicates(subset=['GUPI','value'],ignore_index=True).groupby('GUPI',as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'TokenStub'})
categorical_baseline_predictors = categorical_baseline_predictors.merge(ER_MR_imaging,how='left',on='GUPI')
categorical_baseline_predictors.TokenStub = categorical_baseline_predictors.TokenStub.fillna('BaselineERMRImaging_NAN')
categorical_baseline_predictors['Token'] = categorical_baseline_predictors['Token'] + ' ' + categorical_baseline_predictors['TokenStub']
categorical_baseline_predictors = categorical_baseline_predictors.drop(columns='TokenStub')

## Compile and save baseline predictors
# Categorical predictors - ensure unique tokens per GUPI and save
for curr_GUPI in tqdm(categorical_baseline_predictors.GUPI.unique(), 'Cleaning baseline categorical predictors'):
    curr_token_set = categorical_baseline_predictors.Token[categorical_baseline_predictors.GUPI == curr_GUPI].values[0]
    cleaned_token_set = ' '.join(np.sort(np.unique(curr_token_set.split())))
    categorical_baseline_predictors.Token[categorical_baseline_predictors.GUPI == curr_GUPI] = cleaned_token_set
    
categorical_baseline_predictors = categorical_baseline_predictors.sort_values(by='GUPI').reset_index(drop=True)
categorical_baseline_predictors.to_pickle(os.path.join(form_pred_dir,'categorical_baseline_predictors.pkl'))

# Numeric predictors - pivot dataframe to long form, and merge different dataframes before saving
numeric_baseline_predictors = baseline_demo_info[np.insert(numeric_baseline_names,0,'GUPI')].dropna(axis=1,how='all').dropna(subset=numeric_baseline_names,how='all').reset_index(drop=True)
numeric_baseline_predictors = pd.concat([numeric_baseline_predictors.melt(id_vars='GUPI',var_name='VARIABLE',value_name='VALUE'),
                                         numeric_ER_labs.melt(id_vars='GUPI',var_name='VARIABLE',value_name='VALUE'),
                                         numeric_ER_CT_imaging.melt(id_vars='GUPI',var_name='VARIABLE',value_name='VALUE')],ignore_index=True).sort_values(by=['GUPI','VARIABLE','VALUE'])

numeric_baseline_predictors.reset_index(drop=True).to_pickle(os.path.join(form_pred_dir,'numeric_baseline_predictors.pkl'))

### III. Format discharge predictors in CENTER-TBI
# Load demographic, history, injury characeristic discharge predictors
demo_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
demo_info = demo_info[demo_info.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)
demo_names = list(demo_info.columns)

# Load inspection table and gather names of discharge variables
demo_discharge_variable_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/DemoInjHospMedHx/inspection_table.xlsx',sheet_name='discharge_variables',na_values = ["NA","NaN"," ", ""])
demo_discharge_names = demo_discharge_variable_desc.name[demo_discharge_variable_desc.name.isin(demo_names)].to_list()
discharge_demo_info = demo_info[['GUPI']+demo_discharge_names].dropna(subset=demo_discharge_names,how='all').reset_index(drop=True)

# Prepend 'Discharge' to all predictor names in `discharge_demo_info` dataframe
new_discharge_names = ['Discharge'+n for n in demo_discharge_names]
discharge_demo_info = discharge_demo_info[['GUPI']+demo_discharge_names].dropna(subset=demo_discharge_names,how='all').rename(columns=dict(zip(demo_discharge_names,new_discharge_names))).reset_index(drop=True)

## Split discharge variables by categorical and numeric predictors
# Categorize predictors meeting criteria
discharge_demo_info = discharge_demo_info.apply(categorizer)

# Extract discharge numeric and categorical names
numeric_discharge_names = np.sort(discharge_demo_info.select_dtypes(include=['number']).columns.values)
categorical_discharge_names = np.sort(discharge_demo_info.select_dtypes(exclude=['number']).drop(columns='GUPI').columns.values)

# Remove formatting from all categorical variable string values
categorical_discharge_predictors = discharge_demo_info[np.insert(categorical_discharge_names,0,'GUPI')].dropna(axis=1,how='all').dropna(subset=categorical_discharge_names,how='all').reset_index(drop=True)
format_cols = categorical_discharge_predictors.columns.difference(['GUPI'])
categorical_discharge_predictors[format_cols] = categorical_discharge_predictors[format_cols].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

# Concatenate tokens per patient
categorical_discharge_predictors = categorical_discharge_predictors.melt(id_vars='GUPI').drop_duplicates(subset=['GUPI','value'],ignore_index=True).groupby('GUPI',as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Save discharge predictors
# Categorical
categorical_discharge_predictors = categorical_discharge_predictors.sort_values(by='GUPI').reset_index(drop=True)
categorical_discharge_predictors.to_pickle(os.path.join(form_pred_dir,'categorical_discharge_predictors.pkl'))

# Numeric predictors - pivot dataframe to long form, and merge different dataframes before saving
numeric_discharge_predictors = discharge_demo_info[np.insert(numeric_discharge_names,0,'GUPI')].dropna(axis=1,how='all').dropna(subset=numeric_discharge_names,how='all').reset_index(drop=True)
numeric_discharge_predictors = numeric_discharge_predictors.melt(id_vars='GUPI',var_name='VARIABLE',value_name='VALUE').sort_values(by=['GUPI','VARIABLE','VALUE'])
numeric_discharge_predictors.to_pickle(os.path.join(form_pred_dir,'numeric_discharge_predictors.pkl'))

### IV. Format date-intervalled predictors in CENTER-TBI
# Load demographic, history, injury characeristic interval predictors
demo_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
demo_info = demo_info[demo_info.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)
demo_names = list(demo_info.columns)

# Load inspection table and gather names of date-intervalled variables
demo_interval_variable_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/DemoInjHospMedHx/inspection_table.xlsx',sheet_name='interval_variables',na_values = ["NA","NaN"," ", ""])

# Filter out timestamped interval variables
demo_interval_variable_desc = demo_interval_variable_desc.groupby('IntervalType',as_index=False).filter(lambda g: ~g['name'].str.contains('Time').any()).reset_index(drop=True)
demo_interval_names = demo_interval_variable_desc.name[demo_interval_variable_desc.name.isin(demo_names)].to_list()
interval_demo_info = demo_info[['GUPI']+demo_interval_names].dropna(subset=demo_interval_names,how='all').reset_index(drop=True)

# Add ICU discharge date to interval demo info dataframe
interval_demo_info = interval_demo_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischDate']],how='left',on='GUPI')

## DVTProphylaxisMech
# Get columns names pertaining to DVTProphylaxisMech
curr_names = demo_interval_variable_desc[demo_interval_variable_desc.IntervalType=='DVTProphylaxisMech'].name.values

# Extract current interval variable information
curr_interval_demo_info = interval_demo_info[np.insert(curr_names,0,['GUPI','ICUDischDate'])].dropna(subset=['DVTProphylaxisMechStartDate','DVTProphylaxisMechStopDate'],how='all').reset_index(drop=True)

# Replace missing stop dates with discharge date
curr_interval_demo_info.DVTProphylaxisMechStopDate[curr_interval_demo_info.DVTProphylaxisMechStopDate.isna()] = curr_interval_demo_info.ICUDischDate[curr_interval_demo_info.DVTProphylaxisMechStopDate.isna()]

# Filter out instance with start date after discharge date
curr_interval_demo_info[['DVTProphylaxisMechStartDate','DVTProphylaxisMechStopDate','ICUDischDate']] = curr_interval_demo_info[['DVTProphylaxisMechStartDate','DVTProphylaxisMechStopDate','ICUDischDate']].apply(lambda x: pd.to_datetime(x,format = '%Y-%m-%d' ))
curr_interval_demo_info = curr_interval_demo_info[curr_interval_demo_info.DVTProphylaxisMechStartDate <= curr_interval_demo_info.ICUDischDate].reset_index(drop=True)

# If stop date extends beyond ICU discharge date, fix '...Ongoing' value to 1
curr_interval_demo_info.DVTMechOngoing[curr_interval_demo_info.DVTProphylaxisMechStopDate > curr_interval_demo_info.ICUDischDate] = 1
curr_interval_demo_info = curr_interval_demo_info.apply(categorizer)

# Tokenise categorical variables
curr_interval_demo_info[['DVTMechOngoing','DVTProphylaxisMech','DVTProphylaxisMechType']] = curr_interval_demo_info[['DVTMechOngoing','DVTProphylaxisMech','DVTProphylaxisMechType']].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
DVTProphylaxisMech_interval_info = curr_interval_demo_info[['GUPI','DVTProphylaxisMechStartDate','DVTProphylaxisMechStopDate','DVTProphylaxisMech','DVTMechOngoing','DVTProphylaxisMechType']].rename(columns={'DVTProphylaxisMechStartDate':'StartDate','DVTProphylaxisMechStopDate':'StopDate'}).melt(id_vars=['GUPI','StartDate','StopDate']).groupby(['GUPI','StartDate','StopDate'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## DVTProphylaxisPharm
# Get columns names pertaining to DVTProphylaxisPharm
curr_names = demo_interval_variable_desc[demo_interval_variable_desc.IntervalType=='DVTProphylaxisPharm'].name.values

# Extract current interval variable information
curr_interval_demo_info = interval_demo_info[np.insert(curr_names,0,['GUPI','ICUDischDate'])].dropna(subset=['DVTProphylaxisStartDate','DVTProphylaxisStopDate'],how='all').reset_index(drop=True)

# Replace missing stop dates with discharge date
curr_interval_demo_info.DVTProphylaxisStopDate[curr_interval_demo_info.DVTProphylaxisStopDate.isna()] = curr_interval_demo_info.ICUDischDate[curr_interval_demo_info.DVTProphylaxisStopDate.isna()]

# Filter out instance with start date after discharge date
curr_interval_demo_info[['DVTProphylaxisStartDate','DVTProphylaxisStopDate','ICUDischDate']] = curr_interval_demo_info[['DVTProphylaxisStartDate','DVTProphylaxisStopDate','ICUDischDate']].apply(lambda x: pd.to_datetime(x,format = '%Y-%m-%d'))
curr_interval_demo_info = curr_interval_demo_info[curr_interval_demo_info.DVTProphylaxisStartDate <= curr_interval_demo_info.ICUDischDate].reset_index(drop=True)

# If stop date extends beyond ICU discharge date, fix '...Ongoing' value to 1
curr_interval_demo_info.DVTPharmOngoing[curr_interval_demo_info.DVTProphylaxisStopDate > curr_interval_demo_info.ICUDischDate] = 1
curr_interval_demo_info = curr_interval_demo_info.apply(categorizer)

# Tokenise categorical variables
curr_interval_demo_info[['DVTPharmOngoing','DVTProphylaxisPharm','DVTPharmType']] = curr_interval_demo_info[['DVTPharmOngoing','DVTProphylaxisPharm','DVTPharmType']].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
DVTProphylaxisPharm_interval_info = curr_interval_demo_info[['GUPI','DVTProphylaxisStartDate','DVTProphylaxisStopDate','DVTProphylaxisPharm','DVTPharmOngoing','DVTPharmType']].rename(columns={'DVTProphylaxisStartDate':'StartDate','DVTProphylaxisStopDate':'StopDate'}).melt(id_vars=['GUPI','StartDate','StopDate']).groupby(['GUPI','StartDate','StopDate'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## EnteralNutrition
# Get columns names pertaining to EnteralNutrition
curr_names = demo_interval_variable_desc[demo_interval_variable_desc.IntervalType=='EnteralNutrition'].name.values

# Extract current interval variable information
curr_interval_demo_info = interval_demo_info[np.insert(curr_names,0,['GUPI','ICUDischDate'])].dropna(subset=['EnteralNutritionStartDate','EnteralNutritionStopDate'],how='all').reset_index(drop=True)

# Replace missing stop dates with discharge date
curr_interval_demo_info.EnteralNutritionStopDate[curr_interval_demo_info.EnteralNutritionStopDate.isna()] = curr_interval_demo_info.ICUDischDate[curr_interval_demo_info.EnteralNutritionStopDate.isna()]

# Filter out instance with start date after discharge date
curr_interval_demo_info[['EnteralNutritionStartDate','EnteralNutritionStopDate','ICUDischDate']] = curr_interval_demo_info[['EnteralNutritionStartDate','EnteralNutritionStopDate','ICUDischDate']].apply(lambda x: pd.to_datetime(x,format = '%Y-%m-%d'))
curr_interval_demo_info = curr_interval_demo_info[curr_interval_demo_info.EnteralNutritionStartDate <= curr_interval_demo_info.ICUDischDate].apply(categorizer).reset_index(drop=True)

# Tokenise categorical variables
curr_interval_demo_info[['EnteralNutrition','EnteralNutritionRoute']] = curr_interval_demo_info[['EnteralNutrition','EnteralNutritionRoute']].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
EnteralNutrition_interval_info = curr_interval_demo_info[['GUPI','EnteralNutritionStartDate','EnteralNutritionStopDate','EnteralNutrition','EnteralNutritionRoute']].rename(columns={'EnteralNutritionStartDate':'StartDate','EnteralNutritionStopDate':'StopDate'}).melt(id_vars=['GUPI','StartDate','StopDate']).groupby(['GUPI','StartDate','StopDate'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Nasogastric
# Get columns names pertaining to Nasogastric
curr_names = demo_interval_variable_desc[demo_interval_variable_desc.IntervalType=='Nasogastric'].name.values

# Extract current interval variable information
curr_interval_demo_info = interval_demo_info[np.insert(curr_names,0,['GUPI','ICUDischDate'])].dropna(subset=['NasogastricStartDate','NasogastricStopDate'],how='all').reset_index(drop=True)

# Replace missing stop dates with discharge date
curr_interval_demo_info.NasogastricStopDate[curr_interval_demo_info.NasogastricStopDate.isna()] = curr_interval_demo_info.ICUDischDate[curr_interval_demo_info.NasogastricStopDate.isna()]

# Filter out instance with start date after discharge date
curr_interval_demo_info[['NasogastricStartDate','NasogastricStopDate','ICUDischDate']] = curr_interval_demo_info[['NasogastricStartDate','NasogastricStopDate','ICUDischDate']].apply(lambda x: pd.to_datetime(x,format = '%Y-%m-%d'))
curr_interval_demo_info = curr_interval_demo_info[curr_interval_demo_info.NasogastricStartDate <= curr_interval_demo_info.ICUDischDate].reset_index(drop=True)

# If stop date extends beyond ICU discharge date, fix '...Ongoing' value to 1
curr_interval_demo_info.NasogastricOngoing[curr_interval_demo_info.NasogastricStopDate > curr_interval_demo_info.ICUDischDate] = 1
curr_interval_demo_info = curr_interval_demo_info.apply(categorizer)

# Tokenise categorical variables
curr_interval_demo_info[['Nasogastric','NasogastricOngoing']] = curr_interval_demo_info[['Nasogastric','NasogastricOngoing']].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
Nasogastric_interval_info = curr_interval_demo_info[['GUPI','NasogastricStartDate','NasogastricStopDate','Nasogastric','NasogastricOngoing']].rename(columns={'NasogastricStartDate':'StartDate','NasogastricStopDate':'StopDate'}).melt(id_vars=['GUPI','StartDate','StopDate']).groupby(['GUPI','StartDate','StopDate'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## OxygenAdm
# Get columns names pertaining to OxygenAdm
curr_names = demo_interval_variable_desc[demo_interval_variable_desc.IntervalType=='OxygenAdm'].name.values

# Extract current interval variable information
curr_interval_demo_info = interval_demo_info[np.insert(curr_names,0,['GUPI','ICUDischDate'])].dropna(subset=['OxygenAdmStartDate','OxygenAdmStopDate'],how='all').reset_index(drop=True)

# Replace missing stop dates with discharge date
curr_interval_demo_info.OxygenAdmStopDate[curr_interval_demo_info.OxygenAdmStopDate.isna()] = curr_interval_demo_info.ICUDischDate[curr_interval_demo_info.OxygenAdmStopDate.isna()]

# Filter out instance with start date after discharge date
curr_interval_demo_info[['OxygenAdmStartDate','OxygenAdmStopDate','ICUDischDate']] = curr_interval_demo_info[['OxygenAdmStartDate','OxygenAdmStopDate','ICUDischDate']].apply(lambda x: pd.to_datetime(x,format = '%Y-%m-%d'))
curr_interval_demo_info = curr_interval_demo_info[curr_interval_demo_info.OxygenAdmStartDate <= curr_interval_demo_info.ICUDischDate].reset_index(drop=True)

# If stop date extends beyond ICU discharge date, fix '...Ongoing' value to 1
curr_interval_demo_info.OxygenAdmOngoing[curr_interval_demo_info.OxygenAdmStopDate > curr_interval_demo_info.ICUDischDate] = 1
curr_interval_demo_info = curr_interval_demo_info.apply(categorizer)

# Tokenise categorical variables
curr_interval_demo_info[['OxygenAdm','OxygenAdmOngoing']] = curr_interval_demo_info[['OxygenAdm','OxygenAdmOngoing']].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
OxygenAdm_interval_info = curr_interval_demo_info[['GUPI','OxygenAdmStartDate','OxygenAdmStopDate','OxygenAdm','OxygenAdmOngoing']].rename(columns={'OxygenAdmStartDate':'StartDate','OxygenAdmStopDate':'StopDate'}).melt(id_vars=['GUPI','StartDate','StopDate']).groupby(['GUPI','StartDate','StopDate'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## ParenteralNutrition
# Get columns names pertaining to ParenteralNutrition
curr_names = demo_interval_variable_desc[demo_interval_variable_desc.IntervalType=='ParenteralNutrition'].name.values

# Extract current interval variable information
curr_interval_demo_info = interval_demo_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUAdmDate']],how='left',on='GUPI')[np.insert(curr_names,0,['GUPI','ICUAdmDate','ICUDischDate'])].dropna(subset=['ParenteralNutritionStartDate','ParenteralNutritionStopDate'],how='all').reset_index(drop=True)

# Replace missing stop dates with discharge date and missing start dates with admission date
curr_interval_demo_info.ParenteralNutritionStopDate[curr_interval_demo_info.ParenteralNutritionStopDate.isna()] = curr_interval_demo_info.ICUDischDate[curr_interval_demo_info.ParenteralNutritionStopDate.isna()]
curr_interval_demo_info.ParenteralNutritionStartDate[curr_interval_demo_info.ParenteralNutritionStartDate.isna()] = curr_interval_demo_info.ICUAdmDate[curr_interval_demo_info.ParenteralNutritionStartDate.isna()]

# Filter out instance with start date after discharge date
curr_interval_demo_info[['ParenteralNutritionStartDate','ParenteralNutritionStopDate','ICUDischDate']] = curr_interval_demo_info[['ParenteralNutritionStartDate','ParenteralNutritionStopDate','ICUDischDate']].apply(lambda x: pd.to_datetime(x,format = '%Y-%m-%d'))
curr_interval_demo_info = curr_interval_demo_info[curr_interval_demo_info.ParenteralNutritionStartDate <= curr_interval_demo_info.ICUDischDate].apply(categorizer).reset_index(drop=True)

# Tokenise categorical variables
curr_interval_demo_info[['ParenteralNutrition']] = curr_interval_demo_info[['ParenteralNutrition']].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
ParenteralNutrition_interval_info = curr_interval_demo_info[['GUPI','ParenteralNutritionStartDate','ParenteralNutritionStopDate','ParenteralNutrition']].rename(columns={'ParenteralNutritionStartDate':'StartDate','ParenteralNutritionStopDate':'StopDate'}).melt(id_vars=['GUPI','StartDate','StopDate']).groupby(['GUPI','StartDate','StopDate'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## PEGTube
# Get columns names pertaining to PEGTube
curr_names = demo_interval_variable_desc[demo_interval_variable_desc.IntervalType=='PEGTube'].name.values

# Extract current interval variable information
curr_interval_demo_info = interval_demo_info[np.insert(curr_names,0,['GUPI','ICUDischDate'])].dropna(subset=['PEGTubeStartDate','PEGTubeStopDate'],how='all').reset_index(drop=True)

# Replace missing stop dates with discharge date
curr_interval_demo_info.PEGTubeStopDate[curr_interval_demo_info.PEGTubeStopDate.isna()] = curr_interval_demo_info.ICUDischDate[curr_interval_demo_info.PEGTubeStopDate.isna()]

# Filter out instance with start date after discharge date
curr_interval_demo_info[['PEGTubeStartDate','PEGTubeStopDate','ICUDischDate']] = curr_interval_demo_info[['PEGTubeStartDate','PEGTubeStopDate','ICUDischDate']].apply(lambda x: pd.to_datetime(x,format = '%Y-%m-%d'))
curr_interval_demo_info = curr_interval_demo_info[curr_interval_demo_info.PEGTubeStartDate <= curr_interval_demo_info.ICUDischDate].reset_index(drop=True)

# If stop date extends beyond ICU discharge date, fix '...Ongoing' value to 1
curr_interval_demo_info.PEGTubeOngoing[curr_interval_demo_info.PEGTubeStopDate > curr_interval_demo_info.ICUDischDate] = 1
curr_interval_demo_info = curr_interval_demo_info.apply(categorizer)

# Tokenise categorical variables
curr_interval_demo_info[['PEGTube','PEGTubeOngoing']] = curr_interval_demo_info[['PEGTube','PEGTubeOngoing']].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
PEGTube_interval_info = curr_interval_demo_info[['GUPI','PEGTubeStartDate','PEGTubeStopDate','PEGTube','PEGTubeOngoing']].rename(columns={'PEGTubeStartDate':'StartDate','PEGTubeStopDate':'StopDate'}).melt(id_vars=['GUPI','StartDate','StopDate']).groupby(['GUPI','StartDate','StopDate'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Tracheostomy
# Get columns names pertaining to Tracheostomy
curr_names = demo_interval_variable_desc[demo_interval_variable_desc.IntervalType=='Tracheostomy'].name.values

# Extract current interval variable information
curr_interval_demo_info = interval_demo_info[np.insert(curr_names,0,['GUPI','ICUDischDate'])].dropna(subset=['TracheostomyStartDate','TracheostomyStopDate'],how='all').reset_index(drop=True)

# Replace missing stop dates with discharge date
curr_interval_demo_info.TracheostomyStopDate[curr_interval_demo_info.TracheostomyStopDate.isna()] = curr_interval_demo_info.ICUDischDate[curr_interval_demo_info.TracheostomyStopDate.isna()]

# Filter out instance with start date after discharge date
curr_interval_demo_info[['TracheostomyStartDate','TracheostomyStopDate','ICUDischDate']] = curr_interval_demo_info[['TracheostomyStartDate','TracheostomyStopDate','ICUDischDate']].apply(lambda x: pd.to_datetime(x,format = '%Y-%m-%d'))
curr_interval_demo_info = curr_interval_demo_info[curr_interval_demo_info.TracheostomyStartDate <= curr_interval_demo_info.ICUDischDate].reset_index(drop=True)

# If stop date extends beyond ICU discharge date, fix '...Ongoing' value to 1
curr_interval_demo_info.TracheostomyOngoing[curr_interval_demo_info.TracheostomyStopDate > curr_interval_demo_info.ICUDischDate] = 1
curr_interval_demo_info = curr_interval_demo_info.apply(categorizer)

# Tokenise categorical variables
curr_interval_demo_info[['Tracheostomy','TracheostomyOngoing']] = curr_interval_demo_info[['Tracheostomy','TracheostomyOngoing']].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
Tracheostomy_interval_info = curr_interval_demo_info[['GUPI','TracheostomyStartDate','TracheostomyStopDate','Tracheostomy','TracheostomyOngoing']].rename(columns={'TracheostomyStartDate':'StartDate','TracheostomyStopDate':'StopDate'}).melt(id_vars=['GUPI','StartDate','StopDate']).groupby(['GUPI','StartDate','StopDate'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## UrineCath
# Get columns names pertaining to UrineCath
curr_names = demo_interval_variable_desc[demo_interval_variable_desc.IntervalType=='UrineCath'].name.values

# Extract current interval variable information
curr_interval_demo_info = interval_demo_info[np.insert(curr_names,0,['GUPI','ICUDischDate'])].dropna(subset=['UrineCathStartDate','UrineCathStopDate'],how='all').reset_index(drop=True)

# Replace missing stop dates with discharge date
curr_interval_demo_info.UrineCathStopDate[curr_interval_demo_info.UrineCathStopDate.isna()] = curr_interval_demo_info.ICUDischDate[curr_interval_demo_info.UrineCathStopDate.isna()]

# Filter out instance with start date after discharge date
curr_interval_demo_info[['UrineCathStartDate','UrineCathStopDate','ICUDischDate']] = curr_interval_demo_info[['UrineCathStartDate','UrineCathStopDate','ICUDischDate']].apply(lambda x: pd.to_datetime(x,format = '%Y-%m-%d'))
curr_interval_demo_info = curr_interval_demo_info[curr_interval_demo_info.UrineCathStartDate <= curr_interval_demo_info.ICUDischDate].reset_index(drop=True)

# If stop date extends beyond ICU discharge date, fix '...Ongoing' value to 1
curr_interval_demo_info.UrineCathOngoing[curr_interval_demo_info.UrineCathStopDate > curr_interval_demo_info.ICUDischDate] = 1
curr_interval_demo_info = curr_interval_demo_info.apply(categorizer)

# Tokenise categorical variables
curr_interval_demo_info[['UrineCath','UrineCathOngoing']] = curr_interval_demo_info[['UrineCath','UrineCathOngoing']].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
UrineCath_interval_info = curr_interval_demo_info[['GUPI','UrineCathStartDate','UrineCathStopDate','UrineCath','UrineCathOngoing']].rename(columns={'UrineCathStartDate':'StartDate','UrineCathStopDate':'StopDate'}).melt(id_vars=['GUPI','StartDate','StopDate']).groupby(['GUPI','StartDate','StopDate'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Catheter Related Bloodstream Infection (CRBSI)
# Extract current interval variable information
curr_interval_demo_info = demo_info[['GUPI','ComplCRBSIDateDiagnosis','ICUDisComplCRBSI']].dropna(subset=['ComplCRBSIDateDiagnosis','ICUDisComplCRBSI'],how='all').reset_index(drop=True)
curr_interval_demo_info = curr_interval_demo_info[curr_interval_demo_info.ICUDisComplCRBSI != 0].reset_index(drop=True)

# Compare against date of discharge
curr_interval_demo_info = curr_interval_demo_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischDate']],how='left',on='GUPI')
curr_interval_demo_info[['ComplCRBSIDateDiagnosis','ICUDischDate']] = curr_interval_demo_info[['ComplCRBSIDateDiagnosis','ICUDischDate']].apply(lambda x: pd.to_datetime(x,format = '%Y-%m-%d'))
curr_interval_demo_info = curr_interval_demo_info[curr_interval_demo_info.ComplCRBSIDateDiagnosis <= curr_interval_demo_info.ICUDischDate].apply(categorizer).reset_index(drop=True)

# Tokenise categorical variables
curr_interval_demo_info[['ICUDisComplCRBSI']] = curr_interval_demo_info[['ICUDisComplCRBSI']].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
CRBSI_interval_info = curr_interval_demo_info[['GUPI','ComplCRBSIDateDiagnosis','ICUDischDate','ICUDisComplCRBSI']].rename(columns={'ComplCRBSIDateDiagnosis':'StartDate','ICUDischDate':'StopDate'}).melt(id_vars=['GUPI','StartDate','StopDate']).groupby(['GUPI','StartDate','StopDate'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Pneumonia Antibiotic Treatment
# Extract current interval variable information
curr_interval_demo_info = demo_info[['GUPI','ICUDisPneumAntibiotic1StartDate','ICUDisPneumAntibiotic2StartDate','ICUDisPneumAntibiotic3StartDate','ICUDisPneumAntibiotic4StartDate','ICUDisPneumAntibioticTreat']].dropna(subset=['ICUDisPneumAntibiotic1StartDate','ICUDisPneumAntibiotic2StartDate','ICUDisPneumAntibiotic3StartDate','ICUDisPneumAntibiotic4StartDate','ICUDisPneumAntibioticTreat'],how='all').reset_index(drop=True)
curr_interval_demo_info = curr_interval_demo_info[curr_interval_demo_info.ICUDisPneumAntibioticTreat != 0].merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischDate']],how='left',on='GUPI').reset_index(drop=True)

# If treatment date is missing, replace with discharge date
curr_interval_demo_info.ICUDisPneumAntibiotic1StartDate[(curr_interval_demo_info.ICUDisPneumAntibioticTreat == 1)*(curr_interval_demo_info.ICUDisPneumAntibiotic1StartDate.isna())] = curr_interval_demo_info.ICUDischDate[(curr_interval_demo_info.ICUDisPneumAntibioticTreat == 1)*(curr_interval_demo_info.ICUDisPneumAntibiotic1StartDate.isna())]

# Convert to long form and merge discharge date inforation
curr_interval_demo_info = curr_interval_demo_info.melt(id_vars=['GUPI','ICUDischDate','ICUDisPneumAntibioticTreat'])
curr_interval_demo_info[['value','ICUDischDate']] = curr_interval_demo_info[['value','ICUDischDate']].apply(lambda x: pd.to_datetime(x,format = '%Y-%m-%d'))

# Only permit values with ICU stay date
curr_interval_demo_info = curr_interval_demo_info[curr_interval_demo_info.value <= curr_interval_demo_info.ICUDischDate].reset_index(drop=True)

# Change variable name to designate treatment number
curr_interval_demo_info['variable'] = curr_interval_demo_info['variable'].str.replace('ICUDisPneumAntibiotic','').str.replace('StartDate','').astype(int)
curr_interval_demo_info = curr_interval_demo_info.apply(categorizer).rename(columns={'variable':'ICUDisPneumAntibioticTreatmentNo','value':'StartDate','ICUDischDate':'StopDate'})

# Tokenize variables and reorder columns
curr_interval_demo_info[['ICUDisPneumAntibioticTreat','ICUDisPneumAntibioticTreatmentNo']] = curr_interval_demo_info[['ICUDisPneumAntibioticTreat','ICUDisPneumAntibioticTreatmentNo']].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
PneumAntibiotic_interval_info = curr_interval_demo_info.melt(id_vars=['GUPI','StartDate','StopDate']).groupby(['GUPI','StartDate','StopDate'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Ventilator-associated Pneumonia
# Extract current interval variable information
curr_interval_demo_info = demo_info[['GUPI','ICUDisPneumDate','ICUDisNososcomialPneumNum','ICUDisPneumSepsis','ICUDisPneumClinical','ICUDisPneumChestX','ICUDisPneumBacteriaSmpl','ICUDisPneumPathogen1','ICUDisPneumPathogen1QuantUCFml','ICUDisPneumPathogen2','ICUDisPneumPathogen2QuantUCFml','ICUDisPneumPathogen3','ICUDisPneumPathogen3QuantUCFml','ICUDisPneumPathogen4','ICUDisPneumPathogen4QuantUCFml']].dropna(subset=['ICUDisNososcomialPneumNum','ICUDisPneumSepsis','ICUDisPneumClinical','ICUDisPneumChestX','ICUDisPneumBacteriaSmpl','ICUDisPneumPathogen1','ICUDisPneumPathogen1QuantUCFml','ICUDisPneumPathogen2','ICUDisPneumPathogen2QuantUCFml','ICUDisPneumPathogen3','ICUDisPneumPathogen3QuantUCFml','ICUDisPneumPathogen4','ICUDisPneumPathogen4QuantUCFml'],how='all').reset_index(drop=True).merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischDate']],how='left',on='GUPI').reset_index(drop=True)

# Only permit values with ICU stay date
curr_interval_demo_info[['ICUDisPneumDate','ICUDischDate']] = curr_interval_demo_info[['ICUDisPneumDate','ICUDischDate']].apply(lambda x: pd.to_datetime(x,format = '%Y-%m-%d'))
curr_interval_demo_info = curr_interval_demo_info[curr_interval_demo_info.ICUDisPneumDate <= curr_interval_demo_info.ICUDischDate].reset_index(drop=True).apply(categorizer).rename(columns={'ICUDisPneumDate':'StartDate','ICUDischDate':'StopDate'})

# Tokenize variables and reorder columns
curr_interval_demo_info[['ICUDisNososcomialPneumNum','ICUDisPneumSepsis','ICUDisPneumClinical','ICUDisPneumChestX','ICUDisPneumBacteriaSmpl','ICUDisPneumPathogen1','ICUDisPneumPathogen1QuantUCFml','ICUDisPneumPathogen2','ICUDisPneumPathogen2QuantUCFml','ICUDisPneumPathogen3','ICUDisPneumPathogen3QuantUCFml','ICUDisPneumPathogen4','ICUDisPneumPathogen4QuantUCFml']] = curr_interval_demo_info[['ICUDisNososcomialPneumNum','ICUDisPneumSepsis','ICUDisPneumClinical','ICUDisPneumChestX','ICUDisPneumBacteriaSmpl','ICUDisPneumPathogen1','ICUDisPneumPathogen1QuantUCFml','ICUDisPneumPathogen2','ICUDisPneumPathogen2QuantUCFml','ICUDisPneumPathogen3','ICUDisPneumPathogen3QuantUCFml','ICUDisPneumPathogen4','ICUDisPneumPathogen4QuantUCFml']].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
VAP_interval_info = curr_interval_demo_info.melt(id_vars=['GUPI','StartDate','StopDate']).groupby(['GUPI','StartDate','StopDate'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## ICU medication information
# Load medication administration data
meds_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/Medication/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
meds_info = meds_info[meds_info.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)
meds_names = list(meds_info.columns.difference(['GUPI']))
new_meds_names = ['Medication'+n for n in meds_names]

# Remove entries with missing information, rename columns, and merge ICU admission + discharge timestamps
meds_info = meds_info.dropna(subset=meds_names,how='all').reset_index(drop=True).rename(columns=dict(zip(meds_names,new_meds_names))).merge(CENTER_TBI_ICU_datetime[['GUPI','ICUAdmDate','ICUDischDate']],how='left',on='GUPI')

# Format date variables correctly to datetime
meds_info[['MedicationStartDate','MedicationStopDate','ICUAdmDate','ICUDischDate']] = meds_info[['MedicationStartDate','MedicationStopDate','ICUAdmDate','ICUDischDate']].apply(lambda x: pd.to_datetime(x,format = '%Y-%m-%d'))

# Impute missing start dates with ICU admission date and missing stop dates with discharge date
meds_info.MedicationStartDate[meds_info.MedicationStartDate.isna()] = meds_info.ICUAdmDate[meds_info.MedicationStartDate.isna()]
meds_info.MedicationStopDate[meds_info.MedicationStopDate.isna()] = meds_info.ICUDischDate[meds_info.MedicationStopDate.isna()]

# If medication stop date extends beyond ICU discharge date, fix 'MedicationOngoing' value to 1
meds_info.MedicationOngoing[meds_info.MedicationStopDate > meds_info.ICUDischDate] = 1

# Filter out medications administered after ICU discharge, apply the categorizer, and remove ICU adm/disch timestamps
meds_info = meds_info[meds_info.MedicationStartDate <= meds_info.ICUDischDate].reset_index(drop=True).apply(categorizer,args=(100,)).drop(columns=['ICUAdmDate','ICUDischDate']).rename(columns={'MedicationStartDate':'StartDate','MedicationStopDate':'StopDate'})

# Tokenize variables and reorder columns
meds_info[['MedicationClass','MedicationAgent','MedicationRoute','MedicationReason','MedicationReasonOther','MedicationHighestDailyDose','MedicationAgentOther','MedicationOngoing']] = meds_info[['MedicationClass','MedicationAgent','MedicationRoute','MedicationReason','MedicationReasonOther','MedicationHighestDailyDose','MedicationAgentOther','MedicationOngoing']].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
meds_info = meds_info.melt(id_vars=['GUPI','StartDate','StopDate']).drop_duplicates(subset=['GUPI','StartDate','StopDate','value']).reset_index(drop=True).groupby(['GUPI','StartDate','StopDate'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Concatenate date-intervalled predictors
# Concatenate
date_interval_predictors = pd.concat([DVTProphylaxisMech_interval_info,DVTProphylaxisPharm_interval_info,EnteralNutrition_interval_info,Nasogastric_interval_info,OxygenAdm_interval_info,ParenteralNutrition_interval_info,PEGTube_interval_info,Tracheostomy_interval_info,UrineCath_interval_info,CRBSI_interval_info,PneumAntibiotic_interval_info,VAP_interval_info,meds_info],ignore_index=True)

# Group by GUPI, StartDate, StopDate, and merge tokens
date_interval_predictors = date_interval_predictors.groupby(['GUPI','StartDate','StopDate'],as_index=False).Token.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'}).reset_index(drop=True)

# Iterate through entries, ensure unique tokens, and extract EndTokens
date_interval_predictors['EndToken'] = np.nan
for curr_idx in tqdm(range(date_interval_predictors.shape[0]), 'Cleaning date-intervalled predictors'):
    curr_token_set = date_interval_predictors.Token[curr_idx]
    cleaned_token_list = np.sort(np.unique(curr_token_set.split()))
    end_token_list = [t for t in cleaned_token_list if 'Ongoing_' in t]
    if len(end_token_list) > 0:
        cleaned_token_list = [t for t in cleaned_token_list if t not in end_token_list]
        cleaned_token_set = ' '.join(cleaned_token_list)
        end_token_set = ' '.join(end_token_list)
    else:
        cleaned_token_set = ' '.join(cleaned_token_list)
        end_token_set = np.nan
    date_interval_predictors.Token[curr_idx] = cleaned_token_set
    date_interval_predictors.EndToken[curr_idx] = end_token_set

# Ad-hoc correction of patients with incorrectly labelled start dates
date_interval_predictors.StopDate[(date_interval_predictors.StartDate>date_interval_predictors.StopDate)&(date_interval_predictors.GUPI=='3zJm265')] = date_interval_predictors.StopDate[(date_interval_predictors.StartDate>date_interval_predictors.StopDate)&(date_interval_predictors.GUPI=='3zJm265')] + pd.DateOffset(months=1)

date_interval_predictors.StopDate[(date_interval_predictors.StartDate>date_interval_predictors.StopDate)&(date_interval_predictors.GUPI=='6XWQ346')] = date_interval_predictors.StopDate[(date_interval_predictors.StartDate>date_interval_predictors.StopDate)&(date_interval_predictors.GUPI=='6XWQ346')] + pd.DateOffset(months=1)

date_interval_predictors.StartDate[(date_interval_predictors.StartDate>date_interval_predictors.StopDate)&(date_interval_predictors.GUPI=='6aVa533')] = date_interval_predictors.StartDate[(date_interval_predictors.StartDate>date_interval_predictors.StopDate)&(date_interval_predictors.GUPI=='6aVa533')] - pd.DateOffset(months=1)

# FIlter out all datapoints with start date after ICU discharge
date_interval_predictors = date_interval_predictors.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
date_interval_predictors = date_interval_predictors[date_interval_predictors.StartDate <= date_interval_predictors.ICUDischTimeStamp].drop(columns='ICUDischTimeStamp').reset_index(drop=True)

# Sort values and save date-intervalled predictors
date_interval_predictors = date_interval_predictors.sort_values(by=['GUPI','StartDate','StopDate']).reset_index(drop=True)
date_interval_predictors.to_pickle(os.path.join(form_pred_dir,'categorical_date_interval_predictors.pkl'))

### V. Format time-intervalled predictors in CENTER-TBI
# Load demographic, history, injury characeristic interval predictors
demo_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
demo_info = demo_info[demo_info.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)
demo_names = list(demo_info.columns)

# Load inspection table and gather names of date-intervalled variables
demo_interval_variable_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/DemoInjHospMedHx/inspection_table.xlsx',sheet_name='interval_variables',na_values = ["NA","NaN"," ", ""])

# Filter interval variables
demo_interval_names = demo_interval_variable_desc.name[demo_interval_variable_desc.name.isin(demo_names)].to_list()
interval_demo_info = demo_info[['GUPI']+demo_interval_names].dropna(subset=demo_interval_names,how='all').reset_index(drop=True)

## ICP monitoring information
# Get columns names pertaining to ICP monitoring
curr_names = demo_interval_variable_desc[demo_interval_variable_desc.IntervalType=='ICP'].name.values

# Separate catheter revision variables
revis_names = ['ICUCatheterICP','ICUCatheterICPDate','ICUCatheterICPTime']
curr_names = [n for n in curr_names if n not in revis_names]

# Extract current interval variable information
curr_interval_demo_info = interval_demo_info[np.insert(curr_names,0,['GUPI'])].dropna(subset=curr_names,how='all').reset_index(drop=True)
revis_interval_demo_info = interval_demo_info[np.insert(revis_names,0,['GUPI'])].dropna(subset=revis_names,how='all').reset_index(drop=True)

# Remove all non-revised ICP information
revis_interval_demo_info = revis_interval_demo_info[(revis_interval_demo_info.ICUCatheterICP != 0)|(~revis_interval_demo_info.ICUCatheterICPDate.isna())].reset_index(drop=True)

# Calculate median ICP insertion time, median removal time, and median catheter revision time
median_ICPInsTime = pd.to_datetime(curr_interval_demo_info.ICPInsTime,format = '%H:%M:%S').dropna().median().strftime('%H:%M:%S')
median_ICPRemTime = pd.to_datetime(curr_interval_demo_info.ICPRemTime,format = '%H:%M:%S').dropna().median().strftime('%H:%M:%S')
median_ICUCatheterICPTime = pd.to_datetime(revis_interval_demo_info.ICUCatheterICPTime,format = '%H:%M:%S').dropna().median().strftime('%H:%M:%S')

# If ICP insertion time is missing, fill in with admission time (if on the same day), or fill in with median insertion time
curr_interval_demo_info = curr_interval_demo_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUAdmDate','ICUAdmTime']],how='left',on='GUPI')
curr_interval_demo_info.ICPInsTime[(curr_interval_demo_info.ICPInsTime.isna())&(curr_interval_demo_info.ICPInsDate == curr_interval_demo_info.ICUAdmDate)] = curr_interval_demo_info.ICUAdmTime[(curr_interval_demo_info.ICPInsTime.isna())&(curr_interval_demo_info.ICPInsDate == curr_interval_demo_info.ICUAdmDate)]
curr_interval_demo_info.ICPInsTime[curr_interval_demo_info.ICPInsTime.isna()] = median_ICPInsTime
curr_interval_demo_info = curr_interval_demo_info.drop(columns=['ICUAdmDate','ICUAdmTime'])

# Create ICP insertation timestamp values
curr_interval_demo_info['ICPInsTimestamp'] = curr_interval_demo_info[['ICPInsDate', 'ICPInsTime']].astype(str).agg(' '.join, axis=1)
curr_interval_demo_info['ICPInsTimestamp'][curr_interval_demo_info.ICPInsDate.isna() | curr_interval_demo_info.ICPInsTime.isna()] = np.nan
curr_interval_demo_info['ICPInsTimestamp'] = pd.to_datetime(curr_interval_demo_info['ICPInsTimestamp'],format = '%Y-%m-%d %H:%M:%S' )
curr_interval_demo_info = curr_interval_demo_info.drop(columns=['ICPInsDate','ICPInsTime'])

# If ICP removal time is missing, fill in with discharge time (if on the same day), or fill in with median removal time
curr_interval_demo_info = curr_interval_demo_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischDate','ICUDischTime']],how='left',on='GUPI')
curr_interval_demo_info.ICPRemTime[(curr_interval_demo_info.ICPRemTime.isna())&(curr_interval_demo_info.ICPRemDate == curr_interval_demo_info.ICUDischDate)] = curr_interval_demo_info.ICUDischTime[(curr_interval_demo_info.ICPRemTime.isna())&(curr_interval_demo_info.ICPRemDate == curr_interval_demo_info.ICUDischDate)]
curr_interval_demo_info.ICPRemTime[curr_interval_demo_info.ICPRemTime.isna()] = median_ICPRemTime
curr_interval_demo_info = curr_interval_demo_info.drop(columns=['ICUDischDate','ICUDischTime'])

# Create ICP removal timestamp values
curr_interval_demo_info['ICPRemTimestamp'] = curr_interval_demo_info[['ICPRemDate', 'ICPRemTime']].astype(str).agg(' '.join, axis=1)
curr_interval_demo_info['ICPRemTimestamp'][curr_interval_demo_info.ICPRemDate.isna() | curr_interval_demo_info.ICPRemTime.isna()] = np.nan
curr_interval_demo_info['ICPRemTimestamp'] = pd.to_datetime(curr_interval_demo_info['ICPRemTimestamp'],format = '%Y-%m-%d %H:%M:%S' )
curr_interval_demo_info = curr_interval_demo_info.drop(columns=['ICPRemDate','ICPRemTime'])

# Fill in remaining missing ICP removal timestamps with ICU discharge timestamp
curr_interval_demo_info = curr_interval_demo_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
curr_interval_demo_info.ICPRemTimestamp[curr_interval_demo_info.ICPRemTimestamp.isna()] = curr_interval_demo_info.ICUDischTimeStamp[curr_interval_demo_info.ICPRemTimestamp.isna()] 
curr_interval_demo_info = curr_interval_demo_info.drop(columns=['ICUDischTimeStamp']).apply(categorizer)

# If ICP renewal time is missing, fill in with median removal time, and create timestamp for renewal
revis_interval_demo_info.ICUCatheterICPTime[revis_interval_demo_info.ICUCatheterICPTime.isna()] = median_ICUCatheterICPTime
revis_interval_demo_info['ICUCatheterICPTimestamp'] = revis_interval_demo_info[['ICUCatheterICPDate', 'ICUCatheterICPTime']].astype(str).agg(' '.join, axis=1)
revis_interval_demo_info['ICUCatheterICPTimestamp'][revis_interval_demo_info.ICUCatheterICPDate.isna() | revis_interval_demo_info.ICUCatheterICPTime.isna()] = np.nan
revis_interval_demo_info['ICUCatheterICPTimestamp'] = pd.to_datetime(revis_interval_demo_info['ICUCatheterICPTimestamp'],format = '%Y-%m-%d %H:%M:%S' )
revis_interval_demo_info = revis_interval_demo_info.drop(columns=['ICUCatheterICPDate','ICUCatheterICPTime']).merge(curr_interval_demo_info[['GUPI','ICPRemTimestamp']],how='left',on='GUPI').apply(categorizer)

# Tokenise variables in both ICP dataframes appropriately
format_cols = curr_interval_demo_info.select_dtypes(exclude='number').columns.difference(['GUPI','ICPInsTimestamp','ICPRemTimestamp'])
curr_interval_demo_info[format_cols] = curr_interval_demo_info[format_cols].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

format_cols = revis_interval_demo_info.select_dtypes(exclude='number').columns.difference(['GUPI','ICUCatheterICPTimestamp','ICPRemTimestamp'])
revis_interval_demo_info[format_cols] = revis_interval_demo_info[format_cols].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

# Separate numeric end variable of ICP monitoring and save for timestamped events dataframe
numeric_icp_mont_duration = curr_interval_demo_info[['GUPI','ICPRemTimestamp','ICPMontDuration']].rename(columns={'ICPRemTimestamp':'TimeStamp'}).melt(id_vars=['GUPI','TimeStamp'],var_name='VARIABLE',value_name='VALUE')
curr_interval_demo_info = curr_interval_demo_info.drop(columns='ICPMontDuration')

# Rename timestamp columns
curr_interval_demo_info = curr_interval_demo_info.rename(columns={'ICPInsTimestamp':'StartTimeStamp','ICPRemTimestamp':'EndTimeStamp'})
revis_interval_demo_info = revis_interval_demo_info.rename(columns={'ICUCatheterICPTimestamp':'StartTimeStamp','ICPRemTimestamp':'EndTimeStamp'})

# Separate other ICP end variables and tokenise
ICP_end_variables = ['ICPMonitorStop','ICUProblemsICP','ICUProblemsICPYes','ICURaisedICP']
curr_end_variable_info = curr_interval_demo_info[['GUPI','StartTimeStamp','EndTimeStamp']+ICP_end_variables]
curr_end_variable_info = curr_end_variable_info.melt(id_vars=['GUPI','StartTimeStamp','EndTimeStamp']).drop_duplicates(subset=['GUPI','StartTimeStamp','EndTimeStamp','value']).groupby(['GUPI','StartTimeStamp','EndTimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'EndToken'})

# Tokenise proper interval variables
icp_interval_info = curr_interval_demo_info[curr_interval_demo_info.columns.difference(ICP_end_variables)].melt(id_vars=['GUPI','StartTimeStamp','EndTimeStamp']).drop_duplicates(subset=['GUPI','StartTimeStamp','EndTimeStamp','value']).groupby(['GUPI','StartTimeStamp','EndTimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'}).merge(curr_end_variable_info,how='left',on=['GUPI','StartTimeStamp','EndTimeStamp'])

revis_interval_demo_info = revis_interval_demo_info.melt(id_vars=['GUPI','StartTimeStamp','EndTimeStamp']).drop_duplicates(subset=['GUPI','StartTimeStamp','EndTimeStamp','value']).groupby(['GUPI','StartTimeStamp','EndTimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})
revis_interval_demo_info['EndToken'] = np.nan

icp_interval_info = pd.concat([icp_interval_info,revis_interval_demo_info],ignore_index=True).sort_values(by=['GUPI','StartTimeStamp','EndTimeStamp']).reset_index(drop=True)

# Ad-hoc correction of patient with incorrectly labelled start timestamp
icp_interval_info.StartTimeStamp[icp_interval_info.StartTimeStamp > icp_interval_info.EndTimeStamp] = icp_interval_info.StartTimeStamp[icp_interval_info.StartTimeStamp > icp_interval_info.EndTimeStamp].iloc[0].replace(month=1,day=2)

## Intubation information
# Get columns names pertaining to Intubation
curr_names = demo_interval_variable_desc[demo_interval_variable_desc.IntervalType=='Intubation'].name.values

# Extract current interval variable information
curr_interval_demo_info = interval_demo_info[np.insert(curr_names,0,['GUPI'])].dropna(subset=curr_names,how='all').reset_index(drop=True)

# Remove all non-intubated patients
curr_interval_demo_info = curr_interval_demo_info[(curr_interval_demo_info.Intubation != 0)|(~curr_interval_demo_info.IntubationStartDate.isna())].reset_index(drop=True)

# Calculate median intubation start and stop times
median_IntubationStartTime = pd.to_datetime(curr_interval_demo_info.IntubationStartTime,format = '%H:%M:%S').dropna().median().strftime('%H:%M:%S')
median_IntubationStopTime = pd.to_datetime(curr_interval_demo_info.IntubationStopTime,format = '%H:%M:%S').dropna().median().strftime('%H:%M:%S')

# If intubation start time is missing, fill in with admission time (if on the same day), or fill in with median insertion time
curr_interval_demo_info = curr_interval_demo_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUAdmDate','ICUAdmTime']],how='left',on='GUPI')
curr_interval_demo_info.IntubationStartTime[(curr_interval_demo_info.IntubationStartTime.isna())&(curr_interval_demo_info.IntubationStartDate == curr_interval_demo_info.ICUAdmDate)] = curr_interval_demo_info.ICUAdmTime[(curr_interval_demo_info.IntubationStartTime.isna())&(curr_interval_demo_info.IntubationStartDate == curr_interval_demo_info.ICUAdmDate)]
curr_interval_demo_info.IntubationStartTime[curr_interval_demo_info.IntubationStartTime.isna()] = median_IntubationStartTime
curr_interval_demo_info = curr_interval_demo_info.drop(columns=['ICUAdmDate','ICUAdmTime'])

# Create intubation timestamp values
curr_interval_demo_info['IntubationStartTimestamp'] = curr_interval_demo_info[['IntubationStartDate', 'IntubationStartTime']].astype(str).agg(' '.join, axis=1)
curr_interval_demo_info['IntubationStartTimestamp'][curr_interval_demo_info.IntubationStartDate.isna() | curr_interval_demo_info.IntubationStartTime.isna()] = np.nan
curr_interval_demo_info['IntubationStartTimestamp'] = pd.to_datetime(curr_interval_demo_info['IntubationStartTimestamp'],format = '%Y-%m-%d %H:%M:%S' )
curr_interval_demo_info = curr_interval_demo_info.drop(columns=['IntubationStartDate','IntubationStartTime'])

# If intubation stop is missing, fill in with discharge time (if on the same day), or fill in with median removal time
curr_interval_demo_info = curr_interval_demo_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischDate','ICUDischTime']],how='left',on='GUPI')
curr_interval_demo_info.IntubationStopTime[(curr_interval_demo_info.IntubationStopTime.isna())&(curr_interval_demo_info.IntubationStopDate == curr_interval_demo_info.ICUDischDate)] = curr_interval_demo_info.ICUDischTime[(curr_interval_demo_info.IntubationStopTime.isna())&(curr_interval_demo_info.IntubationStopDate == curr_interval_demo_info.ICUDischDate)]
curr_interval_demo_info.IntubationStopTime[curr_interval_demo_info.IntubationStopTime.isna()] = median_IntubationStopTime
curr_interval_demo_info = curr_interval_demo_info.drop(columns=['ICUDischDate','ICUDischTime'])

# Create intubation stop timestamp values
curr_interval_demo_info['IntubationStopTimestamp'] = curr_interval_demo_info[['IntubationStopDate', 'IntubationStopTime']].astype(str).agg(' '.join, axis=1)
curr_interval_demo_info['IntubationStopTimestamp'][curr_interval_demo_info.IntubationStopDate.isna() | curr_interval_demo_info.IntubationStopTime.isna()] = np.nan
curr_interval_demo_info['IntubationStopTimestamp'] = pd.to_datetime(curr_interval_demo_info['IntubationStopTimestamp'],format = '%Y-%m-%d %H:%M:%S' )
curr_interval_demo_info = curr_interval_demo_info.drop(columns=['IntubationStopDate','IntubationStopTime'])

# Fill in remaining missing intubation stop timestamp with ICU discharge timestamp
curr_interval_demo_info = curr_interval_demo_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
curr_interval_demo_info.IntubationStopTimestamp[curr_interval_demo_info.IntubationStopTimestamp.isna()] = curr_interval_demo_info.ICUDischTimeStamp[curr_interval_demo_info.IntubationStopTimestamp.isna()] 
curr_interval_demo_info = curr_interval_demo_info.drop(columns=['ICUDischTimeStamp']).apply(categorizer)

# Tokenise variables appropriately
curr_interval_demo_info[['Intubation','IntubationStop']] = curr_interval_demo_info[['Intubation','IntubationStop']].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

# Rename timestamp columns
curr_interval_demo_info = curr_interval_demo_info.rename(columns={'IntubationStartTimestamp':'StartTimeStamp','IntubationStopTimestamp':'EndTimeStamp'})

# Separate intubation end variables and tokenise
intubation_end_variables = ['IntubationStop']
curr_end_variable_info = curr_interval_demo_info[['GUPI','StartTimeStamp','EndTimeStamp']+intubation_end_variables]
curr_end_variable_info = curr_end_variable_info.melt(id_vars=['GUPI','StartTimeStamp','EndTimeStamp']).drop_duplicates(subset=['GUPI','StartTimeStamp','EndTimeStamp','value']).groupby(['GUPI','StartTimeStamp','EndTimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'EndToken'})

# Tokenise proper interval variables
intubation_interval_info = curr_interval_demo_info[curr_interval_demo_info.columns.difference(intubation_end_variables)].melt(id_vars=['GUPI','StartTimeStamp','EndTimeStamp']).drop_duplicates(subset=['GUPI','StartTimeStamp','EndTimeStamp','value']).groupby(['GUPI','StartTimeStamp','EndTimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'}).merge(curr_end_variable_info,how='left',on=['GUPI','StartTimeStamp','EndTimeStamp'])

# Ad-hoc correction of patient with incorrectly labelled start timestamp
switched_start_timestamps = intubation_interval_info.EndTimeStamp[(intubation_interval_info.StartTimeStamp > intubation_interval_info.EndTimeStamp)&intubation_interval_info.GUPI.isin(['3QPA537','8LQu798'])]
switched_end_timestamps = intubation_interval_info.StartTimeStamp[(intubation_interval_info.StartTimeStamp > intubation_interval_info.EndTimeStamp)&intubation_interval_info.GUPI.isin(['3QPA537','8LQu798'])]
intubation_interval_info.EndTimeStamp[(intubation_interval_info.StartTimeStamp > intubation_interval_info.EndTimeStamp)&intubation_interval_info.GUPI.isin(['3QPA537','8LQu798'])] = switched_end_timestamps
intubation_interval_info.StartTimeStamp[(intubation_interval_info.StartTimeStamp > intubation_interval_info.EndTimeStamp)&intubation_interval_info.GUPI.isin(['3QPA537','8LQu798'])] = switched_start_timestamps
intubation_interval_info.StartTimeStamp[(intubation_interval_info.StartTimeStamp > intubation_interval_info.EndTimeStamp)&(intubation_interval_info.GUPI=='9VYk836')] = intubation_interval_info.StartTimeStamp[(intubation_interval_info.StartTimeStamp > intubation_interval_info.EndTimeStamp)&(intubation_interval_info.GUPI=='9VYk836')].iloc[0].replace(month=1,day=2)

## Mechanical ventilation information
# Get columns names pertaining to Mechanical ventilation
curr_names = demo_interval_variable_desc[demo_interval_variable_desc.IntervalType=='MechVentilation'].name.values

# Extract current interval variable information
curr_interval_demo_info = interval_demo_info[np.insert(curr_names,0,['GUPI'])].dropna(subset=curr_names,how='all').reset_index(drop=True)

# Remove all non-ventilated patients
curr_interval_demo_info = curr_interval_demo_info[(curr_interval_demo_info.MechVentilation != 0)|(~curr_interval_demo_info.MechVentilationStartDate.isna())].reset_index(drop=True)

# Calculate median MechVentilation start and stop times
median_MechVentilationStartTime = pd.to_datetime(curr_interval_demo_info.MechVentilationStartTime,format = '%H:%M:%S').dropna().median().strftime('%H:%M:%S')
median_MechVentilationStopTime = pd.to_datetime(curr_interval_demo_info.MechVentilationStopTime,format = '%H:%M:%S').dropna().median().strftime('%H:%M:%S')

# If MechVentilation start time is missing, fill in with admission time (if on the same day), or fill in with median insertion time
curr_interval_demo_info = curr_interval_demo_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUAdmDate','ICUAdmTime']],how='left',on='GUPI')
curr_interval_demo_info.MechVentilationStartTime[(curr_interval_demo_info.MechVentilationStartTime.isna())&(curr_interval_demo_info.MechVentilationStartDate == curr_interval_demo_info.ICUAdmDate)] = curr_interval_demo_info.ICUAdmTime[(curr_interval_demo_info.MechVentilationStartTime.isna())&(curr_interval_demo_info.MechVentilationStartDate == curr_interval_demo_info.ICUAdmDate)]
curr_interval_demo_info.MechVentilationStartTime[curr_interval_demo_info.MechVentilationStartTime.isna()] = median_MechVentilationStartTime
curr_interval_demo_info = curr_interval_demo_info.drop(columns=['ICUAdmDate','ICUAdmTime'])

# Create MechVentilation timestamp values
curr_interval_demo_info['MechVentilationStartTimestamp'] = curr_interval_demo_info[['MechVentilationStartDate', 'MechVentilationStartTime']].astype(str).agg(' '.join, axis=1)
curr_interval_demo_info['MechVentilationStartTimestamp'][curr_interval_demo_info.MechVentilationStartDate.isna() | curr_interval_demo_info.MechVentilationStartTime.isna()] = np.nan
curr_interval_demo_info['MechVentilationStartTimestamp'] = pd.to_datetime(curr_interval_demo_info['MechVentilationStartTimestamp'],format = '%Y-%m-%d %H:%M:%S' )
curr_interval_demo_info = curr_interval_demo_info.drop(columns=['MechVentilationStartDate','MechVentilationStartTime'])

# Fill in remaining missing MechVentilation start timestamp with ICU admission timestamp
curr_interval_demo_info = curr_interval_demo_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUAdmTimeStamp']],how='left',on='GUPI')
curr_interval_demo_info.MechVentilationStartTimestamp[curr_interval_demo_info.MechVentilationStartTimestamp.isna()] = curr_interval_demo_info.ICUAdmTimeStamp[curr_interval_demo_info.MechVentilationStartTimestamp.isna()] 
curr_interval_demo_info = curr_interval_demo_info.drop(columns=['ICUAdmTimeStamp']).apply(categorizer)

# If MechVentilation stop is missing, fill in with discharge time (if on the same day), or fill in with median removal time
curr_interval_demo_info = curr_interval_demo_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischDate','ICUDischTime']],how='left',on='GUPI')
curr_interval_demo_info.MechVentilationStopTime[(curr_interval_demo_info.MechVentilationStopTime.isna())&(curr_interval_demo_info.MechVentilationStopDate == curr_interval_demo_info.ICUDischDate)] = curr_interval_demo_info.ICUDischTime[(curr_interval_demo_info.MechVentilationStopTime.isna())&(curr_interval_demo_info.MechVentilationStopDate == curr_interval_demo_info.ICUDischDate)]
curr_interval_demo_info.MechVentilationStopTime[curr_interval_demo_info.MechVentilationStopTime.isna()] = median_MechVentilationStopTime
curr_interval_demo_info = curr_interval_demo_info.drop(columns=['ICUDischDate','ICUDischTime'])

# Create MechVentilation stop timestamp values
curr_interval_demo_info['MechVentilationStopTimestamp'] = curr_interval_demo_info[['MechVentilationStopDate', 'MechVentilationStopTime']].astype(str).agg(' '.join, axis=1)
curr_interval_demo_info['MechVentilationStopTimestamp'][curr_interval_demo_info.MechVentilationStopDate.isna() | curr_interval_demo_info.MechVentilationStopTime.isna()] = np.nan
curr_interval_demo_info['MechVentilationStopTimestamp'] = pd.to_datetime(curr_interval_demo_info['MechVentilationStopTimestamp'],format = '%Y-%m-%d %H:%M:%S' )
curr_interval_demo_info = curr_interval_demo_info.drop(columns=['MechVentilationStopDate','MechVentilationStopTime'])

# Fill in remaining missing MechVentilation stop timestamp with ICU discharge timestamp
curr_interval_demo_info = curr_interval_demo_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
curr_interval_demo_info.MechVentilationStopTimestamp[curr_interval_demo_info.MechVentilationStopTimestamp.isna()] = curr_interval_demo_info.ICUDischTimeStamp[curr_interval_demo_info.MechVentilationStopTimestamp.isna()] 
curr_interval_demo_info = curr_interval_demo_info.drop(columns=['ICUDischTimeStamp']).apply(categorizer)

# Tokenise variables appropriately
curr_interval_demo_info['MechVentilation'] = 'MechVentilation_'+curr_interval_demo_info['MechVentilation'].str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True).fillna('NAN')

# Rename timestamp columns
curr_interval_demo_info = curr_interval_demo_info.rename(columns={'MechVentilationStartTimestamp':'StartTimeStamp','MechVentilationStopTimestamp':'EndTimeStamp'})

# Tokenise proper interval variables
mech_ventilation_interval_info = curr_interval_demo_info.melt(id_vars=['GUPI','StartTimeStamp','EndTimeStamp']).drop_duplicates(subset=['GUPI','StartTimeStamp','EndTimeStamp','value']).groupby(['GUPI','StartTimeStamp','EndTimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

# Ad-hoc correction of patient with incorrectly labelled start timestamp
switched_start_timestamps = mech_ventilation_interval_info.EndTimeStamp[(mech_ventilation_interval_info.StartTimeStamp > mech_ventilation_interval_info.EndTimeStamp)&mech_ventilation_interval_info.GUPI.isin(['4Xwp396', '8EFA895', '8LQu798', '9Eac349', '9LHt796'])]
switched_end_timestamps = mech_ventilation_interval_info.StartTimeStamp[(mech_ventilation_interval_info.StartTimeStamp > mech_ventilation_interval_info.EndTimeStamp)&mech_ventilation_interval_info.GUPI.isin(['4Xwp396', '8EFA895', '8LQu798', '9Eac349', '9LHt796'])]

mech_ventilation_interval_info.EndTimeStamp[(mech_ventilation_interval_info.StartTimeStamp > mech_ventilation_interval_info.EndTimeStamp)&mech_ventilation_interval_info.GUPI.isin(['4Xwp396', '8EFA895', '8LQu798', '9Eac349', '9LHt796'])] = switched_end_timestamps
mech_ventilation_interval_info.StartTimeStamp[(mech_ventilation_interval_info.StartTimeStamp > mech_ventilation_interval_info.EndTimeStamp)&mech_ventilation_interval_info.GUPI.isin(['4Xwp396', '8EFA895', '8LQu798', '9Eac349', '9LHt796'])] = switched_start_timestamps

# Add empty EndToken column
mech_ventilation_interval_info['EndToken'] = np.nan

## Intracranial surgery information
# Load intracranial surgery data
cran_surgery_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/SurgeriesCranial/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
cran_surgery_info = cran_surgery_info[cran_surgery_info.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)

# Load cranial surgery inspection table to extract allowed predictors
cran_surgery_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/SurgeriesCranial/inspection_table.xlsx',sheet_name='variables',na_values = ["NA","NaN"," ", ""])
cran_surgery_names = cran_surgery_desc.name[cran_surgery_desc.name.isin(cran_surgery_info.columns)].to_list()
cran_surgery_info = cran_surgery_info[['GUPI']+cran_surgery_names].dropna(subset=cran_surgery_names,how='all').reset_index(drop=True)

# Calculate median surgery start time and median surgery duration
median_SurgeryStartTime = pd.to_datetime(cran_surgery_info.SurgeryStartTime,format = '%H:%M:%S').dropna().median().strftime('%H:%M:%S')
median_SurgeryDurationSecs = (pd.to_datetime(cran_surgery_info.SurgeryEndDate+' '+cran_surgery_info.SurgeryEndTime,format = '%Y-%m-%d %H:%M:%S') - pd.to_datetime(cran_surgery_info.SurgeryStartDate+' '+cran_surgery_info.SurgeryStartTime,format = '%Y-%m-%d %H:%M:%S')).astype('timedelta64[s]').dropna().median()

# If cranial surgery start time is missing, fill in with admission time (if on the same day), or fill in with median insertion time
cran_surgery_info = cran_surgery_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUAdmDate','ICUAdmTime']],how='left',on='GUPI')
cran_surgery_info.SurgeryStartTime[(cran_surgery_info.SurgeryStartTime.isna())&(cran_surgery_info.SurgeryStartDate == cran_surgery_info.ICUAdmDate)] = cran_surgery_info.ICUAdmTime[(cran_surgery_info.SurgeryStartTime.isna())&(cran_surgery_info.SurgeryStartDate == cran_surgery_info.ICUAdmDate)]
cran_surgery_info.SurgeryStartTime[cran_surgery_info.SurgeryStartTime.isna()] = median_SurgeryStartTime
cran_surgery_info = cran_surgery_info.drop(columns=['ICUAdmDate','ICUAdmTime'])

# Create surgery start timestamp values
cran_surgery_info['SurgeryStartTimestamp'] = cran_surgery_info[['SurgeryStartDate', 'SurgeryStartTime']].astype(str).agg(' '.join, axis=1)
cran_surgery_info['SurgeryStartTimestamp'][cran_surgery_info.SurgeryStartDate.isna() | cran_surgery_info.SurgeryStartTime.isna()] = np.nan
cran_surgery_info['SurgeryStartTimestamp'] = pd.to_datetime(cran_surgery_info['SurgeryStartTimestamp'],format = '%Y-%m-%d %H:%M:%S' )
cran_surgery_info = cran_surgery_info.drop(columns=['SurgeryStartDate','SurgeryStartTime'])

# Create surgery end timestamp values
cran_surgery_info['SurgeryEndTimestamp'] = cran_surgery_info[['SurgeryEndDate', 'SurgeryEndTime']].astype(str).agg(' '.join, axis=1)
cran_surgery_info['SurgeryEndTimestamp'][cran_surgery_info.SurgeryEndDate.isna() | cran_surgery_info.SurgeryEndTime.isna()] = np.nan
cran_surgery_info['SurgeryEndTimestamp'] = pd.to_datetime(cran_surgery_info['SurgeryEndTimestamp'],format = '%Y-%m-%d %H:%M:%S' )
cran_surgery_info = cran_surgery_info.drop(columns=['SurgeryEndDate','SurgeryEndTime'])

# If surgery end timestamp is missing, add median surgery duration to start timestamp
cran_surgery_info.SurgeryEndTimestamp[cran_surgery_info['SurgeryEndTimestamp'].isna()] = cran_surgery_info.SurgeryStartTimestamp[cran_surgery_info['SurgeryEndTimestamp'].isna()] + pd.DateOffset(seconds=median_SurgeryDurationSecs)

# Tokenise variables appropriately
cran_surgery_info[['SurgeryDescCranial','SurgeryCranialDelay']] = cran_surgery_info[['SurgeryDescCranial','SurgeryCranialDelay']].apply(categorizer,args=(100,)).apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

# Rename timestamp columns
cran_surgery_info = cran_surgery_info.rename(columns={'SurgeryStartTimestamp':'StartTimeStamp','SurgeryEndTimestamp':'EndTimeStamp'})

# Tokenise proper interval variables
cran_surgery_info = cran_surgery_info.melt(id_vars=['GUPI','StartTimeStamp','EndTimeStamp']).drop_duplicates(subset=['GUPI','StartTimeStamp','EndTimeStamp','value']).groupby(['GUPI','StartTimeStamp','EndTimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

# If purported surgery duration is largely negative, add a day to the end time
cran_surgery_info.EndTimeStamp[(cran_surgery_info.EndTimeStamp - cran_surgery_info.StartTimeStamp).astype('timedelta64[s]') < -7200] = cran_surgery_info.EndTimeStamp[(cran_surgery_info.EndTimeStamp - cran_surgery_info.StartTimeStamp).astype('timedelta64[s]') < -7200] + pd.DateOffset(days=1)

# If purported surgery duration is barely negative, switch the start and end dates
new_start_timestamp = cran_surgery_info.EndTimeStamp[(cran_surgery_info.EndTimeStamp-cran_surgery_info.StartTimeStamp).astype('timedelta64[s]')<0]
new_end_timestamp = cran_surgery_info.StartTimeStamp[(cran_surgery_info.EndTimeStamp-cran_surgery_info.StartTimeStamp).astype('timedelta64[s]')<0]

cran_surgery_info.EndTimeStamp[(cran_surgery_info.EndTimeStamp-cran_surgery_info.StartTimeStamp).astype('timedelta64[s]')<0] = new_end_timestamp
cran_surgery_info.StartTimeStamp[(cran_surgery_info.EndTimeStamp-cran_surgery_info.StartTimeStamp).astype('timedelta64[s]')<0] = new_start_timestamp

# If purported surgery duration is over a year, subtract a year
cran_surgery_info.EndTimeStamp[(cran_surgery_info.EndTimeStamp - cran_surgery_info.StartTimeStamp).astype('timedelta64[D]') >= 300] = cran_surgery_info.EndTimeStamp[(cran_surgery_info.EndTimeStamp - cran_surgery_info.StartTimeStamp).astype('timedelta64[D]') >= 300] - pd.DateOffset(years=1)

# If purported durations remain over a day, change the duration to a day
cran_surgery_info.EndTimeStamp[(cran_surgery_info.EndTimeStamp - cran_surgery_info.StartTimeStamp).astype('timedelta64[s]') >= 86400] = cran_surgery_info.StartTimeStamp[(cran_surgery_info.EndTimeStamp - cran_surgery_info.StartTimeStamp).astype('timedelta64[s]') >= 86400] + pd.DateOffset(days=1)

# Add empty EndToken column
cran_surgery_info['EndToken'] = np.nan

## Extracranial surgery information
# Load intracranial surgery data
extra_cran_surgery_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/SurgeriesExtraCranial/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
extra_cran_surgery_info = extra_cran_surgery_info[extra_cran_surgery_info.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)

# Load cranial surgery inspection table to extract allowed predictors
extra_cran_surgery_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/SurgeriesExtraCranial/inspection_table.xlsx',sheet_name='variables',na_values = ["NA","NaN"," ", ""])
extra_cran_surgery_names = extra_cran_surgery_desc.name[extra_cran_surgery_desc.name.isin(extra_cran_surgery_info.columns)].to_list()
extra_cran_surgery_info = extra_cran_surgery_info[['GUPI']+extra_cran_surgery_names].dropna(subset=extra_cran_surgery_names,how='all').reset_index(drop=True)

# Calculate median surgery start time and median surgery duration
median_SurgeryStartTime = pd.to_datetime(extra_cran_surgery_info.SurgeryStartTime,format = '%H:%M:%S').dropna().median().strftime('%H:%M:%S')
median_SurgeryDurationSecs = (pd.to_datetime(extra_cran_surgery_info.SurgeryEndDate+' '+extra_cran_surgery_info.SurgeryEndTime,format = '%Y-%m-%d %H:%M:%S') - pd.to_datetime(extra_cran_surgery_info.SurgeryStartDate+' '+extra_cran_surgery_info.SurgeryStartTime,format = '%Y-%m-%d %H:%M:%S')).astype('timedelta64[s]').dropna().median()

# If cranial surgery start time is missing, fill in with admission time (if on the same day), or fill in with median insertion time
extra_cran_surgery_info = extra_cran_surgery_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUAdmDate','ICUAdmTime']],how='left',on='GUPI')
extra_cran_surgery_info.SurgeryStartTime[(extra_cran_surgery_info.SurgeryStartTime.isna())&(extra_cran_surgery_info.SurgeryStartDate == extra_cran_surgery_info.ICUAdmDate)] = extra_cran_surgery_info.ICUAdmTime[(extra_cran_surgery_info.SurgeryStartTime.isna())&(extra_cran_surgery_info.SurgeryStartDate == extra_cran_surgery_info.ICUAdmDate)]
extra_cran_surgery_info.SurgeryStartTime[extra_cran_surgery_info.SurgeryStartTime.isna()] = median_SurgeryStartTime
extra_cran_surgery_info = extra_cran_surgery_info.drop(columns=['ICUAdmDate','ICUAdmTime'])

# Create surgery start timestamp values
extra_cran_surgery_info['SurgeryStartTimestamp'] = extra_cran_surgery_info[['SurgeryStartDate', 'SurgeryStartTime']].astype(str).agg(' '.join, axis=1)
extra_cran_surgery_info['SurgeryStartTimestamp'][extra_cran_surgery_info.SurgeryStartDate.isna() | extra_cran_surgery_info.SurgeryStartTime.isna()] = np.nan
extra_cran_surgery_info['SurgeryStartTimestamp'] = pd.to_datetime(extra_cran_surgery_info['SurgeryStartTimestamp'],format = '%Y-%m-%d %H:%M:%S' )
extra_cran_surgery_info = extra_cran_surgery_info.drop(columns=['SurgeryStartDate','SurgeryStartTime'])

# Create surgery end timestamp values
extra_cran_surgery_info['SurgeryEndTimestamp'] = extra_cran_surgery_info[['SurgeryEndDate', 'SurgeryEndTime']].astype(str).agg(' '.join, axis=1)
extra_cran_surgery_info['SurgeryEndTimestamp'][extra_cran_surgery_info.SurgeryEndDate.isna() | extra_cran_surgery_info.SurgeryEndTime.isna()] = np.nan
extra_cran_surgery_info['SurgeryEndTimestamp'] = pd.to_datetime(extra_cran_surgery_info['SurgeryEndTimestamp'],format = '%Y-%m-%d %H:%M:%S' )
extra_cran_surgery_info = extra_cran_surgery_info.drop(columns=['SurgeryEndDate','SurgeryEndTime'])

# If surgery end timestamp is missing, add median surgery duration to start timestamp
extra_cran_surgery_info.SurgeryEndTimestamp[extra_cran_surgery_info['SurgeryEndTimestamp'].isna()] = extra_cran_surgery_info.SurgeryStartTimestamp[extra_cran_surgery_info['SurgeryEndTimestamp'].isna()] + pd.DateOffset(seconds=median_SurgeryDurationSecs)

# Tokenise variables appropriately
extra_cran_surgery_info[['SurgeryDescExtraCranial','SurgeryExtraCranialDelay']] = extra_cran_surgery_info[['SurgeryDescExtraCranial','SurgeryExtraCranialDelay']].apply(categorizer,args=(100,)).apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

# Rename timestamp columns
extra_cran_surgery_info = extra_cran_surgery_info.rename(columns={'SurgeryStartTimestamp':'StartTimeStamp','SurgeryEndTimestamp':'EndTimeStamp'})

# Tokenise proper interval variables
extra_cran_surgery_info = extra_cran_surgery_info.melt(id_vars=['GUPI','StartTimeStamp','EndTimeStamp']).drop_duplicates(subset=['GUPI','StartTimeStamp','EndTimeStamp','value']).groupby(['GUPI','StartTimeStamp','EndTimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

# If purported surgery duration is largely negative, add a day to the end time
extra_cran_surgery_info.EndTimeStamp[(extra_cran_surgery_info.EndTimeStamp - extra_cran_surgery_info.StartTimeStamp).astype('timedelta64[s]') < -14400] = extra_cran_surgery_info.EndTimeStamp[(extra_cran_surgery_info.EndTimeStamp - extra_cran_surgery_info.StartTimeStamp).astype('timedelta64[s]') < -14400] + pd.DateOffset(days=1)

# If purported surgery duration is barely negative, switch the start and end dates
new_start_timestamp = extra_cran_surgery_info.EndTimeStamp[(extra_cran_surgery_info.EndTimeStamp-extra_cran_surgery_info.StartTimeStamp).astype('timedelta64[s]')<0]
new_end_timestamp = extra_cran_surgery_info.StartTimeStamp[(extra_cran_surgery_info.EndTimeStamp-extra_cran_surgery_info.StartTimeStamp).astype('timedelta64[s]')<0]

extra_cran_surgery_info.EndTimeStamp[(extra_cran_surgery_info.EndTimeStamp-extra_cran_surgery_info.StartTimeStamp).astype('timedelta64[s]')<0] = new_end_timestamp
extra_cran_surgery_info.StartTimeStamp[(extra_cran_surgery_info.EndTimeStamp-extra_cran_surgery_info.StartTimeStamp).astype('timedelta64[s]')<0] = new_start_timestamp

# If purported surgery duration is over a year, subtract a year
extra_cran_surgery_info.EndTimeStamp[(extra_cran_surgery_info.EndTimeStamp - extra_cran_surgery_info.StartTimeStamp).astype('timedelta64[D]') >= 300] = extra_cran_surgery_info.EndTimeStamp[(extra_cran_surgery_info.EndTimeStamp - extra_cran_surgery_info.StartTimeStamp).astype('timedelta64[D]') >= 300] - pd.DateOffset(years=1)

# If purported durations remain over a day, change the duration to a day
extra_cran_surgery_info.EndTimeStamp[(extra_cran_surgery_info.EndTimeStamp - extra_cran_surgery_info.StartTimeStamp).astype('timedelta64[s]') >= 86400] = extra_cran_surgery_info.StartTimeStamp[(extra_cran_surgery_info.EndTimeStamp - extra_cran_surgery_info.StartTimeStamp).astype('timedelta64[s]') >= 86400] + pd.DateOffset(days=1)

# Add empty EndToken column
extra_cran_surgery_info['EndToken'] = np.nan

## Concatenate time-intervalled predictors
# Concatenate
time_interval_predictors = pd.concat([icp_interval_info,intubation_interval_info,mech_ventilation_interval_info,cran_surgery_info,extra_cran_surgery_info],ignore_index=True)

# Group by GUPI, StartTimeStamp, EndTimeStamp, and merge tokens
time_interval_tokens = time_interval_predictors[['GUPI','StartTimeStamp','EndTimeStamp','Token']].groupby(['GUPI','StartTimeStamp','EndTimeStamp'],as_index=False).Token.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'}).reset_index(drop=True)
time_interval_end_tokens = time_interval_predictors[['GUPI','StartTimeStamp','EndTimeStamp','EndToken']].fillna('').groupby(['GUPI','StartTimeStamp','EndTimeStamp'],as_index=False).EndToken.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'}).reset_index(drop=True)
time_interval_end_tokens.EndToken[time_interval_end_tokens.EndToken == ''] = np.nan
time_interval_predictors = time_interval_tokens.merge(time_interval_end_tokens,how='left',on=['GUPI','StartTimeStamp','EndTimeStamp'])

# Iterate through entries, ensure unique tokens and end tokens
for curr_idx in tqdm(range(time_interval_predictors.shape[0]), 'Cleaning time-intervalled predictors'):
    curr_token_set = time_interval_predictors.Token[curr_idx]
    cleaned_token_set = ' '.join(np.sort(np.unique(curr_token_set.split())))
    time_interval_predictors.Token[curr_idx] = cleaned_token_set

# Filter out all datapoints with start timestamp after ICU discharge
time_interval_predictors = time_interval_predictors.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
time_interval_predictors = time_interval_predictors[time_interval_predictors.StartTimeStamp <= time_interval_predictors.ICUDischTimeStamp].drop(columns='ICUDischTimeStamp').reset_index(drop=True)
    
# Sort values and save time-intervalled predictors
time_interval_predictors = time_interval_predictors.sort_values(by=['GUPI','StartTimeStamp','EndTimeStamp']).reset_index(drop=True)
time_interval_predictors.to_pickle(os.path.join(form_pred_dir,'categorical_time_interval_predictors.pkl'))

### VI. Format dated single-event predictors in CENTER-TBI
## DailyVitals information
# Load DailyVitals data
daily_vitals_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/DailyVitals/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
daily_vitals_info = daily_vitals_info[daily_vitals_info.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)

# Remove all entries without date or `DVTimepoint`
daily_vitals_info = daily_vitals_info[(daily_vitals_info.DVTimepoint!='None')|(~daily_vitals_info.DVDate.isna())].reset_index(drop=True)

# Convert dates from string to date format
daily_vitals_info.DVDate = pd.to_datetime(daily_vitals_info.DVDate,format = '%Y-%m-%d')

# Load DailyVitals inspection table to extract allowed predictors
daily_vitals_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/DailyVitals/inspection_table.xlsx',sheet_name='daily variables',na_values = ["NA","NaN"," ", ""])
daily_vitals_names = daily_vitals_desc.name[daily_vitals_desc.name.isin(daily_vitals_info.columns)].to_list()
daily_vitals_info = daily_vitals_info[['GUPI']+daily_vitals_names].dropna(subset=[n for n in daily_vitals_names if n not in ['DVTimepoint','DVDate']],how='all').reset_index(drop=True)
daily_vitals_info.columns = daily_vitals_info.columns.str.replace('_','')
daily_vitals_names = [x.replace('_', '') for x in daily_vitals_names]

# Iterate through GUPIs and fix `DVDate` based on `DVTimepoint` information if possible
problem_GUPIs = []
for curr_GUPI in tqdm(daily_vitals_info.GUPI.unique(),'Fixing daily vitals dates if possible'):
    curr_GUPI_daily_vitals = daily_vitals_info[(daily_vitals_info.GUPI==curr_GUPI)&(daily_vitals_info.DVTimepoint!='None')].reset_index(drop=True)
    if curr_GUPI_daily_vitals.DVDate.isna().all():
        print('Problem GUPI: '+curr_GUPI)
        problem_GUPIs.append(curr_GUPI)
        continue
    curr_date_diff = int((curr_GUPI_daily_vitals.DVDate.dt.day - curr_GUPI_daily_vitals.DVTimepoint.astype(float)).mode()[0])
    fixed_date_vector = pd.Series([pd.Timestamp('1970-01-01') + pd.DateOffset(days=dt+curr_date_diff) for dt in (curr_GUPI_daily_vitals.DVTimepoint.astype(float)-1)],index=daily_vitals_info[(daily_vitals_info.GUPI==curr_GUPI)&(daily_vitals_info.DVTimepoint!='None')].index)
    daily_vitals_info.DVDate[(daily_vitals_info.GUPI==curr_GUPI)&(daily_vitals_info.DVTimepoint!='None')] = fixed_date_vector    
    
# Remove `DVTimepoint` from dataframe and apply categorizer
daily_vitals_info = daily_vitals_info.drop(columns='DVTimepoint').apply(categorizer)

# Separate categorical and numeric variables into separate dataframes
numeric_daily_vitals_names = np.sort(daily_vitals_info.select_dtypes(include=['number']).columns.values)
categorical_daily_vitals_names = np.sort(daily_vitals_info.select_dtypes(exclude=['number']).drop(columns=['GUPI','DVDate']).columns.values)

# Melt numeric dataframe into long form
numeric_daily_vitals_predictors = daily_vitals_info[np.insert(numeric_daily_vitals_names,0,['GUPI','DVDate'])].dropna(axis=1,how='all').dropna(subset=numeric_daily_vitals_names,how='all').reset_index(drop=True).melt(id_vars=['GUPI','DVDate'],var_name='VARIABLE',value_name='VALUE').rename(columns={'DVDate':'Date'})

# Remove formatting from all categorical variable string values
categorical_daily_vitals_predictors = daily_vitals_info[np.insert(categorical_daily_vitals_names,0,['GUPI','DVDate'])].dropna(axis=1,how='all').dropna(subset=categorical_daily_vitals_names,how='all').reset_index(drop=True)
categorical_daily_vitals_predictors[categorical_daily_vitals_names] = categorical_daily_vitals_predictors[categorical_daily_vitals_names].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

# Rename date column and construct tokens
categorical_daily_vitals_predictors = categorical_daily_vitals_predictors.rename(columns={'DVDate':'Date'}).melt(id_vars=['GUPI','Date']).drop_duplicates(subset=['GUPI','Date','value']).groupby(['GUPI','Date'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Transitions of Care information
# Load TransitionsOfCare data
toc_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/TransitionsOfCare/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients and in-hospital transfers
toc_info = toc_info[toc_info.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)
toc_info = toc_info[toc_info.Location != 'PostDisch'].reset_index(drop=True)

# Remove all entries without date information
toc_info = toc_info[(~toc_info.DateClinReadyForTransfer.isna())|(~toc_info.DateEffectiveTransfer.isna())].reset_index(drop=True)

# Load TransitionsOfCare inspection table to extract allowed predictors
toc_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/TransitionsOfCare/inspection_table.xlsx',sheet_name='inspection_table',na_values = ["NA","NaN"," ", ""])
toc_names = toc_desc.name[toc_desc.name.isin(toc_info.columns)].to_list()
toc_info = toc_info[['GUPI']+toc_names].dropna(subset=[n for n in toc_names if n not in ['DateClinReadyForTransfer','DateEffectiveTransfer']],how='all').reset_index(drop=True)

# If effective transfer date is missing, replace with ready for transfer date
toc_info.DateEffectiveTransfer[(toc_info.DateEffectiveTransfer.isna())&(~toc_info.DateClinReadyForTransfer.isna())] = toc_info.DateClinReadyForTransfer[(toc_info.DateEffectiveTransfer.isna())&(~toc_info.DateClinReadyForTransfer.isna())]

# Remove ready for transfer date, rename effective transfer date, and categorize variables appropriately
toc_info = toc_info.drop(columns='DateClinReadyForTransfer').rename(columns={'DateEffectiveTransfer':'Date'}).apply(categorizer)

# Create new variable which maps transfer locations
toc_info['Transfer'] = toc_info['TransFrom'] + '-to-' + toc_info['TransTo']
toc_info = toc_info.drop(columns=['TransFrom','TransTo'])

# Convert dates from string to date format
toc_info.Date = pd.to_datetime(toc_info.Date,format = '%Y-%m-%d')

# Merge ICU admission and discharge dates
toc_info = toc_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUAdmDate','ICUDischDate']],how='left',on='GUPI')
toc_info[['ICUAdmDate','ICUDischDate']] = toc_info[['ICUAdmDate','ICUDischDate']].apply(lambda x: pd.to_datetime(x,format = '%Y-%m-%d'))

# Remove ICU discharge transfers
toc_info = toc_info[~((toc_info.ICUDischDate == toc_info.Date)&(toc_info.Transfer.str.startswith('ICU-to-'))&(toc_info.Transfer.isin(['ICU-to-CT','ICU-to-MRI','ICU-to-OR'])))].reset_index(drop=True).drop(columns=['ICUAdmDate','ICUDischDate','Location'])

# Remove formatting from all categorical variable string values and construct tokens
toc_info[[n for n in list(toc_info) if n not in ['GUPI','Date']]] = toc_info[[n for n in list(toc_info) if n not in ['GUPI','Date']]].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
toc_info = toc_info.melt(id_vars=['GUPI','Date']).drop_duplicates(subset=['GUPI','Date','value']).groupby(['GUPI','Date'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Rivermead Post-Concussion Symptoms Questionnaire (RPQ) and Galveston Orientation & Amnesia Test (GOAT)
# Load outcomes data
outcomes_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/Outcomes/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients and baseline outcomes
outcomes_info = outcomes_info[outcomes_info.GUPI.isin(cv_splits.GUPI)].reset_index(drop=True)
outcomes_info = outcomes_info[outcomes_info.Timepoint == 'Base'].dropna(axis=1,how='all').reset_index(drop=True)

# Separate RPQ and GOAT dataframes and construct tokens
rpq_names = [n for n in outcomes_info if n.startswith('RPQ') & (n not in ['RPQCompleteStatus','RPQQuestionnaireMode'])]
goat_names = [n for n in outcomes_info if n.startswith('GOAT') & (n not in ['GOATCompleteStatus','GOATNeuroPsychCompCode','GOATTestCompletedOptions','GOATTestComplNonStandAdminOTHER'])]

rpq_info = outcomes_info[['GUPI']+rpq_names].dropna(subset=[n for n in rpq_names if n not in ['RPQDate']],how='all').dropna(axis=1,how='all').reset_index(drop=True).merge(CENTER_TBI_ICU_datetime[['GUPI','ICUAdmDate','ICUDischDate']],how='left',on='GUPI').reset_index(drop=True)
rpq_info[['ICUAdmDate','ICUDischDate']] = rpq_info[['ICUAdmDate','ICUDischDate']].apply(lambda x: pd.to_datetime(x,format = '%Y-%m-%d'))
rpq_info.RPQDate = pd.to_datetime(rpq_info.RPQDate,format = '%Y-%m-%d')
rpq_info.RPQDate[rpq_info.RPQDate.isna()] = rpq_info.ICUAdmDate[rpq_info.RPQDate.isna()] 
rpq_info = rpq_info[rpq_info.RPQDate <= rpq_info.ICUDischDate].drop(columns=['ICUAdmDate','ICUDischDate']).dropna(axis=1,how='all').rename(columns={'RPQDate':'Date'}).reset_index(drop=True).apply(categorizer,args=(100,))
categorical_rpq_predictors = rpq_info.copy()
categorical_rpq_predictors[[n for n in rpq_info if n not in ['GUPI','Date']]] = rpq_info[[n for n in rpq_info if n not in ['GUPI','Date']]].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
categorical_rpq_predictors = categorical_rpq_predictors.melt(id_vars=['GUPI','Date']).drop_duplicates(subset=['GUPI','Date','value']).groupby(['GUPI','Date'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

goat_info = outcomes_info[['GUPI']+goat_names].dropna(subset=[n for n in goat_names if n not in ['GOATDate']],how='all').dropna(axis=1,how='all').reset_index(drop=True).merge(CENTER_TBI_ICU_datetime[['GUPI','ICUAdmDate','ICUDischDate']],how='left',on='GUPI').reset_index(drop=True)
goat_info[['ICUAdmDate','ICUDischDate']] = goat_info[['ICUAdmDate','ICUDischDate']].apply(lambda x: pd.to_datetime(x,format = '%Y-%m-%d'))
goat_info.GOATDate = pd.to_datetime(goat_info.GOATDate,format = '%Y-%m-%d')
goat_info.GOATDate[goat_info.GOATDate.isna()] = goat_info.ICUAdmDate[goat_info.GOATDate.isna()] 
goat_info = goat_info[goat_info.GOATDate <= goat_info.ICUDischDate].drop(columns=['ICUAdmDate','ICUDischDate']).dropna(axis=1,how='all').rename(columns={'GOATDate':'Date'}).reset_index(drop=True).apply(categorizer,args=(100,))
categorical_goat_predictors = goat_info.copy()
categorical_goat_predictors[[n for n in goat_info if n not in ['GUPI','Date']]] = goat_info[[n for n in goat_info if n not in ['GUPI','Date']]].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
categorical_goat_predictors = categorical_goat_predictors.melt(id_vars=['GUPI','Date']).drop_duplicates(subset=['GUPI','Date','value']).groupby(['GUPI','Date'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Compile and save dated single-event predictors
# Categorical predictors - ensure unique tokens per GUPI and save
categorical_date_tokens = pd.concat([categorical_daily_vitals_predictors,toc_info,categorical_rpq_predictors,categorical_goat_predictors],ignore_index=True)

# Group by GUPI and Date and merge tokens
categorical_date_tokens = categorical_date_tokens.groupby(['GUPI','Date'],as_index=False).Token.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'}).reset_index(drop=True)

# Iterate through entries, ensure unique tokens and end tokens
for curr_idx in tqdm(range(categorical_date_tokens.shape[0]), 'Cleaning dated single-event predictors'):
    curr_token_set = categorical_date_tokens.Token[curr_idx]
    cleaned_token_set = ' '.join(np.sort(np.unique(curr_token_set.split())))
    categorical_date_tokens.Token[curr_idx] = cleaned_token_set

# Filter out all datapoints with date after ICU discharge
categorical_date_tokens = categorical_date_tokens.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
categorical_date_tokens = categorical_date_tokens[categorical_date_tokens.Date <= categorical_date_tokens.ICUDischTimeStamp].drop(columns='ICUDischTimeStamp').reset_index(drop=True)

# Sort values and save dated single-event predictors
categorical_date_tokens = categorical_date_tokens.sort_values(by=['GUPI','Date']).reset_index(drop=True)
categorical_date_tokens.to_pickle(os.path.join(form_pred_dir,'categorical_date_event_predictors.pkl'))

# Numeric predictors
numeric_daily_vitals_predictors = numeric_daily_vitals_predictors.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
numeric_daily_vitals_predictors = numeric_daily_vitals_predictors[numeric_daily_vitals_predictors.Date <= numeric_daily_vitals_predictors.ICUDischTimeStamp].drop(columns='ICUDischTimeStamp').reset_index(drop=True)
numeric_daily_vitals_predictors = numeric_daily_vitals_predictors.sort_values(by=['GUPI','Date','VARIABLE']).reset_index(drop=True)
numeric_daily_vitals_predictors.to_pickle(os.path.join(form_pred_dir,'numeric_date_event_predictors.pkl'))

### VII. Format timestamped single-event predictors in CENTER-TBI
# Load demographic, history, injury characeristic timestamped predictors
demo_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
demo_info = demo_info[demo_info.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)
demo_names = list(demo_info.columns)

# Load inspection table and gather names of timestamped variables
demo_timestamped_variable_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/DemoInjHospMedHx/inspection_table.xlsx',sheet_name='time_sensitive_baseline',na_values = ["NA","NaN"," ", ""])

# Remove variables which have already been accounted for
demo_timestamped_variable_desc = demo_timestamped_variable_desc[~demo_timestamped_variable_desc.variableType.isin(['ComplCRBSI','PneumAntibiotic','VentPneum'])].reset_index(drop=True)

# Filter timestamped variables
demo_timestamped_names = demo_timestamped_variable_desc.name[demo_timestamped_variable_desc.name.isin(demo_names)].to_list()
timestamped_demo_info = demo_info[['GUPI']+demo_timestamped_names].dropna(subset=demo_timestamped_names,how='all').reset_index(drop=True)

## First hospital assessment information
# Extract names for `GCSFirstHosp` variables
GCSFirstHosp_names = demo_timestamped_variable_desc.name[(demo_timestamped_variable_desc.name.isin(demo_names))&(demo_timestamped_variable_desc.variableType=='GCSFirstHosp')].to_list()
GCSFirstHosp_info = timestamped_demo_info[['GUPI']+GCSFirstHosp_names].dropna(subset=[n for n in GCSFirstHosp_names if n not in ['GCSFirstHospScoreDate','GCSFirstHospScoreTime']],how='all').reset_index(drop=True)

# Construct first hospital assessment timestamp
GCSFirstHosp_info['TimeStamp'] = GCSFirstHosp_info[['GCSFirstHospScoreDate', 'GCSFirstHospScoreTime']].astype(str).agg(' '.join, axis=1)
GCSFirstHosp_info['TimeStamp'][GCSFirstHosp_info.GCSFirstHospScoreDate.isna() | GCSFirstHosp_info.GCSFirstHospScoreTime.isna()] = np.nan
GCSFirstHosp_info['TimeStamp'] = pd.to_datetime(GCSFirstHosp_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# If timestamp is missing, replace with ICU admission timestamp and apply the categorizer
GCSFirstHosp_info = GCSFirstHosp_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUAdmTimeStamp']],how='left',on='GUPI')
GCSFirstHosp_info.TimeStamp[GCSFirstHosp_info.TimeStamp.isna()] = GCSFirstHosp_info.ICUAdmTimeStamp[GCSFirstHosp_info.TimeStamp.isna()]
GCSFirstHosp_info = GCSFirstHosp_info.drop(columns=['ICUAdmTimeStamp','GCSFirstHospScoreDate','GCSFirstHospScoreTime']).apply(categorizer)

# Fix `GCSFirstHospEyes` variable mismatch
GCSFirstHosp_info.GCSFirstHospEyes[GCSFirstHosp_info.GCSFirstHospEyes.isin([4.0, 2.0, 1.0, 3.0])] = GCSFirstHosp_info.GCSFirstHospEyes[GCSFirstHosp_info.GCSFirstHospEyes.isin([4.0, 2.0, 1.0, 3.0])].astype(int).astype(str)

# Tokenise variables and construct final tokens
GCSFirstHosp_info[[n for n in list(GCSFirstHosp_info) if n not in ['GUPI','TimeStamp']]] = GCSFirstHosp_info[[n for n in list(GCSFirstHosp_info) if n not in ['GUPI','TimeStamp']]].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
GCSFirstHosp_info = GCSFirstHosp_info.melt(id_vars=['GUPI','TimeStamp']).drop_duplicates(subset=['GUPI','TimeStamp','value']).groupby(['GUPI','TimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Other assessment information
# Extract names for `GCSOther` variables
GCSOther_names = demo_timestamped_variable_desc.name[(demo_timestamped_variable_desc.name.isin(demo_names))&(demo_timestamped_variable_desc.variableType=='GCSOther')].to_list()
GCSOther_info = timestamped_demo_info[['GUPI']+GCSOther_names].dropna(subset=[n for n in GCSOther_names if n not in ['GCSOtherDate','GCSOtherTime']],how='all').reset_index(drop=True)

# Construct Other assessment timestamp
GCSOther_info['TimeStamp'] = GCSOther_info[['GCSOtherDate', 'GCSOtherTime']].astype(str).agg(' '.join, axis=1)
GCSOther_info['TimeStamp'][GCSOther_info.GCSOtherDate.isna() | GCSOther_info.GCSOtherTime.isna()] = np.nan
GCSOther_info['TimeStamp'] = pd.to_datetime(GCSOther_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S')

# If timestamp is missing, replace with ICU admission timestamp if on the same day
GCSOther_info = GCSOther_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUAdmTimeStamp','ICUAdmDate']],how='left',on='GUPI')
GCSOther_info.TimeStamp[GCSOther_info.TimeStamp.isna()&(GCSOther_info.GCSOtherDate == GCSOther_info.ICUAdmDate)] = GCSOther_info.ICUAdmTimeStamp[GCSOther_info.TimeStamp.isna()&(GCSOther_info.GCSOtherDate == GCSOther_info.ICUAdmDate)]
GCSOther_info = GCSOther_info.drop(columns=['ICUAdmTimeStamp','ICUAdmDate'])

# If timestamp is still missing, replace with median assessment time and apply categorizer
median_assessment_time = pd.to_datetime(GCSOther_info.GCSOtherTime,format = '%H:%M:%S').dropna().median().strftime('%H:%M:%S')
GCSOther_info.TimeStamp[GCSOther_info.TimeStamp.isna()] = pd.to_datetime(GCSOther_info.GCSOtherDate[GCSOther_info.TimeStamp.isna()] + ' ' + median_assessment_time,format = '%Y-%m-%d %H:%M:%S')
GCSOther_info = GCSOther_info.drop(columns=['GCSOtherDate','GCSOtherTime']).apply(categorizer)

# Fix `GCSOtherEyes` variable mismatch
GCSOther_info.GCSOtherEyes[GCSOther_info.GCSOtherEyes.isin([4.0, 2.0, 1.0, 3.0])] = GCSOther_info.GCSOtherEyes[GCSOther_info.GCSOtherEyes.isin([4.0, 2.0, 1.0, 3.0])].astype(int).astype(str)

# Tokenise variables and construct final tokens
GCSOther_info[[n for n in list(GCSOther_info) if n not in ['GUPI','TimeStamp']]] = GCSOther_info[[n for n in list(GCSOther_info) if n not in ['GUPI','TimeStamp']]].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
GCSOther_info = GCSOther_info.melt(id_vars=['GUPI','TimeStamp']).drop_duplicates(subset=['GUPI','TimeStamp','value']).groupby(['GUPI','TimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Hospital discharge information
# Extract names for `HospDisch` variables
HospDisch_names = demo_timestamped_variable_desc.name[(demo_timestamped_variable_desc.name.isin(demo_names))&(demo_timestamped_variable_desc.variableType=='HospDisch')].to_list()
HospDisch_info = timestamped_demo_info[['GUPI']+HospDisch_names].dropna(subset=[n for n in HospDisch_names if n not in ['HospDischDate','HospDischTime']],how='all').reset_index(drop=True)

# Remove all instances with missing discharge date
HospDisch_info = HospDisch_info[~HospDisch_info.HospDischDate.isna()].reset_index(drop=True)

# Construct hospital discharge timestamp
HospDisch_info['TimeStamp'] = HospDisch_info[['HospDischDate', 'HospDischTime']].astype(str).agg(' '.join, axis=1)
HospDisch_info.loc[HospDisch_info.TimeStamp.str.contains('nan'), 'TimeStamp'] = np.nan
HospDisch_info['TimeStamp'] = pd.to_datetime(HospDisch_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S')

# If discharge time is missing, and ICU discharge date is the same date, replace with ICU dicharge time. Also filter patients with concurrent hospital/ICU discharge
HospDisch_info = HospDisch_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp','ICUDischDate']],how='left',on='GUPI')
HospDisch_info.TimeStamp[HospDisch_info.TimeStamp.isna()&(HospDisch_info.HospDischDate == HospDisch_info.ICUDischDate)] = HospDisch_info.ICUDischTimeStamp[HospDisch_info.TimeStamp.isna()&(HospDisch_info.HospDischDate == HospDisch_info.ICUDischDate)]
HospDisch_info = HospDisch_info[HospDisch_info.TimeStamp <= HospDisch_info.ICUDischTimeStamp].dropna(axis=1,how='all').reset_index(drop=True)
HospDisch_info = HospDisch_info.drop(columns=['ICUDischTimeStamp','ICUDischDate','HospDischDate','HospDischTime'])

# Collapse repeated columns
HospNeuroworseEpisode_col = (HospDisch_info['HospNeuroworseEpisode'].sum(1)/4).astype(int)
HospDisch_info = HospDisch_info.drop(columns = ['HospNeuroworseEpisode'])
HospDisch_info['HospNeuroworseEpisode'] = HospNeuroworseEpisode_col

# Apply categorizer
HospDisch_info = HospDisch_info.apply(categorizer)

# Tokenise variables and construct final tokens
format_cols = [n for n in list(HospDisch_info) if n not in ['GUPI','TimeStamp']]
HospDisch_info[format_cols] = HospDisch_info[format_cols].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
HospDisch_info = HospDisch_info.melt(id_vars=['GUPI','TimeStamp']).drop_duplicates(subset=['GUPI','TimeStamp','value']).groupby(['GUPI','TimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Reintubation information
# Extract names for `Reintubation` variables
Reintubation_names = demo_timestamped_variable_desc.name[(demo_timestamped_variable_desc.name.isin(demo_names))&(demo_timestamped_variable_desc.variableType=='Reintubation')].to_list()
Reintubation_info = timestamped_demo_info[['GUPI']+Reintubation_names].dropna(subset=Reintubation_names,how='all').reset_index(drop=True)
Reintubation_info = Reintubation_info[(Reintubation_info.ReIntubation != 0)|(~Reintubation_info.ReIntubationStartDate.isna())].reset_index(drop=True)

# Construct reintubation timestamp
Reintubation_info['TimeStamp'] = Reintubation_info[['ReIntubationStartDate', 'ReIntubationStartTime']].astype(str).agg(' '.join, axis=1)
Reintubation_info['TimeStamp'][Reintubation_info.ReIntubationStartDate.isna() | Reintubation_info.ReIntubationStartTime.isna()] = np.nan
Reintubation_info['TimeStamp'] = pd.to_datetime(Reintubation_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S')

# If timestamp is missing, replace with median reintubation time and apply categorizer
median_reintubation_time = pd.to_datetime(Reintubation_info.ReIntubationStartTime,format = '%H:%M:%S').dropna().median().strftime('%H:%M:%S')
Reintubation_info.TimeStamp[Reintubation_info.TimeStamp.isna()] = pd.to_datetime(Reintubation_info.ReIntubationStartDate[Reintubation_info.TimeStamp.isna()] + ' ' + median_reintubation_time,format = '%Y-%m-%d %H:%M:%S')
Reintubation_info = Reintubation_info.drop(columns=['ReIntubationStartDate','ReIntubationStartTime']).apply(categorizer)

# Tokenise variables and construct final tokens
Reintubation_info['ReIntubation'] = 'ReIntubation_'+Reintubation_info['ReIntubation'].str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True).fillna('NAN')
Reintubation_info = Reintubation_info.melt(id_vars=['GUPI','TimeStamp']).drop_duplicates(subset=['GUPI','TimeStamp','value']).groupby(['GUPI','TimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## ReMechVentilation information
# Extract names for `ReMechVentilation` variables
ReMechVentilation_names = demo_timestamped_variable_desc.name[(demo_timestamped_variable_desc.name.isin(demo_names))&(demo_timestamped_variable_desc.variableType=='ReMechVentilation')].to_list()
ReMechVentilation_info = timestamped_demo_info[['GUPI']+ReMechVentilation_names].dropna(subset=ReMechVentilation_names,how='all').reset_index(drop=True)
ReMechVentilation_info = ReMechVentilation_info[(ReMechVentilation_info.ReMechVentilation != 0)|(~ReMechVentilation_info.ReMechVentilationStartDate.isna())].reset_index(drop=True)

# Construct ReMechVentilation timestamp
ReMechVentilation_info['TimeStamp'] = ReMechVentilation_info[['ReMechVentilationStartDate', 'ReMechVentilationStartTime']].astype(str).agg(' '.join, axis=1)
ReMechVentilation_info['TimeStamp'][ReMechVentilation_info.ReMechVentilationStartDate.isna() | ReMechVentilation_info.ReMechVentilationStartTime.isna()] = np.nan
ReMechVentilation_info['TimeStamp'] = pd.to_datetime(ReMechVentilation_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S')

# If timestamp is missing, replace with median ReMechVentilation time and apply categorizer
median_ReMechVentilation_time = pd.to_datetime(ReMechVentilation_info.ReMechVentilationStartTime,format = '%H:%M:%S').dropna().median().strftime('%H:%M:%S')
ReMechVentilation_info.TimeStamp[ReMechVentilation_info.TimeStamp.isna()] = pd.to_datetime(ReMechVentilation_info.ReMechVentilationStartDate[ReMechVentilation_info.TimeStamp.isna()] + ' ' + median_ReMechVentilation_time,format = '%Y-%m-%d %H:%M:%S')
ReMechVentilation_info = ReMechVentilation_info.drop(columns=['ReMechVentilationStartDate','ReMechVentilationStartTime']).apply(categorizer)

# Tokenise variables and construct final tokens
format_cols = [n for n in list(ReMechVentilation_info) if n not in ['GUPI','TimeStamp']]
ReMechVentilation_info[format_cols] = ReMechVentilation_info[format_cols].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
ReMechVentilation_info = ReMechVentilation_info.melt(id_vars=['GUPI','TimeStamp']).drop_duplicates(subset=['GUPI','TimeStamp','value']).groupby(['GUPI','TimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

# Load DailyVitals data
daily_vitals_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/DailyVitals/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
daily_vitals_info = daily_vitals_info[daily_vitals_info.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)

# Remove all entries without date or `DVTimepoint`
daily_vitals_info = daily_vitals_info[(daily_vitals_info.DVTimepoint!='None')|(~daily_vitals_info.DVDate.isna())].reset_index(drop=True)

# Convert dates from string to date format
daily_vitals_info.DVDate = pd.to_datetime(daily_vitals_info.DVDate,format = '%Y-%m-%d')

# Load inspection table and gather names of timestamped variables
daily_vitals_names = list(daily_vitals_info.columns)
daily_vitals_timestamped_variable_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/DailyVitals/inspection_table.xlsx',sheet_name='time-stamped variables',na_values = ["NA","NaN"," ", ""])
daily_vitals_timestamped_names = ['DVTimepoint']+daily_vitals_timestamped_variable_desc.name[daily_vitals_timestamped_variable_desc.name.isin(daily_vitals_names)].unique().tolist()

# Filter timestamped variables
timestamped_daily_vitals_info = daily_vitals_info[['GUPI']+daily_vitals_timestamped_names].dropna(subset=[n for n in daily_vitals_timestamped_names if n not in ['DVTimepoint','DVDate'] and not n.endswith('Time')],how='all').reset_index(drop=True)

# Iterate through GUPIs and fix `DVDate` based on `DVTimepoint` information if possible
problem_GUPIs = []
for curr_GUPI in tqdm(timestamped_daily_vitals_info.GUPI.unique(),'Fixing daily vitals dates if possible'):
    curr_GUPI_daily_vitals = timestamped_daily_vitals_info[(timestamped_daily_vitals_info.GUPI==curr_GUPI)&(timestamped_daily_vitals_info.DVTimepoint!='None')].reset_index(drop=True)
    if curr_GUPI_daily_vitals.DVDate.isna().all():
        print('Problem GUPI: '+curr_GUPI)
        problem_GUPIs.append(curr_GUPI)
        continue
    curr_date_diff = int((curr_GUPI_daily_vitals.DVDate.dt.day - curr_GUPI_daily_vitals.DVTimepoint.astype(float)).mode()[0])
    fixed_date_vector = pd.Series([pd.Timestamp('1970-01-01') + pd.DateOffset(days=dt+curr_date_diff) for dt in (curr_GUPI_daily_vitals.DVTimepoint.astype(float)-1)],index=timestamped_daily_vitals_info[(timestamped_daily_vitals_info.GUPI==curr_GUPI)&(timestamped_daily_vitals_info.DVTimepoint!='None')].index)
    timestamped_daily_vitals_info.DVDate[(timestamped_daily_vitals_info.GUPI==curr_GUPI)&(timestamped_daily_vitals_info.DVTimepoint!='None')] = fixed_date_vector
    
# Remove `DVTimepoint` from dataframe and apply categorizer
timestamped_daily_vitals_info = timestamped_daily_vitals_info.drop(columns='DVTimepoint').apply(categorizer)

## BestGCS information
# Filter from timestamped daily vitals information
best_gcs_names = daily_vitals_timestamped_variable_desc.name[daily_vitals_timestamped_variable_desc.varType=='BestGCS'].to_list()
best_gcs_info = timestamped_daily_vitals_info[['GUPI','DVDate']+best_gcs_names].dropna(subset=[n for n in best_gcs_names if n not in ['DVGCSBestTime']],how='all').reset_index(drop=True)
median_best_gcs = best_gcs_info.copy().dropna(subset=['GUPI','DVGCSBestTime'])
median_best_gcs['DVGCSBestTime'] = pd.to_datetime(median_best_gcs.DVGCSBestTime,format = '%H:%M:%S')
median_best_gcs = median_best_gcs.groupby(['GUPI'],as_index=False).DVGCSBestTime.aggregate('median')
overall_median_best_gcs = median_best_gcs.DVGCSBestTime.median().strftime('%H:%M:%S')
median_best_gcs['DVGCSBestTime'] = median_best_gcs['DVGCSBestTime'].dt.strftime('%H:%M:%S')
median_best_gcs = median_best_gcs.rename(columns={'DVGCSBestTime':'medianDVGCSBestTime'})

# Merge median Best GCS time to daily Best GCS dataframe
best_gcs_info = best_gcs_info.merge(median_best_gcs,how='left',on='GUPI')

# If daily Best GCS assessment time is missing, first impute with patient-specific median time
best_gcs_info.DVGCSBestTime[best_gcs_info.DVGCSBestTime.isna()&~best_gcs_info.medianDVGCSBestTime.isna()] = best_gcs_info.medianDVGCSBestTime[best_gcs_info.DVGCSBestTime.isna()&~best_gcs_info.medianDVGCSBestTime.isna()]

# If daily Best GCS assessment time is still missing, then impute with overall-set median time
best_gcs_info.DVGCSBestTime[best_gcs_info.DVGCSBestTime.isna()] = overall_median_best_gcs

# Construct daily Best GCS assessment timestamp
best_gcs_info['TimeStamp'] = best_gcs_info[['DVDate', 'DVGCSBestTime']].astype(str).agg(' '.join, axis=1)
best_gcs_info['TimeStamp'] = pd.to_datetime(best_gcs_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# If daily Best GCS Date matches the ICU discharge date, and the timestamp falls after discharge, then fix the timestamp to the discharge timestamp 
best_gcs_info = best_gcs_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
best_gcs_info.TimeStamp[(best_gcs_info.TimeStamp > best_gcs_info.ICUDischTimeStamp)&(best_gcs_info.DVDate == best_gcs_info.ICUDischTimeStamp.dt.date)] = best_gcs_info.ICUDischTimeStamp[(best_gcs_info.TimeStamp > best_gcs_info.ICUDischTimeStamp)&(best_gcs_info.DVDate == best_gcs_info.ICUDischTimeStamp.dt.date)]

# Remove time variables from dataframe and apply categorizer
best_gcs_info = best_gcs_info.drop(columns=['DVDate','DVGCSBestTime','medianDVGCSBestTime','ICUDischTimeStamp']).apply(categorizer)

# Fix `DVGCSEyes` variable mismatch
best_gcs_info.DVGCSEyes[best_gcs_info.DVGCSEyes.isin([4.0, 2.0, 1.0, 3.0])] = best_gcs_info.DVGCSEyes[best_gcs_info.DVGCSEyes.isin([4.0, 2.0, 1.0, 3.0])].astype(int).astype(str)

# Remove formatting from all categorical variable string values
categorical_best_gcs_names = [n for n in best_gcs_info if n not in ['GUPI','TimeStamp']]
categorical_best_gcs_predictors = best_gcs_info.dropna(axis=1,how='all').dropna(subset=categorical_best_gcs_names,how='all').reset_index(drop=True)
categorical_best_gcs_predictors[categorical_best_gcs_names] = categorical_best_gcs_predictors[categorical_best_gcs_names].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

# Construct tokens
categorical_best_gcs_predictors = categorical_best_gcs_predictors.melt(id_vars=['GUPI','TimeStamp']).drop_duplicates(subset=['GUPI','TimeStamp','value']).groupby(['GUPI','TimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## HighHR information
# Filter from timestamped daily vitals information
high_hr_names = daily_vitals_timestamped_variable_desc.name[daily_vitals_timestamped_variable_desc.varType=='HighHR'].to_list()
high_hr_info = timestamped_daily_vitals_info[['GUPI','DVDate']+high_hr_names].dropna(subset=[n for n in high_hr_names if n not in ['DVHRHighTime']],how='all').reset_index(drop=True)
median_high_hr = high_hr_info.copy().dropna(subset=['GUPI','DVHRHighTime'])
median_high_hr['DVHRHighTime'] = pd.to_datetime(median_high_hr.DVHRHighTime,format = '%H:%M:%S')
median_high_hr = median_high_hr.groupby(['GUPI'],as_index=False).DVHRHighTime.aggregate('median')
overall_median_high_hr = median_high_hr.DVHRHighTime.median().strftime('%H:%M:%S')
median_high_hr['DVHRHighTime'] = median_high_hr['DVHRHighTime'].dt.strftime('%H:%M:%S')
median_high_hr = median_high_hr.rename(columns={'DVHRHighTime':'medianDVHRHighTime'})

# Merge median HighHR time to daily HighHR dataframe
high_hr_info = high_hr_info.merge(median_high_hr,how='left',on='GUPI')

# If daily HighHR assessment time is missing, first impute with patient-specific median time
high_hr_info.DVHRHighTime[high_hr_info.DVHRHighTime.isna()&~high_hr_info.medianDVHRHighTime.isna()] = high_hr_info.medianDVHRHighTime[high_hr_info.DVHRHighTime.isna()&~high_hr_info.medianDVHRHighTime.isna()]

# If daily HighHR assessment time is still missing, then impute with overall-set median time
high_hr_info.DVHRHighTime[high_hr_info.DVHRHighTime.isna()] = overall_median_high_hr

# Construct daily HighHR assessment timestamp
high_hr_info['TimeStamp'] = high_hr_info[['DVDate', 'DVHRHighTime']].astype(str).agg(' '.join, axis=1)
high_hr_info['TimeStamp'] = pd.to_datetime(high_hr_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# If daily HighHR Date matches the ICU discharge date, and the timestamp falls after discharge, then fix the timestamp to the discharge timestamp 
high_hr_info = high_hr_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
high_hr_info.TimeStamp[(high_hr_info.TimeStamp > high_hr_info.ICUDischTimeStamp)&(high_hr_info.DVDate == high_hr_info.ICUDischTimeStamp.dt.date)] = high_hr_info.ICUDischTimeStamp[(high_hr_info.TimeStamp > high_hr_info.ICUDischTimeStamp)&(high_hr_info.DVDate == high_hr_info.ICUDischTimeStamp.dt.date)]

# Remove time variables from dataframe and apply categorizer
high_hr_info = high_hr_info.drop(columns=['DVDate','DVHRHighTime','medianDVHRHighTime','ICUDischTimeStamp']).apply(categorizer)

# Melt numeric dataframe into long form
numeric_high_hr_predictors = high_hr_info.dropna(axis=1,how='all').dropna(subset=[n for n in high_hr_names if n not in ['DVHRHighTime']],how='all').reset_index(drop=True).melt(id_vars=['GUPI','TimeStamp'],var_name='VARIABLE',value_name='VALUE')

## HighSBP information
# Filter from timestamped daily vitals information
high_sbp_names = daily_vitals_timestamped_variable_desc.name[daily_vitals_timestamped_variable_desc.varType=='HighSBP'].to_list()
high_sbp_info = timestamped_daily_vitals_info[['GUPI','DVDate']+high_sbp_names].dropna(subset=[n for n in high_sbp_names if n not in ['DVSBPHighTime']],how='all').reset_index(drop=True)
median_high_sbp = high_sbp_info.copy().dropna(subset=['GUPI','DVSBPHighTime'])
median_high_sbp['DVSBPHighTime'] = pd.to_datetime(median_high_sbp.DVSBPHighTime,format = '%H:%M:%S')
median_high_sbp = median_high_sbp.groupby(['GUPI'],as_index=False).DVSBPHighTime.aggregate('median')
overall_median_high_sbp = median_high_sbp.DVSBPHighTime.median().strftime('%H:%M:%S')
median_high_sbp['DVSBPHighTime'] = median_high_sbp['DVSBPHighTime'].dt.strftime('%H:%M:%S')
median_high_sbp = median_high_sbp.rename(columns={'DVSBPHighTime':'medianDVSBPHighTime'})

# Merge median HighSBP time to daily HighSBP dataframe
high_sbp_info = high_sbp_info.merge(median_high_sbp,how='left',on='GUPI')

# If daily HighSBP assessment time is missing, first impute with patient-specific median time
high_sbp_info.DVSBPHighTime[high_sbp_info.DVSBPHighTime.isna()&~high_sbp_info.medianDVSBPHighTime.isna()] = high_sbp_info.medianDVSBPHighTime[high_sbp_info.DVSBPHighTime.isna()&~high_sbp_info.medianDVSBPHighTime.isna()]

# If daily HighSBP assessment time is still missing, then impute with overall-set median time
high_sbp_info.DVSBPHighTime[high_sbp_info.DVSBPHighTime.isna()] = overall_median_high_sbp

# Construct daily HighSBP assessment timestamp
high_sbp_info['TimeStamp'] = high_sbp_info[['DVDate', 'DVSBPHighTime']].astype(str).agg(' '.join, axis=1)
high_sbp_info['TimeStamp'] = pd.to_datetime(high_sbp_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# If daily HighSBP Date matches the ICU discharge date, and the timestamp falls after discharge, then fix the timestamp to the discharge timestamp 
high_sbp_info = high_sbp_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
high_sbp_info.TimeStamp[(high_sbp_info.TimeStamp > high_sbp_info.ICUDischTimeStamp)&(high_sbp_info.DVDate == high_sbp_info.ICUDischTimeStamp.dt.date)] = high_sbp_info.ICUDischTimeStamp[(high_sbp_info.TimeStamp > high_sbp_info.ICUDischTimeStamp)&(high_sbp_info.DVDate == high_sbp_info.ICUDischTimeStamp.dt.date)]

# Remove time variables from dataframe and apply categorizer
high_sbp_info = high_sbp_info.drop(columns=['DVDate','DVSBPHighTime','medianDVSBPHighTime','ICUDischTimeStamp']).apply(categorizer)

# Melt numeric dataframe into long form
numeric_high_sbp_predictors = high_sbp_info.dropna(axis=1,how='all').dropna(subset=[n for n in high_sbp_names if n not in ['DVSBPHighTime']],how='all').reset_index(drop=True).melt(id_vars=['GUPI','TimeStamp'],var_name='VARIABLE',value_name='VALUE')

## HighSpO2 information
# Filter from timestamped daily vitals information
high_spo2_names = daily_vitals_timestamped_variable_desc.name[daily_vitals_timestamped_variable_desc.varType=='HighSpO2'].to_list()
high_spo2_info = timestamped_daily_vitals_info[['GUPI','DVDate']+high_spo2_names].dropna(subset=[n for n in high_spo2_names if n not in ['DVSpO2HighTime']],how='all').reset_index(drop=True)
median_high_spo2 = high_spo2_info.copy().dropna(subset=['GUPI','DVSpO2HighTime'])
median_high_spo2['DVSpO2HighTime'] = pd.to_datetime(median_high_spo2.DVSpO2HighTime,format = '%H:%M:%S')
median_high_spo2 = median_high_spo2.groupby(['GUPI'],as_index=False).DVSpO2HighTime.aggregate('median')
overall_median_high_spo2 = median_high_spo2.DVSpO2HighTime.median().strftime('%H:%M:%S')
median_high_spo2['DVSpO2HighTime'] = median_high_spo2['DVSpO2HighTime'].dt.strftime('%H:%M:%S')
median_high_spo2 = median_high_spo2.rename(columns={'DVSpO2HighTime':'medianDVSpO2HighTime'})

# Merge median HighSpO2 time to daily HighSpO2 dataframe
high_spo2_info = high_spo2_info.merge(median_high_spo2,how='left',on='GUPI')

# If daily HighSpO2 assessment time is missing, first impute with patient-specific median time
high_spo2_info.DVSpO2HighTime[high_spo2_info.DVSpO2HighTime.isna()&~high_spo2_info.medianDVSpO2HighTime.isna()] = high_spo2_info.medianDVSpO2HighTime[high_spo2_info.DVSpO2HighTime.isna()&~high_spo2_info.medianDVSpO2HighTime.isna()]

# If daily HighSpO2 assessment time is still missing, then impute with overall-set median time
high_spo2_info.DVSpO2HighTime[high_spo2_info.DVSpO2HighTime.isna()] = overall_median_high_spo2

# Construct daily HighSpO2 assessment timestamp
high_spo2_info['TimeStamp'] = high_spo2_info[['DVDate', 'DVSpO2HighTime']].astype(str).agg(' '.join, axis=1)
high_spo2_info['TimeStamp'] = pd.to_datetime(high_spo2_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# If daily HighSpO2 Date matches the ICU discharge date, and the timestamp falls after discharge, then fix the timestamp to the discharge timestamp 
high_spo2_info = high_spo2_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
high_spo2_info.TimeStamp[(high_spo2_info.TimeStamp > high_spo2_info.ICUDischTimeStamp)&(high_spo2_info.DVDate == high_spo2_info.ICUDischTimeStamp.dt.date)] = high_spo2_info.ICUDischTimeStamp[(high_spo2_info.TimeStamp > high_spo2_info.ICUDischTimeStamp)&(high_spo2_info.DVDate == high_spo2_info.ICUDischTimeStamp.dt.date)]

# Remove time variables from dataframe and apply categorizer
high_spo2_info = high_spo2_info.drop(columns=['DVDate','DVSpO2HighTime','medianDVSpO2HighTime','ICUDischTimeStamp']).apply(categorizer)

# Melt numeric dataframe into long form
numeric_high_spo2_predictors = high_spo2_info.dropna(axis=1,how='all').dropna(subset=[n for n in high_spo2_names if n not in ['DVSpO2HighTime']],how='all').reset_index(drop=True).melt(id_vars=['GUPI','TimeStamp'],var_name='VARIABLE',value_name='VALUE')

## HighTemp information
# Filter from timestamped daily vitals information
high_temp_names = daily_vitals_timestamped_variable_desc.name[daily_vitals_timestamped_variable_desc.varType=='HighTemp'].to_list()
high_temp_info = timestamped_daily_vitals_info[['GUPI','DVDate']+high_temp_names].dropna(subset=[n for n in high_temp_names if n not in ['DVTempHighTime']],how='all').reset_index(drop=True)
median_high_temp = high_temp_info.copy().dropna(subset=['GUPI','DVTempHighTime'])
median_high_temp['DVTempHighTime'] = pd.to_datetime(median_high_temp.DVTempHighTime,format = '%H:%M:%S')
median_high_temp = median_high_temp.groupby(['GUPI'],as_index=False).DVTempHighTime.aggregate('median')
overall_median_high_temp = median_high_temp.DVTempHighTime.median().strftime('%H:%M:%S')
median_high_temp['DVTempHighTime'] = median_high_temp['DVTempHighTime'].dt.strftime('%H:%M:%S')
median_high_temp = median_high_temp.rename(columns={'DVTempHighTime':'medianDVTempHighTime'})

# Merge median HighTemp time to daily HighTemp dataframe
high_temp_info = high_temp_info.merge(median_high_temp,how='left',on='GUPI')

# If daily HighTemp assessment time is missing, first impute with patient-specific median time
high_temp_info = high_temp_info.convert_dtypes()
high_temp_info.DVTempHighTime[(high_temp_info.DVTempHighTime.isna())&(~high_temp_info.medianDVTempHighTime.isna())] = high_temp_info.medianDVTempHighTime[(high_temp_info.DVTempHighTime.isna())&(~high_temp_info.medianDVTempHighTime.isna())]

# If daily HighTemp assessment time is still missing, then impute with overall-set median time
high_temp_info.DVTempHighTime[high_temp_info.DVTempHighTime.isna()] = overall_median_high_temp

# Construct daily HighTemp assessment timestamp
high_temp_info['TimeStamp'] = high_temp_info[['DVDate', 'DVTempHighTime']].astype(str).agg(' '.join, axis=1)
high_temp_info['TimeStamp'] = pd.to_datetime(high_temp_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# If daily HighTemp Date matches the ICU discharge date, and the timestamp falls after discharge, then fix the timestamp to the discharge timestamp 
high_temp_info = high_temp_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
high_temp_info.TimeStamp[(high_temp_info.TimeStamp > high_temp_info.ICUDischTimeStamp)&(high_temp_info.DVDate == high_temp_info.ICUDischTimeStamp.dt.date)] = high_temp_info.ICUDischTimeStamp[(high_temp_info.TimeStamp > high_temp_info.ICUDischTimeStamp)&(high_temp_info.DVDate == high_temp_info.ICUDischTimeStamp.dt.date)]

# Remove time variables from dataframe and apply categorizer
high_temp_info = high_temp_info.drop(columns=['DVDate','DVTempHighTime','medianDVTempHighTime','ICUDischTimeStamp']).apply(categorizer)

# Separate categorical and numeric variables into separate dataframes
numeric_high_temp_names = np.sort(high_temp_info.select_dtypes(include=['number']).columns.values)
categorical_high_temp_names = np.sort(high_temp_info.select_dtypes(exclude=['number']).drop(columns=['GUPI','TimeStamp']).columns.values)

# Melt numeric dataframe into long form
numeric_high_temp_predictors = high_temp_info[np.insert(numeric_high_temp_names,0,['GUPI','TimeStamp'])].dropna(axis=1,how='all').dropna(subset=numeric_high_temp_names,how='all').reset_index(drop=True).melt(id_vars=['GUPI','TimeStamp'],var_name='VARIABLE',value_name='VALUE')

# Remove formatting from all categorical variable string values
categorical_high_temp_predictors = high_temp_info[np.insert(categorical_high_temp_names,0,['GUPI','TimeStamp'])].dropna(axis=1,how='all').dropna(subset=categorical_high_temp_names,how='all').reset_index(drop=True)
categorical_high_temp_predictors[categorical_high_temp_names] = categorical_high_temp_predictors[categorical_high_temp_names].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

# Construct tokens
categorical_high_temp_predictors = categorical_high_temp_predictors.melt(id_vars=['GUPI','TimeStamp']).drop_duplicates(subset=['GUPI','TimeStamp','value']).groupby(['GUPI','TimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## LowHR information
# Filter from timestamped daily vitals information
low_hr_names = daily_vitals_timestamped_variable_desc.name[daily_vitals_timestamped_variable_desc.varType=='LowHR'].to_list()
low_hr_info = timestamped_daily_vitals_info[['GUPI','DVDate']+low_hr_names].dropna(subset=[n for n in low_hr_names if n not in ['DVHRLowTime']],how='all').reset_index(drop=True)
median_low_hr = low_hr_info.copy().dropna(subset=['GUPI','DVHRLowTime'])
median_low_hr['DVHRLowTime'] = pd.to_datetime(median_low_hr.DVHRLowTime,format = '%H:%M:%S')
median_low_hr = median_low_hr.groupby(['GUPI'],as_index=False).DVHRLowTime.aggregate('median')
overall_median_low_hr = median_low_hr.DVHRLowTime.median().strftime('%H:%M:%S')
median_low_hr['DVHRLowTime'] = median_low_hr['DVHRLowTime'].dt.strftime('%H:%M:%S')
median_low_hr = median_low_hr.rename(columns={'DVHRLowTime':'medianDVHRLowTime'})

# Merge median LowHR time to daily LowHR dataframe
low_hr_info = low_hr_info.merge(median_low_hr,how='left',on='GUPI')

# If daily LowHR assessment time is missing, first impute with patient-specific median time
low_hr_info.DVHRLowTime[low_hr_info.DVHRLowTime.isna()&~low_hr_info.medianDVHRLowTime.isna()] = low_hr_info.medianDVHRLowTime[low_hr_info.DVHRLowTime.isna()&~low_hr_info.medianDVHRLowTime.isna()]

# If daily LowHR assessment time is still missing, then impute with overall-set median time
low_hr_info.DVHRLowTime[low_hr_info.DVHRLowTime.isna()] = overall_median_low_hr

# Construct daily LowHR assessment timestamp
low_hr_info['TimeStamp'] = low_hr_info[['DVDate', 'DVHRLowTime']].astype(str).agg(' '.join, axis=1)
low_hr_info['TimeStamp'] = pd.to_datetime(low_hr_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# If daily LowHR Date matches the ICU discharge date, and the timestamp falls after discharge, then fix the timestamp to the discharge timestamp 
low_hr_info = low_hr_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
low_hr_info.TimeStamp[(low_hr_info.TimeStamp > low_hr_info.ICUDischTimeStamp)&(low_hr_info.DVDate == low_hr_info.ICUDischTimeStamp.dt.date)] = low_hr_info.ICUDischTimeStamp[(low_hr_info.TimeStamp > low_hr_info.ICUDischTimeStamp)&(low_hr_info.DVDate == low_hr_info.ICUDischTimeStamp.dt.date)]

# Remove time variables from dataframe and apply categorizer
low_hr_info = low_hr_info.drop(columns=['DVDate','DVHRLowTime','medianDVHRLowTime','ICUDischTimeStamp']).apply(categorizer)

# Melt numeric dataframe into long form
numeric_low_hr_predictors = low_hr_info.dropna(axis=1,how='all').dropna(subset=[n for n in low_hr_names if n not in ['DVHRLowTime']],how='all').reset_index(drop=True).melt(id_vars=['GUPI','TimeStamp'],var_name='VARIABLE',value_name='VALUE')

## LowSBP information
# Filter from timestamped daily vitals information
low_sbp_names = daily_vitals_timestamped_variable_desc.name[daily_vitals_timestamped_variable_desc.varType=='LowSBP'].to_list()
low_sbp_info = timestamped_daily_vitals_info[['GUPI','DVDate']+low_sbp_names].dropna(subset=[n for n in low_sbp_names if n not in ['DVSBPLowTime']],how='all').reset_index(drop=True)
median_low_sbp = low_sbp_info.copy().dropna(subset=['GUPI','DVSBPLowTime'])
median_low_sbp['DVSBPLowTime'] = pd.to_datetime(median_low_sbp.DVSBPLowTime,format = '%H:%M:%S')
median_low_sbp = median_low_sbp.groupby(['GUPI'],as_index=False).DVSBPLowTime.aggregate('median')
overall_median_low_sbp = median_low_sbp.DVSBPLowTime.median().strftime('%H:%M:%S')
median_low_sbp['DVSBPLowTime'] = median_low_sbp['DVSBPLowTime'].dt.strftime('%H:%M:%S')
median_low_sbp = median_low_sbp.rename(columns={'DVSBPLowTime':'medianDVSBPLowTime'})

# Merge median LowSBP time to daily LowSBP dataframe
low_sbp_info = low_sbp_info.merge(median_low_sbp,how='left',on='GUPI')

# If daily LowSBP assessment time is missing, first impute with patient-specific median time
low_sbp_info.DVSBPLowTime[low_sbp_info.DVSBPLowTime.isna()&~low_sbp_info.medianDVSBPLowTime.isna()] = low_sbp_info.medianDVSBPLowTime[low_sbp_info.DVSBPLowTime.isna()&~low_sbp_info.medianDVSBPLowTime.isna()]

# If daily LowSBP assessment time is still missing, then impute with overall-set median time
low_sbp_info.DVSBPLowTime[low_sbp_info.DVSBPLowTime.isna()] = overall_median_low_sbp

# Construct daily LowSBP assessment timestamp
low_sbp_info['TimeStamp'] = low_sbp_info[['DVDate', 'DVSBPLowTime']].astype(str).agg(' '.join, axis=1)
low_sbp_info['TimeStamp'] = pd.to_datetime(low_sbp_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# If daily LowSBP Date matches the ICU discharge date, and the timestamp falls after discharge, then fix the timestamp to the discharge timestamp 
low_sbp_info = low_sbp_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
low_sbp_info.TimeStamp[(low_sbp_info.TimeStamp > low_sbp_info.ICUDischTimeStamp)&(low_sbp_info.DVDate == low_sbp_info.ICUDischTimeStamp.dt.date)] = low_sbp_info.ICUDischTimeStamp[(low_sbp_info.TimeStamp > low_sbp_info.ICUDischTimeStamp)&(low_sbp_info.DVDate == low_sbp_info.ICUDischTimeStamp.dt.date)]

# Remove time variables from dataframe and apply categorizer
low_sbp_info = low_sbp_info.drop(columns=['DVDate','DVSBPLowTime','medianDVSBPLowTime','ICUDischTimeStamp']).apply(categorizer)

# Melt numeric dataframe into long form
numeric_low_sbp_predictors = low_sbp_info.dropna(axis=1,how='all').dropna(subset=[n for n in low_sbp_names if n not in ['DVSBPLowTime']],how='all').reset_index(drop=True).melt(id_vars=['GUPI','TimeStamp'],var_name='VARIABLE',value_name='VALUE')

## LowSpO2 information
# Filter from timestamped daily vitals information
low_spo2_names = daily_vitals_timestamped_variable_desc.name[daily_vitals_timestamped_variable_desc.varType=='LowSpO2'].to_list()
low_spo2_info = timestamped_daily_vitals_info[['GUPI','DVDate']+low_spo2_names].dropna(subset=[n for n in low_spo2_names if n not in ['DVSpO2LowTime']],how='all').reset_index(drop=True)
median_low_spo2 = low_spo2_info.copy().dropna(subset=['GUPI','DVSpO2LowTime'])
median_low_spo2['DVSpO2LowTime'] = pd.to_datetime(median_low_spo2.DVSpO2LowTime,format = '%H:%M:%S')
median_low_spo2 = median_low_spo2.groupby(['GUPI'],as_index=False).DVSpO2LowTime.aggregate('median')
overall_median_low_spo2 = median_low_spo2.DVSpO2LowTime.median().strftime('%H:%M:%S')
median_low_spo2['DVSpO2LowTime'] = median_low_spo2['DVSpO2LowTime'].dt.strftime('%H:%M:%S')
median_low_spo2 = median_low_spo2.rename(columns={'DVSpO2LowTime':'medianDVSpO2LowTime'})

# Merge median LowSpO2 time to daily LowSpO2 dataframe
low_spo2_info = low_spo2_info.merge(median_low_spo2,how='left',on='GUPI')

# If daily LowSpO2 assessment time is missing, first impute with patient-specific median time
low_spo2_info.DVSpO2LowTime[low_spo2_info.DVSpO2LowTime.isna()&~low_spo2_info.medianDVSpO2LowTime.isna()] = low_spo2_info.medianDVSpO2LowTime[low_spo2_info.DVSpO2LowTime.isna()&~low_spo2_info.medianDVSpO2LowTime.isna()]

# If daily LowSpO2 assessment time is still missing, then impute with overall-set median time
low_spo2_info.DVSpO2LowTime[low_spo2_info.DVSpO2LowTime.isna()] = overall_median_low_spo2

# Construct daily LowSpO2 assessment timestamp
low_spo2_info['TimeStamp'] = low_spo2_info[['DVDate', 'DVSpO2LowTime']].astype(str).agg(' '.join, axis=1)
low_spo2_info['TimeStamp'] = pd.to_datetime(low_spo2_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# If daily LowSpO2 Date matches the ICU discharge date, and the timestamp falls after discharge, then fix the timestamp to the discharge timestamp 
low_spo2_info = low_spo2_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
low_spo2_info.TimeStamp[(low_spo2_info.TimeStamp > low_spo2_info.ICUDischTimeStamp)&(low_spo2_info.DVDate == low_spo2_info.ICUDischTimeStamp.dt.date)] = low_spo2_info.ICUDischTimeStamp[(low_spo2_info.TimeStamp > low_spo2_info.ICUDischTimeStamp)&(low_spo2_info.DVDate == low_spo2_info.ICUDischTimeStamp.dt.date)]

# Remove time variables from dataframe and apply categorizer
low_spo2_info = low_spo2_info.drop(columns=['DVDate','DVSpO2LowTime','medianDVSpO2LowTime','ICUDischTimeStamp']).apply(categorizer)

# Melt numeric dataframe into long form
numeric_low_spo2_predictors = low_spo2_info.dropna(axis=1,how='all').dropna(subset=[n for n in low_spo2_names if n not in ['DVSpO2LowTime']],how='all').reset_index(drop=True).melt(id_vars=['GUPI','TimeStamp'],var_name='VARIABLE',value_name='VALUE')

## LowTemp information
# Filter from timestamped daily vitals information
low_temp_names = daily_vitals_timestamped_variable_desc.name[daily_vitals_timestamped_variable_desc.varType=='LowTemp'].to_list()
low_temp_info = timestamped_daily_vitals_info[['GUPI','DVDate']+low_temp_names].dropna(subset=[n for n in low_temp_names if n not in ['DVTempLowTime']],how='all').reset_index(drop=True)
median_low_temp = low_temp_info.copy().dropna(subset=['GUPI','DVTempLowTime'])
median_low_temp['DVTempLowTime'] = pd.to_datetime(median_low_temp.DVTempLowTime,format = '%H:%M:%S')
median_low_temp = median_low_temp.groupby(['GUPI'],as_index=False).DVTempLowTime.aggregate('median')
overall_median_low_temp = median_low_temp.DVTempLowTime.median().strftime('%H:%M:%S')
median_low_temp['DVTempLowTime'] = median_low_temp['DVTempLowTime'].dt.strftime('%H:%M:%S')
median_low_temp = median_low_temp.rename(columns={'DVTempLowTime':'medianDVTempLowTime'})

# Merge median LowTemp time to daily LowTemp dataframe
low_temp_info = low_temp_info.merge(median_low_temp,how='left',on='GUPI')

# If daily LowTemp assessment time is missing, first impute with patient-specific median time
low_temp_info = low_temp_info.convert_dtypes()
low_temp_info.DVTempLowTime[(low_temp_info.DVTempLowTime.isna())&(~low_temp_info.medianDVTempLowTime.isna())] = low_temp_info.medianDVTempLowTime[(low_temp_info.DVTempLowTime.isna())&(~low_temp_info.medianDVTempLowTime.isna())]

# If daily LowTemp assessment time is still missing, then impute with overall-set median time
low_temp_info.DVTempLowTime[low_temp_info.DVTempLowTime.isna()] = overall_median_low_temp

# Construct daily LowTemp assessment timestamp
low_temp_info['TimeStamp'] = low_temp_info[['DVDate', 'DVTempLowTime']].astype(str).agg(' '.join, axis=1)
low_temp_info['TimeStamp'] = pd.to_datetime(low_temp_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# If daily LowTemp Date matches the ICU discharge date, and the timestamp falls after discharge, then fix the timestamp to the discharge timestamp 
low_temp_info = low_temp_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
low_temp_info.TimeStamp[(low_temp_info.TimeStamp > low_temp_info.ICUDischTimeStamp)&(low_temp_info.DVDate == low_temp_info.ICUDischTimeStamp.dt.date)] = low_temp_info.ICUDischTimeStamp[(low_temp_info.TimeStamp > low_temp_info.ICUDischTimeStamp)&(low_temp_info.DVDate == low_temp_info.ICUDischTimeStamp.dt.date)]

# Remove time variables from dataframe and apply categorizer
low_temp_info = low_temp_info.drop(columns=['DVDate','DVTempLowTime','medianDVTempLowTime','ICUDischTimeStamp']).apply(categorizer)

# Separate categorical and numeric variables into separate dataframes
numeric_low_temp_names = np.sort(low_temp_info.select_dtypes(include=['number']).columns.values)
categorical_low_temp_names = np.sort(low_temp_info.select_dtypes(exclude=['number']).drop(columns=['GUPI','TimeStamp']).columns.values)

# Melt numeric dataframe into long form
numeric_low_temp_predictors = low_temp_info[np.insert(numeric_low_temp_names,0,['GUPI','TimeStamp'])].dropna(axis=1,how='all').dropna(subset=numeric_low_temp_names,how='all').reset_index(drop=True).melt(id_vars=['GUPI','TimeStamp'],var_name='VARIABLE',value_name='VALUE')

# Remove formatting from all categorical variable string values
categorical_low_temp_predictors = low_temp_info[np.insert(categorical_low_temp_names,0,['GUPI','TimeStamp'])].dropna(axis=1,how='all').dropna(subset=categorical_low_temp_names,how='all').reset_index(drop=True)
categorical_low_temp_predictors[categorical_low_temp_names] = categorical_low_temp_predictors[categorical_low_temp_names].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

# Construct tokens
categorical_low_temp_predictors = categorical_low_temp_predictors.melt(id_vars=['GUPI','TimeStamp']).drop_duplicates(subset=['GUPI','TimeStamp','value']).groupby(['GUPI','TimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## WorstGCS information
# Filter from timestamped daily vitals information
worst_gcs_names = daily_vitals_timestamped_variable_desc.name[daily_vitals_timestamped_variable_desc.varType=='WorstGCS'].to_list()
worst_gcs_info = timestamped_daily_vitals_info[['GUPI','DVDate']+worst_gcs_names].dropna(subset=[n for n in worst_gcs_names if n not in ['DailyGCSTime']],how='all').reset_index(drop=True)
median_worst_gcs = worst_gcs_info.copy().dropna(subset=['GUPI','DailyGCSTime'])
median_worst_gcs['DailyGCSTime'] = pd.to_datetime(median_worst_gcs.DailyGCSTime,format = '%H:%M:%S')
median_worst_gcs = median_worst_gcs.groupby(['GUPI'],as_index=False).DailyGCSTime.aggregate('median')
overall_median_worst_gcs = median_worst_gcs.DailyGCSTime.median().strftime('%H:%M:%S')
median_worst_gcs['DailyGCSTime'] = median_worst_gcs['DailyGCSTime'].dt.strftime('%H:%M:%S')
median_worst_gcs = median_worst_gcs.rename(columns={'DailyGCSTime':'medianDailyGCSTime'})

# Merge median WorstGCS time to daily WorstGCS dataframe
worst_gcs_info = worst_gcs_info.merge(median_worst_gcs,how='left',on='GUPI')

# If daily WorstGCS assessment time is missing, first impute with patient-specific median time
worst_gcs_info.DailyGCSTime[worst_gcs_info.DailyGCSTime.isna()&~worst_gcs_info.medianDailyGCSTime.isna()] = worst_gcs_info.medianDailyGCSTime[worst_gcs_info.DailyGCSTime.isna()&~worst_gcs_info.medianDailyGCSTime.isna()]

# If daily WorstGCS assessment time is still missing, then impute with overall-set median time
worst_gcs_info.DailyGCSTime[worst_gcs_info.DailyGCSTime.isna()] = overall_median_worst_gcs

# Construct daily WorstGCS assessment timestamp
worst_gcs_info['TimeStamp'] = worst_gcs_info[['DVDate', 'DailyGCSTime']].astype(str).agg(' '.join, axis=1)
worst_gcs_info['TimeStamp'] = pd.to_datetime(worst_gcs_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# If daily WorstGCS Date matches the ICU discharge date, and the timestamp falls after discharge, then fix the timestamp to the discharge timestamp 
worst_gcs_info = worst_gcs_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
worst_gcs_info.TimeStamp[(worst_gcs_info.TimeStamp > worst_gcs_info.ICUDischTimeStamp)&(worst_gcs_info.DVDate == worst_gcs_info.ICUDischTimeStamp.dt.date)] = worst_gcs_info.ICUDischTimeStamp[(worst_gcs_info.TimeStamp > worst_gcs_info.ICUDischTimeStamp)&(worst_gcs_info.DVDate == worst_gcs_info.ICUDischTimeStamp.dt.date)]

# Remove time variables from dataframe and apply categorizer
worst_gcs_info = worst_gcs_info.drop(columns=['DVDate','DailyGCSTime','medianDailyGCSTime','ICUDischTimeStamp']).apply(categorizer)

# Fix `DVGCSWorstEyes` variable mismatch
worst_gcs_info.DVGCSWorstEyes[worst_gcs_info.DVGCSWorstEyes.isin([4.0, 2.0, 1.0, 3.0])] = worst_gcs_info.DVGCSWorstEyes[worst_gcs_info.DVGCSWorstEyes.isin([4.0, 2.0, 1.0, 3.0])].astype(int).astype(str)

# Remove formatting from all categorical variable string values
categorical_worst_gcs_names = [n for n in worst_gcs_info if n not in ['GUPI','TimeStamp']]
categorical_worst_gcs_predictors = worst_gcs_info.dropna(axis=1,how='all').dropna(subset=categorical_worst_gcs_names,how='all').reset_index(drop=True)
categorical_worst_gcs_predictors[categorical_worst_gcs_names] = categorical_worst_gcs_predictors[categorical_worst_gcs_names].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

# Construct tokens
categorical_worst_gcs_predictors = categorical_worst_gcs_predictors.melt(id_vars=['GUPI','TimeStamp']).drop_duplicates(subset=['GUPI','TimeStamp','value']).groupby(['GUPI','TimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Daily therapy intensity level information
# Load DailyTIL data
daily_TIL_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/DailyTIL/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
daily_TIL_info = daily_TIL_info[daily_TIL_info.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)
daily_TIL_names = list(daily_TIL_info.columns)

# Load inspection table and gather names of daily TIL variables
daily_TIL_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/DailyTIL/inspection_table.xlsx',sheet_name='variables',na_values = ["NA","NaN"," ", ""])
daily_TIL_names = daily_TIL_desc.name[daily_TIL_desc.name.isin(daily_TIL_names)].to_list()
daily_TIL_info = daily_TIL_info[['GUPI']+daily_TIL_names].dropna(subset=[n for n in daily_TIL_names if n not in ['TILTimepoint','TILDate','TILTime']],how='all').reset_index(drop=True)

# Remove all entries without date or `TILTimepoint`
daily_TIL_info = daily_TIL_info[(daily_TIL_info.TILTimepoint!='None')|(~daily_TIL_info.TILDate.isna())].reset_index(drop=True)

# Convert dates from string to date format
daily_TIL_info.TILDate = pd.to_datetime(daily_TIL_info.TILDate,format = '%Y-%m-%d')

# For each patient, and for the overall set, calculate median TIL evaluation time
median_TILTime = daily_TIL_info.copy().dropna(subset=['GUPI','TILTime'])
median_TILTime['TILTime'] = pd.to_datetime(median_TILTime.TILTime,format = '%H:%M:%S')
median_TILTime = median_TILTime.groupby(['GUPI'],as_index=False).TILTime.aggregate('median')
overall_median_TILTime = median_TILTime.TILTime.median().strftime('%H:%M:%S')
median_TILTime['TILTime'] = median_TILTime['TILTime'].dt.strftime('%H:%M:%S')
median_TILTime = median_TILTime.rename(columns={'TILTime':'medianTILTime'})

# Iterate through GUPIs and fix `TILDate` based on `TILTimepoint` information if possible
problem_GUPIs = []
for curr_GUPI in tqdm(daily_TIL_info.GUPI.unique(),'Fixing daily TIL dates if possible'):
    curr_GUPI_daily_TIL = daily_TIL_info[(daily_TIL_info.GUPI==curr_GUPI)&(daily_TIL_info.TILTimepoint!='None')].reset_index(drop=True)
    if curr_GUPI_daily_TIL.TILDate.isna().all():
        print('Problem GUPI: '+curr_GUPI)
        problem_GUPIs.append(curr_GUPI)
        continue
    curr_date_diff = int((curr_GUPI_daily_TIL.TILDate.dt.day - curr_GUPI_daily_TIL.TILTimepoint.astype(float)).mode()[0])
    fixed_date_vector = pd.Series([pd.Timestamp('1970-01-01') + pd.DateOffset(days=dt+curr_date_diff) for dt in (curr_GUPI_daily_TIL.TILTimepoint.astype(float)-1)],index=daily_TIL_info[(daily_TIL_info.GUPI==curr_GUPI)&(daily_TIL_info.TILTimepoint!='None')].index)
    daily_TIL_info.TILDate[(daily_TIL_info.GUPI==curr_GUPI)&(daily_TIL_info.TILTimepoint!='None')] = fixed_date_vector    

# For problem_GUPIs, for whom the TILDates are still missing, find a patient with the closest ICU admission time, and employ their date difference
non_problem_set = CENTER_TBI_ICU_datetime[(~CENTER_TBI_ICU_datetime.GUPI.isin(problem_GUPIs))&(CENTER_TBI_ICU_datetime.GUPI.isin(daily_TIL_info.GUPI))].reset_index(drop=True)
for curr_GUPI in tqdm(problem_GUPIs, 'Fixing problem TIL dates'):
    
    # Extract current ICU admission timestamp
    curr_ICUAdmTimeStamp = CENTER_TBI_ICU_datetime.ICUAdmTimeStamp[CENTER_TBI_ICU_datetime.GUPI == curr_GUPI].values[0]
    
    # Find a non-problem-GUPI patient with the closest ICU admission time
    closest_GUPI = non_problem_set.GUPI[(non_problem_set.ICUAdmTimeStamp - curr_ICUAdmTimeStamp).dt.total_seconds().abs().argmin()]
    
    # Calculate date difference on closest GUPI
    closest_GUPI_daily_TIL = daily_TIL_info[(daily_TIL_info.GUPI==closest_GUPI)&(daily_TIL_info.TILTimepoint!='None')].reset_index(drop=True)
    curr_date_diff = int((closest_GUPI_daily_TIL.TILDate.dt.day - closest_GUPI_daily_TIL.TILTimepoint.astype(float)).mode()[0])
    
    # Calulcate fixed date vector for current problem GUPI
    curr_GUPI_daily_TIL = daily_TIL_info[(daily_TIL_info.GUPI==curr_GUPI)&(daily_TIL_info.TILTimepoint!='None')].reset_index(drop=True)
    fixed_date_vector = pd.Series([pd.Timestamp('1970-01-01') + pd.DateOffset(days=dt+curr_date_diff) for dt in (curr_GUPI_daily_TIL.TILTimepoint.astype(float)-1)],index=daily_TIL_info[(daily_TIL_info.GUPI==curr_GUPI)&(daily_TIL_info.TILTimepoint!='None')].index)

    # Fix problem GUPI dates in the original dataframe
    daily_TIL_info.TILDate[(daily_TIL_info.GUPI==curr_GUPI)&(daily_TIL_info.TILTimepoint!='None')] = fixed_date_vector    
    
# Merge median TIL time to daily TIL dataframe
daily_TIL_info = daily_TIL_info.merge(median_TILTime,how='left',on='GUPI')

# If daily TIL assessment time is missing, first impute with patient-specific median time
daily_TIL_info.TILTime[daily_TIL_info.TILTime.isna()&~daily_TIL_info.medianTILTime.isna()] = daily_TIL_info.medianTILTime[daily_TIL_info.TILTime.isna()&~daily_TIL_info.medianTILTime.isna()]

# If daily TIL assessment time is still missing, then impute with overall-set median time
daily_TIL_info.TILTime[daily_TIL_info.TILTime.isna()] = overall_median_TILTime

# Construct daily TIL assessment timestamp
daily_TIL_info['TimeStamp'] = daily_TIL_info[['TILDate', 'TILTime']].astype(str).agg(' '.join, axis=1)
daily_TIL_info['TimeStamp'] = pd.to_datetime(daily_TIL_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# If daily TIL Date matches the ICU discharge date, and the timestamp falls after discharge, then fix the timestamp to the discharge timestamp 
daily_TIL_info = daily_TIL_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
daily_TIL_info.TimeStamp[(daily_TIL_info.TimeStamp > daily_TIL_info.ICUDischTimeStamp)&(daily_TIL_info.TILDate == daily_TIL_info.ICUDischTimeStamp.dt.date)] = daily_TIL_info.ICUDischTimeStamp[(daily_TIL_info.TimeStamp > daily_TIL_info.ICUDischTimeStamp)&(daily_TIL_info.TILDate == daily_TIL_info.ICUDischTimeStamp.dt.date)]

# Fix volume and dose variables if incorrectly casted as character types
fix_TIL_columns = [col for col, dt in daily_TIL_info.dtypes.items() if (col.endswith('Dose')|('Volume' in col))&(dt == object)]
daily_TIL_info[fix_TIL_columns] = daily_TIL_info[fix_TIL_columns].replace(to_replace='^\D*$', value=np.nan, regex=True)
daily_TIL_info[fix_TIL_columns] = daily_TIL_info[fix_TIL_columns].apply(lambda x: x.str.replace(',','.',regex=False))
daily_TIL_info[fix_TIL_columns] = daily_TIL_info[fix_TIL_columns].apply(lambda x: x.str.replace('[^0-9\\.]','',regex=True))
daily_TIL_info[fix_TIL_columns] = daily_TIL_info[fix_TIL_columns].apply(lambda x: x.str.replace('\\.\\.','.',regex=True))
daily_TIL_info[fix_TIL_columns] = daily_TIL_info[fix_TIL_columns].apply(pd.to_numeric)

# Remove time variables from dataframe and apply categorizer
daily_TIL_info = daily_TIL_info.drop(columns=['TILTimepoint','TILDate','TILTime','medianTILTime','ICUDischTimeStamp']).apply(categorizer)

# Separate categorical and numeric variables into separate dataframes
numeric_daily_TIL_names = np.sort(daily_TIL_info.select_dtypes(include=['number']).columns.values)
categorical_daily_TIL_names = np.sort(daily_TIL_info.select_dtypes(exclude=['number']).drop(columns=['GUPI','TimeStamp']).columns.values)

# Melt numeric dataframe into long form
numeric_daily_TIL_predictors = daily_TIL_info[np.insert(numeric_daily_TIL_names,0,['GUPI','TimeStamp'])].dropna(axis=1,how='all').dropna(subset=numeric_daily_TIL_names,how='all').reset_index(drop=True).melt(id_vars=['GUPI','TimeStamp'],var_name='VARIABLE',value_name='VALUE')

# Remove formatting from all categorical variable string values
categorical_daily_TIL_predictors = daily_TIL_info[np.insert(categorical_daily_TIL_names,0,['GUPI','TimeStamp'])].dropna(axis=1,how='all').dropna(subset=categorical_daily_TIL_names,how='all').reset_index(drop=True)
categorical_daily_TIL_predictors[categorical_daily_TIL_names] = categorical_daily_TIL_predictors[categorical_daily_TIL_names].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

# Construct tokens
categorical_daily_TIL_predictors = categorical_daily_TIL_predictors.melt(id_vars=['GUPI','TimeStamp']).drop_duplicates(subset=['GUPI','TimeStamp','value']).groupby(['GUPI','TimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Daily hourly information
# Load DailyHourly data
daily_hourly_info = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/DailyHourlyValues/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
daily_hourly_info = daily_hourly_info[daily_hourly_info.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)
daily_hourly_names = list(daily_hourly_info.columns)

# Load inspection table and gather names of daily hourly variables
daily_hourly_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/DailyHourlyValues/inspection_table.xlsx',sheet_name='variables',na_values = ["NA","NaN"," ", ""])
daily_hourly_names = daily_hourly_desc.name[daily_hourly_desc.name.isin(daily_hourly_names)].to_list()
daily_hourly_info = daily_hourly_info[['GUPI']+daily_hourly_names].dropna(subset=[n for n in daily_hourly_names if n not in ['HourlyValueTimePoint','HVDate','HVTime']],how='all').reset_index(drop=True)

# Remove all entries without date or `HourlyValueTimePoint`
daily_hourly_info = daily_hourly_info[(daily_hourly_info.HourlyValueTimePoint!='None')|(~daily_hourly_info.HVDate.isna())].reset_index(drop=True)

# Convert dates from string to date format
daily_hourly_info.HVDate = pd.to_datetime(daily_hourly_info.HVDate,format = '%Y-%m-%d')

# Iterate through GUPIs and fix `HVDate` based on `HourlyValueTimePoint` information if possible
problem_GUPIs = []
for curr_GUPI in tqdm(daily_hourly_info.GUPI.unique(),'Fixing daily hourly dates if possible'):
    curr_GUPI_daily_hourly = daily_hourly_info[(daily_hourly_info.GUPI==curr_GUPI)&(daily_hourly_info.HourlyValueTimePoint!='None')].reset_index(drop=True)
    if curr_GUPI_daily_hourly.HVDate.isna().all():
        print('Problem GUPI: '+curr_GUPI)
        problem_GUPIs.append(curr_GUPI)
        continue
    curr_date_diff = int((curr_GUPI_daily_hourly.HVDate.dt.day - curr_GUPI_daily_hourly.HourlyValueTimePoint.astype(float)).mode()[0])
    fixed_date_vector = pd.Series([pd.Timestamp('1970-01-01') + pd.DateOffset(days=dt+curr_date_diff) for dt in (curr_GUPI_daily_hourly.HourlyValueTimePoint.astype(float)-1)],index=daily_hourly_info[(daily_hourly_info.GUPI==curr_GUPI)&(daily_hourly_info.HourlyValueTimePoint!='None')].index)
    daily_hourly_info.HVDate[(daily_hourly_info.GUPI==curr_GUPI)&(daily_hourly_info.HourlyValueTimePoint!='None')] = fixed_date_vector    

# For problem_GUPIs, for whom the HVDates are still missing, find a patient with the closest ICU admission time, and employ their date difference
non_problem_set = CENTER_TBI_ICU_datetime[(~CENTER_TBI_ICU_datetime.GUPI.isin(problem_GUPIs))&(CENTER_TBI_ICU_datetime.GUPI.isin(daily_hourly_info.GUPI))].reset_index(drop=True)
for curr_GUPI in tqdm(problem_GUPIs, 'Fixing problem HV dates'):
    
    # Extract current ICU admission timestamp
    curr_ICUAdmTimeStamp = CENTER_TBI_ICU_datetime.ICUAdmTimeStamp[CENTER_TBI_ICU_datetime.GUPI == curr_GUPI].values[0]
    
    # Find a non-problem-GUPI patient with the closest ICU admission time
    closest_GUPI = non_problem_set.GUPI[(non_problem_set.ICUAdmTimeStamp - curr_ICUAdmTimeStamp).dt.total_seconds().abs().argmin()]
    
    # Calculate date difference on closest GUPI
    closest_GUPI_daily_hourly = daily_hourly_info[(daily_hourly_info.GUPI==closest_GUPI)&(daily_hourly_info.HourlyValueTimePoint!='None')].reset_index(drop=True)
    curr_date_diff = int((closest_GUPI_daily_hourly.HVDate.dt.day - closest_GUPI_daily_hourly.HourlyValueTimePoint.astype(float)).mode()[0])
    
    # Calulcate fixed date vector for current problem GUPI
    curr_GUPI_daily_hourly = daily_hourly_info[(daily_hourly_info.GUPI==curr_GUPI)&(daily_hourly_info.HourlyValueTimePoint!='None')].reset_index(drop=True)
    fixed_date_vector = pd.Series([pd.Timestamp('1970-01-01') + pd.DateOffset(days=dt+curr_date_diff) for dt in (curr_GUPI_daily_hourly.HourlyValueTimePoint.astype(float)-1)],index=daily_hourly_info[(daily_hourly_info.GUPI==curr_GUPI)&(daily_hourly_info.HourlyValueTimePoint!='None')].index)

    # Fix problem GUPI dates in the original dataframe
    daily_hourly_info.HVDate[(daily_hourly_info.GUPI==curr_GUPI)&(daily_hourly_info.HourlyValueTimePoint!='None')] = fixed_date_vector    

# If 'HVTime' is missing, replace with 'HVHour'
daily_hourly_info.HVTime[(daily_hourly_info.HVTime.isna())&(~daily_hourly_info.HVHour.isna())] = daily_hourly_info.HVHour[(daily_hourly_info.HVTime.isna())&(~daily_hourly_info.HVHour.isna())]

# Fix cases in which 'HVTime' equals '24:00:00'
daily_hourly_info.HVDate[daily_hourly_info.HVTime == '24:00:00'] = daily_hourly_info.HVDate[daily_hourly_info.HVTime == '24:00:00'] + timedelta(days=1)
daily_hourly_info.HVTime[daily_hourly_info.HVTime == '24:00:00'] = '00:00:00'

# Construct daily hourly assessment timestamp
daily_hourly_info['TimeStamp'] = daily_hourly_info[['HVDate', 'HVTime']].astype(str).agg(' '.join, axis=1)
daily_hourly_info['TimeStamp'] = pd.to_datetime(daily_hourly_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# If daily hourly Date matches the ICU discharge date, and the timestamp falls after discharge, then fix the timestamp to the discharge timestamp 
daily_hourly_info = daily_hourly_info.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
daily_hourly_info.TimeStamp[(daily_hourly_info.TimeStamp > daily_hourly_info.ICUDischTimeStamp)&(daily_hourly_info.HVDate == daily_hourly_info.ICUDischTimeStamp.dt.date)] = daily_hourly_info.ICUDischTimeStamp[(daily_hourly_info.TimeStamp > daily_hourly_info.ICUDischTimeStamp)&(daily_hourly_info.HVDate == daily_hourly_info.ICUDischTimeStamp.dt.date)]

# Remove time variables from dataframe and apply categorizer
daily_hourly_info = daily_hourly_info.drop(columns=['HourlyValueTimePoint','HVHour','HVDate','HVTime','ICUDischTimeStamp']).apply(categorizer)

# Separate categorical and numeric variables into separate dataframes
numeric_daily_hourly_names = np.sort(daily_hourly_info.select_dtypes(include=['number']).columns.values)
categorical_daily_hourly_names = np.sort(daily_hourly_info.select_dtypes(exclude=['number']).drop(columns=['GUPI','TimeStamp']).columns.values)

# Melt numeric dataframe into long form
numeric_daily_hourly_predictors = daily_hourly_info[np.insert(numeric_daily_hourly_names,0,['GUPI','TimeStamp'])].dropna(axis=1,how='all').dropna(subset=numeric_daily_hourly_names,how='all').reset_index(drop=True).melt(id_vars=['GUPI','TimeStamp'],var_name='VARIABLE',value_name='VALUE')

# Remove formatting from all categorical variable string values
categorical_daily_hourly_predictors = daily_hourly_info[np.insert(categorical_daily_hourly_names,0,['GUPI','TimeStamp'])].dropna(axis=1,how='all').dropna(subset=categorical_daily_hourly_names,how='all').reset_index(drop=True)
categorical_daily_hourly_predictors[categorical_daily_hourly_names] = categorical_daily_hourly_predictors[categorical_daily_hourly_names].apply(lambda x: x.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

# Construct tokens
categorical_daily_hourly_predictors = categorical_daily_hourly_predictors.melt(id_vars=['GUPI','TimeStamp']).drop_duplicates(subset=['GUPI','TimeStamp','value']).groupby(['GUPI','TimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Labs information
# Load labs data
labs = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/Labs/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
labs = labs[labs.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)
labs_names = list(labs.columns)

# Load inspection table and gather names of labs variables
labs_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/Labs/inspection_table.xlsx',sheet_name='variables',na_values = ["NA","NaN"," ", ""])
labs_names = labs_desc.name[labs_desc.name.isin(labs_names)].to_list()
labs = labs[['GUPI']+labs_names].dropna(subset=[n for n in labs_names if n not in ['DLReason','LabsLocation','DLDate','DLTime']],how='all').reset_index(drop=True)

# Remove all ER labs and labs without an associated date (if not in ICU), and drop unnecessary columns
labs = labs[(labs.LabsLocation != 'ER')&((~labs.DLDate.isna())|(labs.LabsLocation == 'ICU'))].reset_index(drop=True)

# If DLDate is missing, replace with date and time of ICU discharge
labs = labs.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischDate','ICUDischTime']],how='left',on='GUPI')
labs.DLTime[labs.DLDate.isna()] = labs.ICUDischTime[labs.DLDate.isna()]
labs.DLDate[labs.DLDate.isna()] = labs.ICUDischDate[labs.DLDate.isna()]
labs = labs.drop(columns=['ICUDischDate','ICUDischTime'])

# Construct labs timestamp
labs['TimeStamp'] = labs[['DLDate', 'DLTime']].astype(str).agg(' '.join, axis=1)
labs['TimeStamp'][labs.DLDate.isna() | labs.DLTime.isna()] = np.nan
labs['TimeStamp'] = pd.to_datetime(labs['TimeStamp'],format = '%Y-%m-%d %H:%M:%S')

# If timestamp is missing, replace with median lab time and apply categorizer
median_lab_time = pd.to_datetime(labs.DLTime,format = '%H:%M:%S').dropna().median().strftime('%H:%M:%S')
labs.TimeStamp[labs.TimeStamp.isna()] = pd.to_datetime(labs.DLDate[labs.TimeStamp.isna()] + ' ' + median_lab_time,format = '%Y-%m-%d %H:%M:%S')
labs = labs.drop(columns=['DLDate','DLTime']).apply(categorizer).dropna(axis=1,how='all')

# If daily labs Date matches the ICU discharge date, and the timestamp falls after discharge, then fix the timestamp to the discharge timestamp 
labs = labs.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
labs.TimeStamp[(labs.TimeStamp > labs.ICUDischTimeStamp)&(labs.TimeStamp.dt.date == labs.ICUDischTimeStamp.dt.date)&(labs.LabsLocation=='ICU')] = labs.ICUDischTimeStamp[(labs.TimeStamp > labs.ICUDischTimeStamp)&(labs.TimeStamp.dt.date == labs.ICUDischTimeStamp.dt.date)&(labs.LabsLocation=='ICU')]

# Remove LabsLocation variable from dataframe
labs = labs.drop(columns=['LabsLocation','ICUDischTimeStamp'])

# Inspect `object` type variables to check for hidden numerics
object_col_uniques = labs.select_dtypes(include=[object]).apply(lambda x: x.astype(str)).apply(lambda x: len(x.unique()))
object_cols_to_fix = object_col_uniques[(object_col_uniques > 20)&(object_col_uniques.index != 'GUPI')].index.to_list()
labs[object_cols_to_fix] = labs[object_cols_to_fix].apply(lambda x: pd.to_numeric(x, errors='coerce')).apply(categorizer).dropna(axis=1,how='all')

# Separate categorical and numeric variables into separate dataframes
numeric_labs_names = np.sort(labs.select_dtypes(include=['number']).columns.values)
categorical_labs_names = np.sort(labs.select_dtypes(exclude=['number']).drop(columns=['GUPI','TimeStamp']).columns.values)

# Melt numeric dataframe into long form
numeric_labs_predictors = labs[np.insert(numeric_labs_names,0,['GUPI','TimeStamp'])].dropna(axis=1,how='all').dropna(subset=numeric_labs_names,how='all').reset_index(drop=True).melt(id_vars=['GUPI','TimeStamp'],var_name='VARIABLE',value_name='VALUE')

# Remove formatting from all categorical variable string values
categorical_labs_predictors = labs[np.insert(categorical_labs_names,0,['GUPI','TimeStamp'])].dropna(axis=1,how='all').dropna(subset=categorical_labs_names,how='all').reset_index(drop=True)
categorical_labs_predictors[categorical_labs_names] = categorical_labs_predictors[categorical_labs_names].apply(lambda x: x.astype(str).str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

# Construct tokens
categorical_labs_predictors = categorical_labs_predictors.melt(id_vars=['GUPI','TimeStamp']).drop_duplicates(subset=['GUPI','TimeStamp','value']).groupby(['GUPI','TimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Biomarkers information
# Load Biomarkers data
biomarkers = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/Biomarkers/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
biomarkers = biomarkers[biomarkers.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)
biomarkers_names = list(biomarkers.columns)

# Load inspection table and gather names of biomarkers variables
biomarkers_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/Biomarkers/inspection_table.xlsx',sheet_name='variables',na_values = ["NA","NaN"," ", ""])
biomarkers_names = biomarkers_desc.name[biomarkers_desc.name.isin(biomarkers_names)].to_list()
biomarkers = biomarkers[['GUPI']+biomarkers_names].dropna(subset=[n for n in biomarkers_names if n not in ['CollectionDate','CollectionTime','CentrifugationDate','CentrifugationTime','FreezerMinusTwentyDate','FreezerMinusTwentyTime','FreezerMinusEightyDate','FreezerMinusEightyTime']],how='all').reset_index(drop=True)

# Remove Biomarker values without any date information
biomarkers = biomarkers[(~biomarkers.CollectionDate.isna())|(~(biomarkers.CentrifugationDate.isna()))|(~(biomarkers.FreezerMinusTwentyDate.isna()))|(~(biomarkers.FreezerMinusEightyDate.isna()))].reset_index(drop=True)

# Construct biomarkers timestamps
biomarkers['CollectionTimeStamp'] = biomarkers[['CollectionDate', 'CollectionTime']].astype(str).agg(' '.join, axis=1)
biomarkers['CollectionTimeStamp'][biomarkers.CollectionDate.isna() | biomarkers.CollectionTime.isna()] = np.nan
biomarkers['CollectionTimeStamp'] = pd.to_datetime(biomarkers['CollectionTimeStamp'],format = '%Y-%m-%d %H:%M:%S')

biomarkers['CentrifugationTimeStamp'] = biomarkers[['CentrifugationDate', 'CentrifugationTime']].astype(str).agg(' '.join, axis=1)
biomarkers['CentrifugationTimeStamp'][biomarkers.CentrifugationDate.isna() | biomarkers.CentrifugationTime.isna()] = np.nan
biomarkers['CentrifugationTimeStamp'] = pd.to_datetime(biomarkers['CentrifugationTimeStamp'],format = '%Y-%m-%d %H:%M:%S')

biomarkers['FreezerMinusTwentyTimeStamp'] = biomarkers[['FreezerMinusTwentyDate', 'FreezerMinusTwentyTime']].astype(str).agg(' '.join, axis=1)
biomarkers['FreezerMinusTwentyTimeStamp'][biomarkers.FreezerMinusTwentyDate.isna() | biomarkers.FreezerMinusTwentyTime.isna()] = np.nan
biomarkers['FreezerMinusTwentyTimeStamp'] = pd.to_datetime(biomarkers['FreezerMinusTwentyTimeStamp'],format = '%Y-%m-%d %H:%M:%S')

biomarkers['FreezerMinusEightyTimeStamp'] = biomarkers[['FreezerMinusEightyDate', 'FreezerMinusEightyTime']].astype(str).agg(' '.join, axis=1)
biomarkers['FreezerMinusEightyTimeStamp'][biomarkers.FreezerMinusEightyDate.isna() | biomarkers.FreezerMinusEightyTime.isna()] = np.nan
biomarkers['FreezerMinusEightyTimeStamp'] = pd.to_datetime(biomarkers['FreezerMinusEightyTimeStamp'],format = '%Y-%m-%d %H:%M:%S')

# If `CollectionTimeStamp` is missing, replace with median collection time
median_collect_time = pd.to_datetime(biomarkers.CollectionTime,format = '%H:%M:%S').dropna().median().strftime('%H:%M:%S')
biomarkers.CollectionTimeStamp[biomarkers.CollectionTimeStamp.isna()] = pd.to_datetime(biomarkers.CollectionDate[biomarkers.CollectionTimeStamp.isna()] + ' ' + median_collect_time,format = '%Y-%m-%d %H:%M:%S')

# If `CentrifugationTimeStamp` is missing, and Centrifugation date matches Collection date, replace with Collection timestamp
biomarkers.CentrifugationTimeStamp[biomarkers.CentrifugationTimeStamp.isna()&(biomarkers.CentrifugationDate==biomarkers.CollectionDate)] = biomarkers.CollectionTimeStamp[biomarkers.CentrifugationTimeStamp.isna()&(biomarkers.CentrifugationDate==biomarkers.CollectionDate)]

# Remove original Date and Time columns and apply categorizer
biomarkers = biomarkers.drop(columns=['CollectionDate','CollectionTime','CentrifugationDate','CentrifugationTime','FreezerMinusTwentyDate','FreezerMinusTwentyTime','FreezerMinusEightyDate','FreezerMinusEightyTime']).apply(categorizer).dropna(axis=1,how='all')

# Assign final timestamp variable based on hierarchy
biomarkers['TimeStamp'] = biomarkers['CentrifugationTimeStamp']
biomarkers.TimeStamp[biomarkers.TimeStamp.isna()] = biomarkers.CollectionTimeStamp[biomarkers.TimeStamp.isna()]
biomarkers.TimeStamp[biomarkers.TimeStamp.isna()] = biomarkers.FreezerMinusTwentyTimeStamp[biomarkers.TimeStamp.isna()]
biomarkers.TimeStamp[biomarkers.TimeStamp.isna()] = biomarkers.FreezerMinusEightyTimeStamp[biomarkers.TimeStamp.isna()]

# Remove individual timestamp variables
biomarkers = biomarkers.drop(columns=['CentrifugationTimeStamp','CollectionTimeStamp','FreezerMinusTwentyTimeStamp','FreezerMinusEightyTimeStamp']).dropna(axis=1,how='all')

# Since all biomarker variables are numeric, melt numeric dataframe into long form
numeric_biomarkers_predictors = biomarkers.dropna(axis=1,how='all').reset_index(drop=True).melt(id_vars=['GUPI','TimeStamp'],var_name='VARIABLE',value_name='VALUE')

## Central Haemostasis information
# Load central_haemostasis data
central_haemostasis = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/CentralHaemostasis/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients
central_haemostasis = central_haemostasis[central_haemostasis.GUPI.isin(cv_splits.GUPI)].dropna(axis=1,how='all').reset_index(drop=True)
central_haemostasis_names = list(central_haemostasis.columns)

# Load inspection table and gather names of central_haemostasis variables
central_haemostasis_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/CentralHaemostasis/inspection_table.xlsx',sheet_name='variables',na_values = ["NA","NaN"," ", ""])
central_haemostasis_names = central_haemostasis_desc.name[central_haemostasis_desc.name.isin(central_haemostasis_names)].to_list()
central_haemostasis = central_haemostasis[['GUPI']+central_haemostasis_names].dropna(subset=[n for n in central_haemostasis_names if n not in ['CollectionDate','CollectionTime','CentrifugationDate','CentrifugationTime','FreezerMinusTwentyDate','FreezerMinusTwentyTime','FreezerMinusEightyDate','FreezerMinusEightyTime','Timepoints']],how='all').reset_index(drop=True)

# Remove central_haemostasis values without any date information
central_haemostasis = central_haemostasis[(~central_haemostasis.CollectionDate.isna())|(~(central_haemostasis.CentrifugationDate.isna()))|(~(central_haemostasis.FreezerMinusTwentyDate.isna()))|(~(central_haemostasis.FreezerMinusEightyDate.isna()))].reset_index(drop=True)

# Construct central_haemostasis timestamps
central_haemostasis['CollectionTimeStamp'] = central_haemostasis[['CollectionDate', 'CollectionTime']].astype(str).agg(' '.join, axis=1)
central_haemostasis['CollectionTimeStamp'][central_haemostasis.CollectionDate.isna() | central_haemostasis.CollectionTime.isna()] = np.nan
central_haemostasis['CollectionTimeStamp'] = pd.to_datetime(central_haemostasis['CollectionTimeStamp'],format = '%Y-%m-%d %H:%M:%S')

central_haemostasis['CentrifugationTimeStamp'] = central_haemostasis[['CentrifugationDate', 'CentrifugationTime']].astype(str).agg(' '.join, axis=1)
central_haemostasis['CentrifugationTimeStamp'][central_haemostasis.CentrifugationDate.isna() | central_haemostasis.CentrifugationTime.isna()] = np.nan
central_haemostasis['CentrifugationTimeStamp'] = pd.to_datetime(central_haemostasis['CentrifugationTimeStamp'],format = '%Y-%m-%d %H:%M:%S')

central_haemostasis['FreezerMinusTwentyTimeStamp'] = central_haemostasis[['FreezerMinusTwentyDate', 'FreezerMinusTwentyTime']].astype(str).agg(' '.join, axis=1)
central_haemostasis['FreezerMinusTwentyTimeStamp'][central_haemostasis.FreezerMinusTwentyDate.isna() | central_haemostasis.FreezerMinusTwentyTime.isna()] = np.nan
central_haemostasis['FreezerMinusTwentyTimeStamp'] = pd.to_datetime(central_haemostasis['FreezerMinusTwentyTimeStamp'],format = '%Y-%m-%d %H:%M:%S')

central_haemostasis['FreezerMinusEightyTimeStamp'] = central_haemostasis[['FreezerMinusEightyDate', 'FreezerMinusEightyTime']].astype(str).agg(' '.join, axis=1)
central_haemostasis['FreezerMinusEightyTimeStamp'][central_haemostasis.FreezerMinusEightyDate.isna() | central_haemostasis.FreezerMinusEightyTime.isna()] = np.nan
central_haemostasis['FreezerMinusEightyTimeStamp'] = pd.to_datetime(central_haemostasis['FreezerMinusEightyTimeStamp'],format = '%Y-%m-%d %H:%M:%S')

# If `CollectionTimeStamp` is missing, replace with median collection time
median_collect_time = pd.to_datetime(central_haemostasis.CollectionTime,format = '%H:%M:%S').dropna().median().strftime('%H:%M:%S')
central_haemostasis.CollectionTimeStamp[central_haemostasis.CollectionTimeStamp.isna()] = pd.to_datetime(central_haemostasis.CollectionDate[central_haemostasis.CollectionTimeStamp.isna()] + ' ' + median_collect_time,format = '%Y-%m-%d %H:%M:%S')

# If `CentrifugationTimeStamp` is missing, and Centrifugation date matches Collection date, replace with Collection timestamp
central_haemostasis.CentrifugationTimeStamp[central_haemostasis.CentrifugationTimeStamp.isna()&(central_haemostasis.CentrifugationDate==central_haemostasis.CollectionDate)] = central_haemostasis.CollectionTimeStamp[central_haemostasis.CentrifugationTimeStamp.isna()&(central_haemostasis.CentrifugationDate==central_haemostasis.CollectionDate)]

# Remove original Date and Time columns and apply categorizer
central_haemostasis = central_haemostasis.drop(columns=['CollectionDate','CollectionTime','CentrifugationDate','CentrifugationTime','FreezerMinusTwentyDate','FreezerMinusTwentyTime','FreezerMinusEightyDate','FreezerMinusEightyTime','Timepoints']).apply(categorizer).dropna(axis=1,how='all')

# Assign final timestamp variable based on hierarchy
central_haemostasis['TimeStamp'] = central_haemostasis['CentrifugationTimeStamp']
central_haemostasis.TimeStamp[central_haemostasis.TimeStamp.isna()] = central_haemostasis.CollectionTimeStamp[central_haemostasis.TimeStamp.isna()]
central_haemostasis.TimeStamp[central_haemostasis.TimeStamp.isna()] = central_haemostasis.FreezerMinusTwentyTimeStamp[central_haemostasis.TimeStamp.isna()]
central_haemostasis.TimeStamp[central_haemostasis.TimeStamp.isna()] = central_haemostasis.FreezerMinusEightyTimeStamp[central_haemostasis.TimeStamp.isna()]

# Remove individual timestamp variables
central_haemostasis = central_haemostasis.drop(columns=['CentrifugationTimeStamp','CollectionTimeStamp','FreezerMinusTwentyTimeStamp','FreezerMinusEightyTimeStamp']).dropna(axis=1,how='all')

# Separate categorical and numeric variables into separate dataframes
numeric_central_haemostasis_names = np.sort(central_haemostasis.select_dtypes(include=['number']).columns.values)
categorical_central_haemostasis_names = np.sort(central_haemostasis.select_dtypes(exclude=['number']).drop(columns=['GUPI','TimeStamp']).columns.values)

# Melt numeric dataframe into long form
numeric_central_haemostasis_predictors = central_haemostasis[np.insert(numeric_central_haemostasis_names,0,['GUPI','TimeStamp'])].dropna(axis=1,how='all').dropna(subset=numeric_central_haemostasis_names,how='all').reset_index(drop=True).melt(id_vars=['GUPI','TimeStamp'],var_name='VARIABLE',value_name='VALUE')
numeric_central_haemostasis_predictors['VARIABLE'] = numeric_central_haemostasis_predictors.VARIABLE.str.replace('_','')

# Remove formatting from all categorical variable string values
categorical_central_haemostasis_predictors = central_haemostasis[np.insert(categorical_central_haemostasis_names,0,['GUPI','TimeStamp'])].dropna(axis=1,how='all').dropna(subset=categorical_central_haemostasis_names,how='all').reset_index(drop=True)
categorical_central_haemostasis_predictors.columns = categorical_central_haemostasis_predictors.columns.str.replace('_','')
categorical_central_haemostasis_names = [x.replace('_', '') for x in categorical_central_haemostasis_names]
categorical_central_haemostasis_predictors[categorical_central_haemostasis_names] = categorical_central_haemostasis_predictors[categorical_central_haemostasis_names].apply(lambda x: x.astype(str).str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)

# Construct tokens
categorical_central_haemostasis_predictors = categorical_central_haemostasis_predictors.melt(id_vars=['GUPI','TimeStamp']).drop_duplicates(subset=['GUPI','TimeStamp','value']).groupby(['GUPI','TimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Imaging information
# Load imaging data
imaging = pd.read_csv('/home/sb2406/rds/hpc-work/CENTER-TBI/Imaging/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter study set patients and exclude ER images
imaging = imaging[(imaging.GUPI.isin(cv_splits.GUPI))&~((imaging.CTPatientLocation=='ED')|(imaging.MRIPatientLocation=='ED'))].dropna(axis=1,how='all').reset_index(drop=True)

# Remove all images without an associated date, if not collected in the ICU
imaging = imaging[(~imaging.ExperimentDate.isna())|(~imaging.AcquisitionDate.isna())|(imaging.CTPatientLocation=='ICU')|(imaging.MRIPatientLocation=='ICU')].reset_index(drop=True)

# Calculate new variable to indicate lesion after CT
imaging_new_variable_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/Imaging/inspection_table.xlsx',sheet_name='new_variable_calculation',na_values = ["NA","NaN"," ", ""])
new_var_imaging_name = imaging_new_variable_desc.name[0]
imaging['CTLesionDetected'] = np.nan
imaging.CTLesionDetected[imaging[new_var_imaging_name] == 0] = 0
imaging.CTLesionDetected[(imaging[new_var_imaging_name] != 0)&(imaging[new_var_imaging_name] != 88)&(~imaging[new_var_imaging_name].isna())] = 1
imaging.CTLesionDetected[imaging[new_var_imaging_name] == 88] = 2

# Load inspection table and gather names of permitted variables
imaging_names = [n for n in imaging if n not in ['GUPI','CRFForm']]
imaging_variable_desc = pd.read_excel('/home/sb2406/rds/hpc-work/CENTER-TBI/Imaging/inspection_table.xlsx',sheet_name='variables',na_values = ["NA","NaN"," ", ""])
imaging_names = imaging_variable_desc.name[imaging_variable_desc.name.isin(imaging_names)].to_list()+['CTLesionDetected']
imaging = imaging[['GUPI']+imaging_names].dropna(subset=imaging_names,how='all').reset_index(drop=True)
imaging.columns = imaging.columns.str.replace('_','')
imaging_names = [x.replace('_', '') for x in imaging_names]

# Construct imaging timestamps
imaging['ExperimentTimeStamp'] = imaging[['ExperimentDate', 'ExperimentTime']].astype(str).agg(' '.join, axis=1)
imaging['ExperimentTimeStamp'][imaging.ExperimentDate.isna() | imaging.ExperimentTime.isna()] = np.nan
imaging['ExperimentTimeStamp'] = pd.to_datetime(imaging['ExperimentTimeStamp'],format = '%Y-%m-%d %H:%M:%S')

imaging['AcquisitionTimeStamp'] = imaging[['AcquisitionDate', 'AcquisitionTime']].astype(str).agg(' '.join, axis=1)
imaging['AcquisitionTimeStamp'][imaging.AcquisitionDate.isna() | imaging.AcquisitionTime.isna()] = np.nan
imaging['AcquisitionTimeStamp'] = pd.to_datetime(imaging['AcquisitionTimeStamp'],format = '%Y-%m-%d %H:%M:%S')

# If `ExperimentTimeStamp` is missing, replace with median Experiment time
median_experiment_time = pd.to_datetime(imaging.ExperimentTime,format = '%H:%M:%S').dropna().median().strftime('%H:%M:%S')
imaging.ExperimentTimeStamp[imaging.ExperimentTimeStamp.isna()] = pd.to_datetime(imaging.ExperimentDate[imaging.ExperimentTimeStamp.isna()] + ' ' + median_experiment_time,format = '%Y-%m-%d %H:%M:%S')

# Remove original Date and Time columns and apply categorizer
imaging = imaging.drop(columns=['ExperimentDate','ExperimentTime','AcquisitionDate','AcquisitionTime']).apply(categorizer).dropna(axis=1,how='all')

# Assign final timestamp variable based on hierarchy
imaging['TimeStamp'] = imaging['AcquisitionTimeStamp']
imaging.TimeStamp[imaging.TimeStamp.isna()] = imaging.ExperimentTimeStamp[imaging.TimeStamp.isna()]

# Remove individual timestamp variables
imaging = imaging.drop(columns=['ExperimentTimeStamp','AcquisitionTimeStamp']).dropna(axis=1,how='all')

# First, separate out CT and MRI dataframes
CT_imaging = imaging[imaging.XsiType == 'xnat:ctSessionData'].merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI').reset_index(drop=True)
CT_imaging = CT_imaging[CT_imaging.TimeStamp <= CT_imaging.ICUDischTimeStamp].drop(columns='ICUDischTimeStamp').dropna(axis=1,how='all').reset_index(drop=True)

MR_imaging = imaging[imaging.XsiType == 'xnat:mrSessionData'].merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI').reset_index(drop=True)
MR_imaging = MR_imaging[MR_imaging.TimeStamp <= MR_imaging.ICUDischTimeStamp].drop(columns='ICUDischTimeStamp').dropna(axis=1,how='all').reset_index(drop=True)

# Split CT dataframe into categorical and numeric
numeric_CT_imaging_names = np.sort(CT_imaging.select_dtypes(include=['number']).columns.values)
numeric_CT_imaging_predictors = CT_imaging[np.insert(numeric_CT_imaging_names,0,['GUPI','TimeStamp'])].dropna(axis=1,how='all').dropna(subset=numeric_CT_imaging_names,how='all').reset_index(drop=True).melt(id_vars=['GUPI','TimeStamp'],var_name='VARIABLE',value_name='VALUE')

categorical_CT_imaging_names = np.sort(CT_imaging.select_dtypes(exclude=['number']).drop(columns=['GUPI','TimeStamp']).columns.values)
categorical_CT_imaging_predictors = CT_imaging[np.insert(categorical_CT_imaging_names,0,['GUPI','TimeStamp'])].dropna(axis=1,how='all').dropna(subset=categorical_CT_imaging_names,how='all').reset_index(drop=True)
categorical_CT_imaging_predictors[categorical_CT_imaging_names] = categorical_CT_imaging_predictors[categorical_CT_imaging_names].apply(lambda x: x.astype(str).str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
categorical_CT_imaging_predictors = categorical_CT_imaging_predictors.melt(id_vars=['GUPI','TimeStamp']).drop_duplicates(subset=['GUPI','TimeStamp','value']).groupby(['GUPI','TimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

# Split MR dataframe into categorical and numeric
numeric_MR_imaging_names = np.sort(MR_imaging.select_dtypes(include=['number']).columns.values)
numeric_MR_imaging_predictors = MR_imaging[np.insert(numeric_MR_imaging_names,0,['GUPI','TimeStamp'])].dropna(axis=1,how='all').dropna(subset=numeric_MR_imaging_names,how='all').reset_index(drop=True).melt(id_vars=['GUPI','TimeStamp'],var_name='VARIABLE',value_name='VALUE')

categorical_MR_imaging_names = np.sort(MR_imaging.select_dtypes(exclude=['number']).drop(columns=['GUPI','TimeStamp']).columns.values)
categorical_MR_imaging_predictors = MR_imaging[np.insert(categorical_MR_imaging_names,0,['GUPI','TimeStamp'])].dropna(axis=1,how='all').dropna(subset=categorical_MR_imaging_names,how='all').reset_index(drop=True)
categorical_MR_imaging_predictors[categorical_MR_imaging_names] = categorical_MR_imaging_predictors[categorical_MR_imaging_names].apply(lambda x: x.astype(str).str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)).fillna('NAN').apply(lambda x: x.name + '_' + x)
categorical_MR_imaging_predictors = categorical_MR_imaging_predictors.melt(id_vars=['GUPI','TimeStamp']).drop_duplicates(subset=['GUPI','TimeStamp','value']).groupby(['GUPI','TimeStamp'],as_index=False).value.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'})

## Compile and save timestamped single-event predictors
# Categorical predictors - ensure unique tokens per GUPI and save
categorical_timestamp_tokens = pd.concat([GCSFirstHosp_info,GCSOther_info,HospDisch_info,Reintubation_info,ReMechVentilation_info,categorical_best_gcs_predictors,categorical_high_temp_predictors,categorical_low_temp_predictors,categorical_worst_gcs_predictors,categorical_daily_TIL_predictors,categorical_daily_hourly_predictors,categorical_labs_predictors,categorical_central_haemostasis_predictors,categorical_CT_imaging_predictors,categorical_MR_imaging_predictors],ignore_index=True)

# Group by GUPI and TimeStamp and merge tokens
categorical_timestamp_tokens = categorical_timestamp_tokens.groupby(['GUPI','TimeStamp'],as_index=False).Token.aggregate(lambda x: ' '.join(x)).rename(columns={'value':'Token'}).reset_index(drop=True)

# Iterate through entries, ensure unique tokens and end tokens
for curr_idx in tqdm(range(categorical_timestamp_tokens.shape[0]), 'Cleaning timestamped single-event predictors'):
    curr_token_set = categorical_timestamp_tokens.Token[curr_idx]
    cleaned_token_set = ' '.join(np.sort(np.unique(curr_token_set.split())))
    categorical_timestamp_tokens.Token[curr_idx] = cleaned_token_set

# Filter out all datapoints with timestamp after ICU discharge
categorical_timestamp_tokens = categorical_timestamp_tokens.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
categorical_timestamp_tokens = categorical_timestamp_tokens[categorical_timestamp_tokens.TimeStamp <= categorical_timestamp_tokens.ICUDischTimeStamp].drop(columns='ICUDischTimeStamp').reset_index(drop=True)

# Sort values and save timestamped single-event predictors
categorical_timestamp_tokens = categorical_timestamp_tokens.sort_values(by=['GUPI','TimeStamp']).reset_index(drop=True)
categorical_timestamp_tokens.to_pickle(os.path.join(form_pred_dir,'categorical_timestamp_event_predictors.pkl'))

# Numeric predictors
numeric_timestamp_values = pd.concat([numeric_high_hr_predictors,numeric_high_sbp_predictors,numeric_high_spo2_predictors,numeric_high_temp_predictors,numeric_low_hr_predictors,numeric_low_sbp_predictors,numeric_low_spo2_predictors,numeric_low_temp_predictors,numeric_daily_TIL_predictors,numeric_daily_hourly_predictors,numeric_labs_predictors,numeric_biomarkers_predictors,numeric_central_haemostasis_predictors,numeric_CT_imaging_predictors,numeric_MR_imaging_predictors],ignore_index=True).drop_duplicates().reset_index(drop=True)
numeric_timestamp_values = numeric_timestamp_values.merge(CENTER_TBI_ICU_datetime[['GUPI','ICUDischTimeStamp']],how='left',on='GUPI')
numeric_timestamp_values = numeric_timestamp_values[numeric_timestamp_values.TimeStamp <= numeric_timestamp_values.ICUDischTimeStamp].drop(columns='ICUDischTimeStamp').reset_index(drop=True)
numeric_timestamp_values = numeric_timestamp_values.sort_values(by=['GUPI','TimeStamp','VARIABLE']).reset_index(drop=True)
numeric_timestamp_values.to_pickle(os.path.join(form_pred_dir,'numeric_timestamp_event_predictors.pkl'))