# Ordinal, full-context prognosis-based trajectories of traumatic brain injury patients in European ICUs
Everything over time: a data-driven disease course of traumatic brain injury in European intensive care units

## Contents

- [Overview](#overview)
- [Abstract](#abstract)
- [Code](#code)
- [License](./LICENSE)
- [Citation](#citation)

## Overview

This repository contains the code underlying the article entitled **Everything over time: a data-driven disease course of traumatic brain injury in European intensive care units** from the Collaborative European NeuroTrauma Effectiveness Research in TBI ([CENTER-TBI](https://www.center-tbi.eu/)) consortium. In this file, we present the abstract, to outline the motivation for the work and the findings, and then a brief description of the code with which we generate these finding and achieve this objective.\
\
The code on this repository is commented throughout to provide a description of each step alongside the code which achieves it.

## Abstract
### Background
Existing methods to characterise the evolving condition of traumatic brain injury (TBI) patients in the intensive care unit (ICU) do not capture the context necessary for individualising treatment. We aimed to develop a modelling strategy which integrates all heterogenous data stored in medical records to produce an interpretable disease course for each TBI patient’s ICU stay.
### Methods
From a prospective cohort (_n_=1,550, 65 centres, 19 countries) of European TBI patients, we extracted all 1,166 variables collected before or during ICU stay as well as six-month functional outcome on the Glasgow Outcome Scale – Extended (GOSE). We trained recurrent neural network models to map a token-embedded time series representation of all variables to an ordinal GOSE prognosis every two hours. With 20 repeats of five-fold cross-validation, we evaluated calibration and the explanation of ordinal variance in GOSE with Somers’ _D<sub>xy</sub>_. Furthermore, we implemented the TimeSHAP algorithm to calculate the contribution of variables and prior timepoints towards significant transitions in patient trajectories.
### Findings
Our modelling strategy achieved calibration at eight hours post-admission, and the full range of variables explained up to 52·2% (95% CI: 50·2%–54·3%) of the variance in ordinal, six-month functional outcome. Most of this explanation was derived from pre-ICU information. Information collected during ICU stay increased explanation, though not enough to counter the difficulty of characterising longer-stay patients. Static variables with the highest contributions were physician-based prognoses and certain demographic and CT features. Among dynamic variables, markers of intracranial hypertension and neurological function contributed the most.
### Interpretation
We show the feasibility of a data-driven approach for individualised TBI characterisation without the need for variable curation, cleaning, or imputation. Our results also highlight potential investigative avenues to help explain the remaining half of variance in functional outcome.
### Funding
NIHR Brain Injury MedTech Co-operative, EU 7<sup>th</sup> Framework, Hannelore Kohl, OneMind, Integra Neurosciences, Gates Cambridge.

## Code 
All of the code used in this work can be found in the `./scripts` directory as Python (`.py`), R (`.R`), or bash (`.sh`) scripts. Moreover, custom classes have been saved in the `./scripts/classes` sub-directory, custom functions have been saved in the `./scripts/functions` sub-directory, and custom PyTorch models have been saved in the `./scripts/models` sub-directory.

### 1. [Extract study sample from CENTER-TBI dataset and define ICU stays](scripts/01_prepare_study_sample_timestamps.py)
In this `.py` file, we extract the study sample from the CENTER-TBI dataset, filter patients by our study criteria, and determine ICU admission and discharge times for time window discretisation. We also perform proportional odds logistic regression analysis to determine significant effects among summary characteristics.

### 2. [Partition CENTER-TBI for stratified, repeated k-fold cross-validation](scripts/02_partition_for_cv.py)
In this `.py` file, we create 100 partitions, stratified by 6-month GOSE, for repeated k-fold cross-validation, and save the splits into a dataframe for subsequent scripts.

### 3. Tokenise all CENTER-TBI variables and place into discretised ICU stay time windows

<ol type="a">
  <li><h4><a href="scripts/03a_format_CENTER_TBI_data_for_tokenisation.py">Format CENTER-TBI variables for tokenisation</a></h4> In this <code>.py</code> file, we extract all heterogeneous types of variables from CENTER-TBI and fix erroneous timestamps and formats.</li>
  <li><h4><a href="scripts/03b_convert_ICU_stays_into_tokenised_sets.py">Convert full patient records over ICU stays into tokenised time windows</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. </li>
</ol>

### 4. Train and evaluate full-context ordinal-trajectory-generating models

<ol type="a">
  <li><h4><a href="scripts/04a_train_full_set_models.py">Train full-context trajectory-generating models</a></h4> In this <code>.py</code> file, we train the trajectory-generating models across the repeated cross-validation splits and the hyperparameter configurations. This is run, with multi-array indexing, on the HPC using a <a href="scripts/04a_train_full_set_models.sh">bash script</a>.</li>
  <li><h4><a href="scripts/04b_compile_full_set_model_predictions.py">Compile generated trajectories across repeated cross-validation and different hyperparameter configurations</a></h4> In this <code>.py</code> file, we compile the training, validation, and testing set trajectories generated by the models and creates bootstrapping resamples for validation set dropout.</li>
  <li><h4><a href="scripts/04c_validation_set_bootstrapping_for_dropout.py">Calculate validation set calibration and discrimination of generated trajectories for hyperparameter configuration dropout</a></h4> In this <code>.py</code> file, we calculate validation set trajectory calibration and discrimination based on provided bootstrapping resample row index. This is run, with multi-array indexing, on the HPC using a <a href="scripts/04c_validation_set_bootstrapping_for_dropout.sh">bash script</a>.</li>
  <li><h4><a href="scripts/04d_dropout_configurations.py">Compile validation set performance metrics and dropout under-performing hyperparameter configurations</a></h4> In this <code>.py</code> file, we compiled the validation set performance metrics and perform bias-corrected bootstrapping dropout for cross-validation (BBCD-CV) to reduce the number of hyperparameter configurations. We also create testing set resamples for final performance calculation bootstrapping. </li>
  <li><h4><a href="scripts/04e_test_set_performance.py">Calculate calibration and discrimination performance metrics of generated trajectories of the testing set with bootstrapping</a></h4> In this <code>.py</code> file, calculate the model calibration and explanation metrics to assess model reliability and information, respectively. This is run, with multi-array indexing, on the HPC using a <a href="scripts/04e_test_set_performance.sh">bash script</a>.</li>
  <li><h4><a href="scripts/04f_test_set_confidence_intervals.py">Compile testing set trajectory performance metrics and calculate confidence intervals</a></h4> In this <code>.py</code> file, we compile the performance metrics and summarise them across bootstrapping resamples to define the 95% confidence intervals for statistical inference. </li>
</ol>

### 5. Interpret variable effects on trajectory generation and evaluate baseline comparison model

<ol type="a">
  <li><h4><a href="scripts/05a_compile_relevance_layer_values.py">Compile and summarise learned weights from model relevance layers</a></h4> In this <code>.py</code> file, we extract and summarise the learned weights from the model relevance layers (trained as PyTorch Embedding layers).</li>
  <li><h4><a href="scripts/05b_prepare_for_TimeSHAP.py">Prepare environment to calculate TimeSHAP for trajectory-generating models</a></h4> In this <code>.py</code> file, we define and identify significant transitions in individual patient trajectories, partition them for bootstrapping, and calculate summarised testing set trajectory information in preparation for TimeSHAP feature contribution calculation. </li>
  <li><h4><a href="scripts/05c_calculate_full_set_TimeSHAP.py">Calculate TimeSHAP values for each patient in parallel</a></h4> In this <code>.py</code> file, we calculate variable and time-window TimeSHAP values for each individual's significant transitions. This is run, with multi-array indexing, on the HPC using a <a href="scripts/05c_calculate_full_set_TimeSHAP.sh">bash script</a>.</li>
  <li><h4><a href="scripts/05d_compile_TimeSHAP_values.py">Compile and summarise TimeSHAP values across the patient set</a></h4> In this <code>.py</code> file, we load all the calculated TimeSHAP values and summarise them for population-level variable and time-window analysis. We also extract the model trajectories of our individual patient for exploration. </li>
  <li><h4><a href="scripts/05e_calculate_feature_robustness.py">Calculate Kendall's tau values for the relationship between variable values and TimeSHAP values to assess feature robustness</a></h4> In this <code>.py</code> file, we perform a variable robustness check by calculating the Kendall's Tau rank correlation coefficient between variable values and the TimeSHAP corresponding to each value. This is run, with multi-array indexing, on the HPC using a <a href="scripts/05e_calculate_feature_robustness.sh">bash script</a>.</li>
  <li><h4><a href="scripts/05f_variable_dropout.py">Dropout variables without significant variable effect direction</a></h4> In this <code>.py</code> file, we determine which variables have a significantly robust relationship with GOSE outcome and drop out those which do not. </li>
  <li><h4><a href="scripts/05g_baseline_model_test_set_performance.py">Calculate testing set performance metrics of baseline ordinal prediction model for comparison</a></h4> In this <code>.py</code> file, calculate the baseline model calibration and explanation metrics to assess model reliability and information, respectively. This model was developed in our previous work (<a href="https://github.com/sbhattacharyay/ordinal_GOSE_prediction">see ordinal GOSE prediction repository</a>) and serves as our baseline for comparison to determine the added value of information collected in the ICU over time. This is run, with multi-array indexing, on the HPC using a <a href="scripts/05g_baseline_model_test_set_performance.sh">bash script</a>.</li>
  <li><h4><a href="scripts/05h_baseline_test_set_confidence_intervals.py">Compile testing set baseline prediction model performance metrics and calculate confidence intervals for comparison</a></h4> In this <code>.py</code> file, compile the baseline model performance metrics and summarise them across bootstrapping resamples to define the 95% confidence intervals for statistical inference. </li>
</ol>

### 6. [Visualise study results for manuscript](scripts/06_manuscript_visualisations.R)
In this `.R` file, we produce the figures for the manuscript and the supplementary figures. The large majority of the quantitative figures in the manuscript are produced using the `ggplot` package.

## Citation
```
```
