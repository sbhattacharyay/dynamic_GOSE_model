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

### 4. Train logistic regression concise-predictor-based models (CPM)

<ol type="a">
  <li><h4><a href="scripts/04a_train_full_set_models.py">Train full-context trajectory-generating models</a></h4> In this <code>.py</code> file, we extract all heterogeneous types of variables from CENTER-TBI and fix erroneous timestamps and formats.</li>
  <li><h4><a href="scripts/04b_compile_full_set_model_predictions.py">Compile generated trajectories across repeated cross-validation and different hyperparameter configurations</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. </li>
  <li><h4><a href="scripts/04c_validation_set_bootstrapping_for_dropout.py">Calculate validation set calibration and discrimination of generated trajectories for hyperparameter configuration dropout</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. </li>
  <li><h4><a href="scripts/04d_dropout_configurations.py">Compile validation set performance metrics and dropout under-performing hyperparameter configurations</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. </li>
  <li><h4><a href="scripts/04e_test_set_performance.py">Calculate calibration and discrimination performance metrics of generated trajectories of the testing set with bootstrapping</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. </li>
  <li><h4><a href="scripts/04f_test_set_confidence_intervals.py">Compile testing set trajectory performance metrics and calculate confidence intervals</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. </li>
</ol>

### 5. Train logistic regression concise-predictor-based models (CPM)

<ol type="a">
  <li><h4><a href="scripts/04a_train_full_set_models.py">Train full-context trajectory-generating models</a></h4> In this <code>.py</code> file, we extract all heterogeneous types of variables from CENTER-TBI and fix erroneous timestamps and formats.</li>
  <li><h4><a href="scripts/04b_compile_full_set_model_predictions.py">Compile generated trajectories across repeated cross-validation and different hyperparameter configurations</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. </li>
  <li><h4><a href="scripts/04c_validation_set_bootstrapping_for_dropout.py">Calculate validation set calibration and discrimination of generated trajectories for hyperparameter configuration dropout</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. </li>
  <li><h4><a href="scripts/04d_dropout_configurations.py">Compile validation set performance metrics and dropout under-performing hyperparameter configurations</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. </li>
  <li><h4><a href="scripts/04e_test_set_performance.py">Calculate calibration and discrimination performance metrics of generated trajectories of the testing set with bootstrapping</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. </li>
  <li><h4><a href="scripts/04f_test_set_confidence_intervals.py">Compile testing set trajectory performance metrics and calculate confidence intervals</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. </li>
  <li><h4><a href="scripts/04f_test_set_confidence_intervals.py">Compile testing set trajectory performance metrics and calculate confidence intervals</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. </li>
  <li><h4><a href="scripts/04f_test_set_confidence_intervals.py">Compile testing set trajectory performance metrics and calculate confidence intervals</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. </li>
</ol>

### 5. [Assess CPM_MNLR and CPM_POLR performance](scripts/05_CPM_logreg_performance.py)
In this `.py` file, we create and save bootstrapping resamples used for all model performance evaluation. We prepare compiled CPM_MNLR and CPM_POLR testing set predictions, and calculate/save performance metrics.

### 6. Train and optimise CPM_DeepMN and CPM_DeepOR

<ol type="a">
  <li><h4><a href="scripts/06a_CPM_deep.py">Train deep learning concise-predictor-based models (CPM)</a></h4> In this <code>.py</code> file, we first create a grid of tuning configuration-cross-validation combinations and train CPM_DeepMN or CPM_DeepOR models based on provided hyperparameter row index. This is run, with multi-array indexing, on the HPC using a <a href="scripts/06a_CPM_deep.sh">bash script</a>.</li>
  <li><h4><a href="scripts/06b_CPM_deep_interrepeat_dropout.py">Perform interrepeat hyperparameter configuration dropout on deep learning concise-predictor-based models (CPM)</a></h4> In this <code>.py</code> file, we calculate ORC of extant validation predictions, prepare bootstrapping resamples for configuration dropout, and dropout configurations that are consistently (<span>&#593;</span> = .05) inferior in performance. </li>
  <li><h4><a href="scripts/06c_CPM_deep_hyperparameter_testing.py">Calculate ORC in bootstrapping resamples to determine dropout configurations</a></h4> In this <code>.py</code> file, we calculate ORC in each resample and compare to 'optimal' configuration. This is run, with multi-array indexing, on the HPC using a <a href="scripts/06c_CPM_deep_hyperparameter_testing.sh">bash script</a>.</li>
</ol>

### 7. Calculate and compile CPM_DeepMN and CPM_DeepOR metrics

<ol type="a">
  <li><h4><a href="scripts/07a_CPM_deep_performance.py">Assess CPM_DeepMN and CPM_DeepOR performance</a></h4> In this <code>.py</code> file, we calculate perfomance metrics on resamples. This is run, with multi-array indexing, on the HPC using a <a href="scripts/07a_CPM_deep_performance.sh">bash script</a>.</li>
  <li><h4><a href="scripts/07b_CPM_compile_metrics.py">Compile CPM_DeepMN and CPM_DeepOR performance metrics and calculate confidence intervals</a></h4> In this <code>.py</code> file, we compile all CPM_DeepMN and CPM_DeepOR performance metrics and calculate confidence intervals on all CPM performance metrics. </li>
</ol>

### 8. [Prepare predictor tokens for the training of all-predictor-based models (APMs)](scripts/08_prepare_APM_tokens.R)
In this `.R` file, we load and prepare formatted CENTER-TBI predictor tokens. Then, convert formatted predictors to tokens for each repeated cross-validation partition.

### 9. [Train APM dictionaries and convert tokens to embedding layer indices](scripts/09_prepare_APM_dictionaries.py)
In this `.py` file, we train APM dictionaries per repeated cross-validation partition and convert tokens to indices.

### 10. Train and optimise APM_MN and APM_OR

<ol type="a">
  <li><h4><a href="scripts/10a_APM_deep.py">Train deep learning all-predictor-based models (APM)</a></h4> In this <code>.py</code> file, we first create a grid of tuning configuration-cross-validation combinations and train APM_MN or APM_OR models based on provided hyperparameter row index. This is run, with multi-array indexing, on the HPC using a <a href="scripts/10a_APM_deep.sh">bash script</a>.</li>
  <li><h4><a href="scripts/10b_APM_deep_interrepeat_dropout.py">Perform interrepeat hyperparameter configuration dropout on deep learning all-predictor-based models (APM)</a></h4> In this <code>.py</code> file, we calculate ORC of extant validation predictions, prepare bootstrapping resamples for configuration dropout, and dropout configurations that are consistently (<span>&#593;</span> = .05) inferior in performance. </li>
  <li><h4><a href="scripts/10c_APM_deep_hyperparameter_testing.py">Calculate ORC in bootstrapping resamples to determine dropout configurations</a></h4> In this <code>.py</code> file, we calculate ORC in each resample and compare to 'optimal' configuration. This is run, with multi-array indexing, on the HPC using a <a href="scripts/10c_APM_deep_hyperparameter_testing.sh">bash script</a>.</li>
</ol>

### 11. Calculate and compile APM_MN and APM_OR metrics

<ol type="a">
  <li><h4><a href="scripts/11a_APM_deep_performance.py">Assess APM_MN and APM_OR performance</a></h4> In this <code>.py</code> file, we calculate perfomance metrics on resamples. This is run, with multi-array indexing, on the HPC using a <a href="scripts/11a_APM_deep_performance.sh">bash script</a>.</li>
  <li><h4><a href="scripts/11b_APM_compile_metrics.py">Compile APM_MN and APM_OR performance metrics and calculate confidence intervals</a></h4> In this <code>.py</code> file, we compile all APM_MN and APM_OR performance metrics and calculate confidence intervals on all CPM performance metrics. </li>
</ol>

### 12. Assess feature significance in APM_MN

<ol type="a">
  <li><h4><a href="scripts/12a_APM_deep_SHAP.py">Calculate SHAP values for APM_MN</a></h4> In this <code>.py</code> file, we find all top-performing model checkpoint files for SHAP calculation and calculate SHAP values based on given parameters. This is run, with multi-array indexing, on the HPC using a <a href="scripts/12a_APM_deep_SHAP.sh">bash script</a>.</li>
  <li><h4><a href="scripts/12b_APM_compile_SHAP.py">Compile SHAP values for each GUPI-output type combination from APM_MN</a></h4> In this <code>.py</code> file, we find all files storing calculated SHAP values and create combinations with study GUPIs and compile SHAP values for the given GUPI and output type combination. This is run, with multi-array indexing, on the HPC using a <a href="scripts/12b_APM_compile_SHAP.sh">bash script</a>.</li>
  <li><h4><a href="scripts/12c_APM_summarise_SHAP.py">Summarise SHAP values across study set</a></h4> In this <code>.py</code> file, we find all files storing GUPI-specific SHAP values and compile and save summary SHAP values across study patient set. </li>
  <li><h4><a href="scripts/12d_APM_compile_significance_weights.py">Summarise aggregation weights across trained APM set</a></h4> In this <code>.py</code> file, we compile significance weights across trained APMs and summarise significance weights. </li>
</ol>

### 13. [Prepare extended concise predictor set for ordinal prediction](scripts/13_prepare_extended_concise_predictor_set.R)
In this `.R` file, we load IMPACT variables from CENTER-TBI, load and prepare the added variables from CENTER-TBI, and multiply impute extended concise predictor set in parallel. The training set for each repeated k-fold CV partition is used to train an independent predictive mean matching imputation transformation for that partition. The result is 100 imputations, one for each repeated k-fold cross validation partition.

### 14. [Train logistic regression extended concise-predictor-based models (eCPM)](scripts/14_eCPM_logreg.py)
In this `.py` file, we define a function to train logistic regression eCPMs given the repeated cross-validation dataframe. Then we perform parallelised training of logistic regression eCPMs and testing set prediction. Finally, we compile testing set predictions.

### 15. [Assess eCPM_MNLR and eCPM_POLR performance](scripts/15_eCPM_logreg_performance.py)
In this `.py` file, we load the common bootstrapping resamples (that will be used for all model performance evaluation), prepare compiled eCPM_MNLR and eCPM_POLR testing set predictions, and calculate/save performance metrics

### 16. Train and optimise eCPM_DeepMN and eCPM_DeepOR

<ol type="a">
  <li><h4><a href="scripts/16a_eCPM_deep.py">Train deep learning extended concise-predictor-based models (eCPM)</a></h4> In this <code>.py</code> file, we first create a grid of tuning configuration-cross-validation combinations and train eCPM_DeepMN or eCPM_DeepOR models based on provided hyperparameter row index. This is run, with multi-array indexing, on the HPC using a <a href="scripts/16a_eCPM_deep.sh">bash script</a>.</li>
  <li><h4><a href="scripts/16b_eCPM_deep_interrepeat_dropout.py">Perform interrepeat hyperparameter configuration dropout on deep learning extended concise-predictor-based models (eCPM)</a></h4> In this <code>.py</code> file, we calculate ORC of extant validation predictions, prepare bootstrapping resamples for configuration dropout, and dropout configurations that are consistently (<span>&#593;</span> = .05) inferior in performance </li>
  <li><h4><a href="scripts/16c_eCPM_deep_hyperparameter_testing.py">Calculate ORC in bootstrapping resamples to determine dropout configurations</a></h4> In this <code>.py</code> file, we calculate ORC in each resample and compare to 'optimal' configuration. This is run, with multi-array indexing, on the HPC using a <a href="scripts/16c_eCPM_deep_hyperparameter_testing.sh">bash script</a>.</li>
</ol>

### 17. Calculate and compile eCPM_DeepMN and eCPM_DeepOR metrics

<ol type="a">
  <li><h4><a href="scripts/17a_eCPM_deep_performance.py">Assess eCPM_DeepMN and eCPM_DeepOR performance</a></h4> In this <code>.py</code> file, we calculate perfomance metrics on resamples. This is run, with multi-array indexing, on the HPC using a <a href="scripts/17a_eCPM_deep_performance.sh">bash script</a>.</li>
  <li><h4><a href="scripts/17b_eCPM_compile_metrics.py">Compile eCPM_DeepMN and eCPM_DeepOR performance metrics and calculate confidence intervals</a></h4> In this <code>.py</code> file, we compile all eCPM_DeepMN and eCPM_DeepOR performance metrics and calculate confidence intervals on all CPM performance metrics. </li>
</ol>

### 18. [Perform ordinal regression analysis on study characteristics and predictors](scripts/18_ordinal_regression_analysis.py)
In this `.py` file, we perform ordinal regression analysis on summary characteristics, perform ordinal regression analysis on CPM characteristics, and perform ordinal regression analysis on eCPM characteristics.

### 19. [Visualise study results for manuscript](scripts/19_manuscript_visualisations.R)
In this `.R` file, we produce the figures for the manuscript and the supplementary figures. The large majority of the quantitative figures in the manuscript are produced using the `ggplot` package.

## Citation
```
@article{10.1371/journal.pone.0270973,
    doi = {10.1371/journal.pone.0270973},
    author = {Bhattacharyay, Shubhayu AND Milosevic, Ioan AND Wilson, Lindsay AND Menon, David K. AND Stevens, Robert D. AND Steyerberg, Ewout W. AND Nelson, David W. AND Ercole, Ari AND the CENTER-TBI investigators participants},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {The leap to ordinal: Detailed functional prognosis after traumatic brain injury with a flexible modelling approach},
    year = {2022},
    month = {07},
    volume = {17},
    url = {https://doi.org/10.1371/journal.pone.0270973},
    pages = {1-29},
    abstract = {When a patient is admitted to the intensive care unit (ICU) after a traumatic brain injury (TBI), an early prognosis is essential for baseline risk adjustment and shared decision making. TBI outcomes are commonly categorised by the Glasgow Outcome Scale–Extended (GOSE) into eight, ordered levels of functional recovery at 6 months after injury. Existing ICU prognostic models predict binary outcomes at a certain threshold of GOSE (e.g., prediction of survival [GOSE > 1]). We aimed to develop ordinal prediction models that concurrently predict probabilities of each GOSE score. From a prospective cohort (n = 1,550, 65 centres) in the ICU stratum of the Collaborative European NeuroTrauma Effectiveness Research in TBI (CENTER-TBI) patient dataset, we extracted all clinical information within 24 hours of ICU admission (1,151 predictors) and 6-month GOSE scores. We analysed the effect of two design elements on ordinal model performance: (1) the baseline predictor set, ranging from a concise set of ten validated predictors to a token-embedded representation of all possible predictors, and (2) the modelling strategy, from ordinal logistic regression to multinomial deep learning. With repeated k-fold cross-validation, we found that expanding the baseline predictor set significantly improved ordinal prediction performance while increasing analytical complexity did not. Half of these gains could be achieved with the addition of eight high-impact predictors to the concise set. At best, ordinal models achieved 0.76 (95% CI: 0.74–0.77) ordinal discrimination ability (ordinal c-index) and 57% (95% CI: 54%– 60%) explanation of ordinal variation in 6-month GOSE (Somers’ Dxy). Model performance and the effect of expanding the predictor set decreased at higher GOSE thresholds, indicating the difficulty of predicting better functional outcomes shortly after ICU admission. Our results motivate the search for informative predictors that improve confidence in prognosis of higher GOSE and the development of ordinal dynamic prediction models.},
    number = {7}
}
```
