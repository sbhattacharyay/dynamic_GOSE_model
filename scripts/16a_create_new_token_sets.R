#### Master Script 16b: Create new token sets from CENTER-TBI database ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Load and prepare formatted CENTER-TBI predictor tokens
# III. Convert formatted predictors to tokens for each repeated cross-validation partition

### I. Initialisation
# Import necessary libraries
library(tidyverse)
library(lubridate)
library(readxl)
library(doParallel)
library(foreach)
library(tidymodels)

# Load token preparation functions
source('functions/token_preparation.R')

# Suppress summarise info
options(dplyr.summarise.inform = FALSE)

# Load ICU admission and discharge information
icu.timestamps <- read.csv('../timestamps/ICU_adm_disch_timestamps.csv') %>%
  mutate(ICUAdmTimeStamp = as.POSIXct(ICUAdmTimeStamp,tz = 'GMT'),
         ICUDischTimeStamp = as.POSIXct(ICUDischTimeStamp,tz = 'GMT'))
study.GUPIs <- sort(unique(icu.timestamps$GUPI))

# Load window limit timestamps
from.adm.timestamps <- read.csv('../timestamps/from_adm_window_timestamps.csv') %>%
  mutate(TimeStampStart = as.POSIXct(TimeStampStart,tz = 'GMT'),
         TimeStampEnd = as.POSIXct(TimeStampEnd,tz = 'GMT'))

from.disch.timestamps <- read.csv('../timestamps/from_disch_window_timestamps.csv') %>%
  mutate(TimeStampStart = as.POSIXct(TimeStampStart,tz = 'GMT'),
         TimeStampEnd = as.POSIXct(TimeStampEnd,tz = 'GMT'))

# Load internal cross-validation folds
cv.folds <- read.csv('../cross_validation_splits.csv')

# Read user-defined argument from command line
array_task_id <- as.integer(commandArgs(trailingOnly=TRUE))

# Create dummy dataframe to encode absolute vs. differential strategy
tokenisation.strategy <- data.frame(key=1,STRATEGY=c('abs','diff'))

# Based on array task ID, get current repeat and fold
tokenisation_combos <- cv.folds %>%
  select(repeat.,fold) %>%
  distinct() %>%
  rename(REPEAT = repeat., FOLD = fold) %>%
  mutate(key = 1) %>%
  full_join(tokenisation.strategy,by='key') %>%
  select(-key)
curr.repeat <- tokenisation_combos$REPEAT[array_task_id]
curr.fold <- tokenisation_combos$FOLD[array_task_id]
curr.strategy <- tokenisation_combos$STRATEGY[array_task_id]

### II. Load and prepare formatted CENTER-TBI predictor tokens
# Load baseline numeric predictors
baseline <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/numeric_baseline_predictors.csv',
                     na.strings=c("NA","NaN",""," "))
er.labs <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/numeric_er_labs.csv',
                    na.strings=c("NA","NaN",""," "))
er.ct.imaging <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/numeric_er_ct_imaging.csv',
                          na.strings=c("NA","NaN",""," ")) %>%
  rename(BaselineERCTFrames = BaselineERFrames)
er.mr.imaging <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/numeric_er_mr_imaging.csv',
                          na.strings=c("NA","NaN",""," ")) %>%
  rename(BaselineERMRFrames = BaselineERFrames)
rpq.outcomes <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/numeric_rpq_outcomes.csv',
                         na.strings=c("NA","NaN",""," ")) %>%
  rename_with(~ paste0('Baseline',.x),.cols = -GUPI) %>%
  mutate(BaselineRPQDate = as.POSIXct(BaselineRPQDate,tz = 'GMT')) %>%
  left_join(icu.timestamps %>% select(GUPI,ICUDischTimeStamp),by = 'GUPI') %>%
  filter(BaselineRPQDate <= ICUDischTimeStamp) %>%
  select(-ICUDischTimeStamp)
goat.outcomes <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/numeric_goat_outcomes.csv',
                          na.strings=c("NA","NaN",""," ")) %>%
  rename_with(~ paste0('Baseline',.x),.cols = -GUPI) %>%
  mutate(BaselineGOATDate = as.POSIXct(BaselineGOATDate,tz = 'GMT')) %>%
  left_join(icu.timestamps %>% select(GUPI,ICUDischTimeStamp),by = 'GUPI') %>%
  filter(BaselineGOATDate <= ICUDischTimeStamp) %>%
  select(-ICUDischTimeStamp)

# Load timestamped, single-event numeric predictors
biomarkers <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/numeric_biomarkers.csv',
                       na.strings=c("NA","NaN",""," ")) %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT')) %>%
  left_join(icu.timestamps %>% select(GUPI,ICUDischTimeStamp),by = 'GUPI') %>%
  filter(Timestamp <= ICUDischTimeStamp) %>%
  select(-ICUDischTimeStamp)

central.haemostasis <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/numeric_central_haemostasis.csv',
                                na.strings=c("NA","NaN",""," ")) %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT')) %>%
  left_join(icu.timestamps %>% select(GUPI,ICUDischTimeStamp),by = 'GUPI') %>%
  filter(Timestamp <= ICUDischTimeStamp) %>%
  select(-ICUDischTimeStamp)
names(central.haemostasis)<-gsub("\\_","",names(central.haemostasis))

ct.imaging <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/numeric_ct_imaging.csv',
                       na.strings=c("NA","NaN",""," ")) %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT')) %>%
  left_join(icu.timestamps %>% select(GUPI,ICUDischTimeStamp),by = 'GUPI') %>%
  filter(Timestamp <= ICUDischTimeStamp) %>%
  select(-ICUDischTimeStamp) %>%
  rename(CTFrames = Frames)

dh.values <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/numeric_daily_hourly.csv',
                      na.strings=c("NA","NaN",""," ")) %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT')) %>%
  left_join(icu.timestamps %>% select(GUPI,ICUDischTimeStamp),by = 'GUPI') %>%
  filter(Timestamp <= ICUDischTimeStamp) %>%
  select(-ICUDischTimeStamp)

daily.TIL <-read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/numeric_daily_TIL.csv',
                     na.strings=c("NA","NaN",""," ")) %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT')) %>%
  left_join(icu.timestamps %>% select(GUPI,ICUDischTimeStamp),by = 'GUPI') %>%
  filter(Timestamp <= ICUDischTimeStamp) %>%
  select(-ICUDischTimeStamp)

labs <-read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/numeric_labs.csv',
                na.strings=c("NA","NaN",""," ")) %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT')) %>%
  left_join(icu.timestamps %>% select(GUPI,ICUDischTimeStamp),by = 'GUPI') %>%
  filter(Timestamp <= ICUDischTimeStamp) %>%
  select(-ICUDischTimeStamp)

mr.imaging <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/numeric_mr_imaging.csv',
                       na.strings=c("NA","NaN",""," ")) %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT')) %>%
  left_join(icu.timestamps %>% select(GUPI,ICUDischTimeStamp),by = 'GUPI') %>%
  filter(Timestamp <= ICUDischTimeStamp) %>%
  select(-ICUDischTimeStamp) %>%
  rename(MRFrames = Frames)

# Load dated, single-event numeric predictors
daily.vitals <-read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/numeric_daily_vitals.csv',
                        na.strings=c("NA","NaN",""," ")) %>%
  mutate(DVDate = as.POSIXct(DVDate,tz = 'GMT')) %>%
  left_join(icu.timestamps %>% select(GUPI,ICUDischTimeStamp),by = 'GUPI') %>%
  filter(DVDate <= ICUDischTimeStamp) %>%
  select(-ICUDischTimeStamp)

# Load timestamped, interval numeric predictors
surgeries.cranial <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/numeric_surgeries_cranial.csv',
                              na.strings=c("NA","NaN",""," ")) %>%
  mutate(StartTimeStamp = as.POSIXct(StartTimeStamp,tz = 'GMT'),
         EndTimeStamp = as.POSIXct(EndTimeStamp,tz = 'GMT')) %>%
  left_join(icu.timestamps %>% select(GUPI,ICUDischTimeStamp),by = 'GUPI') %>%
  filter(StartTimeStamp <= ICUDischTimeStamp) %>%
  select(-ICUDischTimeStamp)

### III. Convert formatted predictors to tokens for each repeated cross-validation partition
# Create directory to store cross-validation formatted tokens
dir.create('/home/sb2406/rds/hpc-work/tokens', showWarnings = F, recursive = T)

repeats <- unique(cv.folds$repeat.)
folds <- unique(cv.folds$fold)

# Print initialization statement of current partition
print(paste('Repeat',curr.repeat,'Fold',curr.fold,'initalized'))

curr.train.GUPIs <- cv.folds$GUPI[(cv.folds$repeat. == curr.repeat) & (cv.folds$fold == curr.fold) & (cv.folds$test_or_train == 'train')]
curr.test.GUPIs <- cv.folds$GUPI[(cv.folds$repeat. == curr.repeat) & (cv.folds$fold == curr.fold) & (cv.folds$test_or_train == 'test')]

# Define number of qunatile cuts
NUM.CUTS = 20

# Create directory to store final formatted tokens of the corresponding number of cuts
fold.dir = file.path('/home/sb2406/rds/hpc-work/tokens',sprintf('repeat%02.f',curr.repeat),sprintf('fold%01.f',curr.fold))
dir.create(fold.dir,showWarnings = F, recursive = T)

# Train cuts and transform fixed baseline numeric variables
tf.baseline <- tf_variables(baseline,build_recipe(baseline %>% filter(GUPI %in% curr.train.GUPIs),NUM.CUTS),NUM.CUTS) %>%
  categorical_tokenizer(startColIdx = 2,prefix = 'Baseline')
tf.baseline$Token <- do.call(paste, c(tf.baseline %>% select(-GUPI), sep=" "))
tf.baseline <- tf.baseline[,c('GUPI','Token')]

tf.er.labs <- tf_variables(er.labs,build_recipe(er.labs %>% filter(GUPI %in% curr.train.GUPIs),NUM.CUTS),NUM.CUTS) %>%
  categorical_tokenizer(startColIdx = 2) 
tf.er.labs$TokenStub <- do.call(paste, c(tf.er.labs %>% select(-GUPI), sep=" "))
tf.er.labs <- tf.er.labs %>%
  group_by(GUPI) %>%
  summarise(TokenStub = paste(TokenStub, collapse = " "))
tf.baseline <- tf.baseline %>%
  left_join(tf.er.labs,by = 'GUPI') %>%
  mutate(TokenStub = replace_na(TokenStub,'BaselineERLabs_NA')) %>%
  mutate(Token = paste(Token,TokenStub)) %>%
  select(-TokenStub)

tf.er.ct.imaging <- tf_variables(er.ct.imaging,build_recipe(er.ct.imaging %>% filter(GUPI %in% curr.train.GUPIs),NUM.CUTS),NUM.CUTS) %>%
  categorical_tokenizer(startColIdx = 2) 
tf.er.ct.imaging$TokenStub <- do.call(paste, c(tf.er.ct.imaging %>% select(-GUPI), sep=" "))
tf.er.ct.imaging <- tf.er.ct.imaging %>%
  group_by(GUPI) %>%
  summarise(TokenStub = paste(TokenStub, collapse = " "))
tf.baseline <- tf.baseline %>%
  left_join(tf.er.ct.imaging,by = 'GUPI') %>%
  mutate(TokenStub = replace_na(TokenStub,'BaselineERCTImaging_NA')) %>%
  mutate(Token = paste(Token,TokenStub)) %>%
  select(-TokenStub)  

tf.er.mr.imaging <- tf_variables(er.mr.imaging,build_recipe(er.mr.imaging %>% filter(GUPI %in% curr.train.GUPIs),NUM.CUTS),NUM.CUTS) %>%
  categorical_tokenizer(startColIdx = 2) 
tf.er.mr.imaging$TokenStub <- do.call(paste, c(tf.er.mr.imaging %>% select(-GUPI), sep=" "))
tf.er.mr.imaging <- tf.er.mr.imaging %>%
  group_by(GUPI) %>%
  summarise(TokenStub = paste(TokenStub, collapse = " "))
tf.baseline <- tf.baseline %>%
  left_join(tf.er.mr.imaging,by = 'GUPI') %>%
  mutate(TokenStub = replace_na(TokenStub,'BaselineERMRImaging_NA')) %>%
  mutate(Token = paste(Token,TokenStub)) %>%
  select(-TokenStub)   

# Train cuts and transform time-sensitive baseline numeric variables
tf.rpq.outcomes <- tf_variables(rpq.outcomes,build_recipe(rpq.outcomes %>% filter(GUPI %in% curr.train.GUPIs),NUM.CUTS),NUM.CUTS) %>%
  mutate(BaselineRPQDate = rpq.outcomes$BaselineRPQDate) %>%
  categorical_tokenizer(startColIdx = 3) %>%
  mutate(BaselineRPQDate = date(BaselineRPQDate))
tf.rpq.outcomes$TokenStub <- do.call(paste, c(tf.rpq.outcomes %>% select(-c(GUPI,BaselineRPQDate)), sep=" "))
tf.rpq.outcomes <- tf.rpq.outcomes %>%
  group_by(GUPI,BaselineRPQDate) %>%
  summarise(TokenStub = paste(TokenStub, collapse = " "))

tf.goat.outcomes <- tf_variables(goat.outcomes,build_recipe(goat.outcomes %>% filter(GUPI %in% curr.train.GUPIs),NUM.CUTS),NUM.CUTS) %>%
  mutate(BaselineGOATDate = goat.outcomes$BaselineGOATDate) %>%
  categorical_tokenizer(startColIdx = 3) %>%
  mutate(BaselineGOATDate = date(BaselineGOATDate))
tf.goat.outcomes$TokenStub <- do.call(paste, c(tf.goat.outcomes %>% select(-c(GUPI,BaselineGOATDate)), sep=" "))
tf.goat.outcomes <- tf.goat.outcomes %>%
  group_by(GUPI,BaselineGOATDate) %>%
  summarise(TokenStub = paste(TokenStub, collapse = " "))

# Train cuts and transform timestamped, single-event cuts
tf.biomarkers <- tf_variables(biomarkers,build_recipe(biomarkers %>% filter(GUPI %in% curr.train.GUPIs),NUM.CUTS),NUM.CUTS) %>%
  mutate(Timestamp = biomarkers$Timestamp) %>%
  categorical_tokenizer(startColIdx = 3) %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT'))

tf.central.haemostasis <- tf_variables(central.haemostasis,build_recipe(central.haemostasis %>% filter(GUPI %in% curr.train.GUPIs),NUM.CUTS),NUM.CUTS) %>%
  mutate(Timestamp = central.haemostasis$Timestamp) %>%
  categorical_tokenizer(startColIdx = 3) %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT'))

tf.ct.imaging <- tf_variables(ct.imaging,build_recipe(ct.imaging %>% filter(GUPI %in% curr.train.GUPIs),NUM.CUTS),NUM.CUTS) %>%
  mutate(Timestamp = ct.imaging$Timestamp) %>%
  categorical_tokenizer(startColIdx = 3) %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT'))  

tf.dh.values <- tf_variables(dh.values,build_recipe(dh.values %>% filter(GUPI %in% curr.train.GUPIs),NUM.CUTS),NUM.CUTS) %>%
  mutate(Timestamp = dh.values$Timestamp) %>%
  categorical_tokenizer(startColIdx = 3) %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT'))  

tf.daily.TIL <- tf_variables(daily.TIL,build_recipe(daily.TIL %>% filter(GUPI %in% curr.train.GUPIs),NUM.CUTS),NUM.CUTS) %>%
  mutate(Timestamp = daily.TIL$Timestamp) %>%
  categorical_tokenizer(startColIdx = 3) %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT'))    

tf.labs <- tf_variables(labs,build_recipe(labs %>% filter(GUPI %in% curr.train.GUPIs),NUM.CUTS),NUM.CUTS) %>%
  mutate(Timestamp = labs$Timestamp) %>%
  categorical_tokenizer(startColIdx = 3) %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT'))    

tf.mr.imaging <- tf_variables(mr.imaging,build_recipe(mr.imaging %>% filter(GUPI %in% curr.train.GUPIs),NUM.CUTS),NUM.CUTS) %>%
  mutate(Timestamp = mr.imaging$Timestamp) %>%
  categorical_tokenizer(startColIdx = 3) %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT'))   

# Load dated, single-event cuts
tf.daily.vitals <- tf_variables(daily.vitals,build_recipe(daily.vitals %>% filter(GUPI %in% curr.train.GUPIs),NUM.CUTS),NUM.CUTS) %>%
  mutate(DVDate = daily.vitals$DVDate) %>%
  categorical_tokenizer(startColIdx = 3) %>%
  mutate(DVDate = date(DVDate))

# Load timestamped, interval cuts
tf.surgeries.cranial <- tf_variables(surgeries.cranial,build_recipe(surgeries.cranial %>% filter(GUPI %in% curr.train.GUPIs),NUM.CUTS),NUM.CUTS) %>%
  mutate(StartTimeStamp = surgeries.cranial$StartTimeStamp,
         EndTimeStamp = surgeries.cranial$EndTimeStamp) %>%
  categorical_tokenizer(startColIdx = 4) %>%
  mutate(StartTimeStamp = as.POSIXct(StartTimeStamp,tz = 'GMT'),
         EndTimeStamp = as.POSIXct(EndTimeStamp,tz = 'GMT'))

# Iterate through patients in parallel
for (curr.GUPI in study.GUPIs){
  
  if (curr.strategy == 'abs'){
    
    # Load categorical tokens of current GUPI and append time of day and baseline indicators 
    curr.from.adm.tokens <- read.csv(file.path('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors',curr.GUPI,paste0('from_admission_categorical_',curr.strategy,'_tokens.csv'))) %>%
      mutate(TimeStampStart = as.POSIXct(TimeStampStart,tz = 'GMT'),
             TimeStampEnd = as.POSIXct(TimeStampEnd,tz = 'GMT')) %>%
      left_join(icu.timestamps %>% select(GUPI,ICUDischTimeStamp),by = 'GUPI') %>%
      filter(TimeStampEnd <= ICUDischTimeStamp) %>%
      select(-ICUDischTimeStamp) %>%
      mutate(BaselineToken = tf.baseline$Token[tf.baseline$GUPI == curr.GUPI]) %>%
      mutate(Token = paste(Token,BaselineToken)) %>%
      select(-c(BaselineToken)) %>%
      mutate(RPQToken = 'BaselineRPQ_NA',
             GOATToken = 'BaselineGOAT_NA')
    
    curr.from.disch.tokens <- read.csv(file.path('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors',curr.GUPI,paste0('from_discharge_categorical_',curr.strategy,'_tokens.csv'))) %>%
      mutate(TimeStampStart = as.POSIXct(TimeStampStart,tz = 'GMT'),
             TimeStampEnd = as.POSIXct(TimeStampEnd,tz = 'GMT')) %>%
      left_join(icu.timestamps %>% select(GUPI,ICUDischTimeStamp),by = 'GUPI') %>%
      filter(TimeStampEnd <= ICUDischTimeStamp) %>%
      select(-ICUDischTimeStamp) %>%
      mutate(BaselineToken = tf.baseline$Token[tf.baseline$GUPI == curr.GUPI]) %>%
      mutate(Token = paste(Token,BaselineToken)) %>%
      select(-c(BaselineToken)) %>%
      mutate(RPQToken = 'BaselineRPQ_NA',
             GOATToken = 'BaselineGOAT_NA')
    
    # Append tokens of time-sensitive baseline indicators
    if (curr.GUPI %in% tf.rpq.outcomes$GUPI){
      rpq.indices <- which(date(curr.from.adm.tokens$TimeStampEnd) >= date(tf.rpq.outcomes$BaselineRPQDate[tf.rpq.outcomes$GUPI == curr.GUPI]))
      curr.from.adm.tokens$RPQToken[rpq.indices] <- tf.rpq.outcomes$TokenStub[tf.rpq.outcomes$GUPI == curr.GUPI]
    } 
    
    if (curr.GUPI %in% tf.rpq.outcomes$GUPI){
      rpq.indices <- which(date(curr.from.disch.tokens$TimeStampEnd) >= date(tf.rpq.outcomes$BaselineRPQDate[tf.rpq.outcomes$GUPI == curr.GUPI]))
      curr.from.disch.tokens$RPQToken[rpq.indices] <- tf.rpq.outcomes$TokenStub[tf.rpq.outcomes$GUPI == curr.GUPI]
    } 
    
    if (curr.GUPI %in% tf.goat.outcomes$GUPI){
      goat.indices <- which(date(curr.from.adm.tokens$TimeStampEnd) >= date(tf.goat.outcomes$BaselineGOATDate[tf.goat.outcomes$GUPI == curr.GUPI]))
      curr.from.adm.tokens$GOATToken[goat.indices] <- tf.goat.outcomes$TokenStub[tf.goat.outcomes$GUPI == curr.GUPI]
    }
    
    if (curr.GUPI %in% tf.goat.outcomes$GUPI){
      goat.indices <- which(date(curr.from.disch.tokens$TimeStampEnd) >= date(tf.goat.outcomes$BaselineGOATDate[tf.goat.outcomes$GUPI == curr.GUPI]))
      curr.from.disch.tokens$GOATToken[goat.indices] <- tf.goat.outcomes$TokenStub[tf.goat.outcomes$GUPI == curr.GUPI]
    }
    
    curr.from.adm.tokens <- curr.from.adm.tokens %>%
      mutate(Token = paste(Token,RPQToken,GOATToken)) %>%
      select(-c(RPQToken,GOATToken))
    
    curr.from.disch.tokens <- curr.from.disch.tokens %>%
      mutate(Token = paste(Token,RPQToken,GOATToken)) %>%
      select(-c(RPQToken,GOATToken))
    
    # Add timestamped single events from admission
    curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,tf.biomarkers,curr.GUPI)
    curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,tf.central.haemostasis,curr.GUPI)
    curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,tf.ct.imaging,curr.GUPI)
    curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,tf.dh.values,curr.GUPI)
    curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,tf.daily.TIL,curr.GUPI)
    curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,tf.labs,curr.GUPI)
    curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,tf.mr.imaging,curr.GUPI)
    
    # Add timestamped single events from discharge
    curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,tf.biomarkers,curr.GUPI)
    curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,tf.central.haemostasis,curr.GUPI)
    curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,tf.ct.imaging,curr.GUPI)
    curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,tf.dh.values,curr.GUPI)
    curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,tf.daily.TIL,curr.GUPI)
    curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,tf.labs,curr.GUPI)
    curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,tf.mr.imaging,curr.GUPI)
    
    # Add dated single events from admission
    curr.from.adm.tokens <- add.date.event.tokens(curr.from.adm.tokens,tf.daily.vitals %>% rename(Date = DVDate),curr.GUPI)
    
    # Add dated single events from discharge
    curr.from.disch.tokens <- add.date.event.tokens(curr.from.disch.tokens,tf.daily.vitals %>% rename(Date = DVDate),curr.GUPI)
    
    # Add timestamped interval variables from admission
    curr.from.adm.tokens <- add.timestamp.interval.tokens(curr.from.adm.tokens,
                                                          tf.surgeries.cranial %>% rename(StartTimestamp = StartTimeStamp, StopTimestamp = EndTimeStamp),
                                                          c(''),
                                                          curr.GUPI)  
    
    # Add timestamped interval variables from discharge
    curr.from.disch.tokens <- add.timestamp.interval.tokens(curr.from.disch.tokens,
                                                            tf.surgeries.cranial %>% rename(StartTimestamp = StartTimeStamp, StopTimestamp = EndTimeStamp),
                                                            c(''),
                                                            curr.GUPI)  
    
    # Ensure all tokens are unique per study window
    curr.from.adm.tokens <- curr.from.adm.tokens %>%
      rowwise() %>%
      mutate(Token = paste(unique(unlist(strsplit(Token,split =' '))),collapse = ' '))
    
    curr.from.disch.tokens <- curr.from.disch.tokens %>%
      rowwise() %>%
      mutate(Token = paste(unique(unlist(strsplit(Token,split =' '))),collapse = ' '))
    
    # Save formatted tokens of the current patient
    write.csv(curr.from.adm.tokens,
              file.path(fold.dir,paste0('from_adm_GUPI_',curr.GUPI,'_strategy_',curr.strategy,'.csv')),
              row.names = F)
    
    write.csv(curr.from.disch.tokens,
              file.path(fold.dir,paste0('from_disch_GUPI_',curr.GUPI,'_strategy_',curr.strategy,'.csv')),
              row.names = F)
    
  } else if (curr.strategy == 'diff'){
    
    # Load categorical tokens of current GUPI and append time of day and baseline indicators 
    curr.from.adm.tokens <- read.csv(file.path('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors',curr.GUPI,paste0('from_admission_categorical_',curr.strategy,'_tokens.csv'))) %>%
      mutate(TimeStampStart = as.POSIXct(TimeStampStart,tz = 'GMT'),
             TimeStampEnd = as.POSIXct(TimeStampEnd,tz = 'GMT')) %>%
      left_join(icu.timestamps %>% select(GUPI,ICUDischTimeStamp),by = 'GUPI') %>%
      filter(TimeStampEnd <= ICUDischTimeStamp) %>%
      select(-ICUDischTimeStamp) %>%
      mutate(RPQToken = 'BaselineRPQ_NA',
             GOATToken = 'BaselineGOAT_NA')
    
    earliest.window.idx <- curr.from.adm.tokens$WindowIdx[curr.from.adm.tokens$TimeStampStart == min(curr.from.adm.tokens$TimeStampStart)]
    curr.from.adm.tokens$Token[curr.from.adm.tokens$WindowIdx == earliest.window.idx] <- paste(curr.from.adm.tokens$Token[curr.from.adm.tokens$WindowIdx == earliest.window.idx],tf.baseline$Token[tf.baseline$GUPI == curr.GUPI])
    
    curr.from.disch.tokens <- read.csv(file.path('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors',curr.GUPI,paste0('from_discharge_categorical_',curr.strategy,'_tokens.csv'))) %>%
      mutate(TimeStampStart = as.POSIXct(TimeStampStart,tz = 'GMT'),
             TimeStampEnd = as.POSIXct(TimeStampEnd,tz = 'GMT')) %>%
      left_join(icu.timestamps %>% select(GUPI,ICUDischTimeStamp),by = 'GUPI') %>%
      filter(TimeStampEnd <= ICUDischTimeStamp) %>%
      select(-ICUDischTimeStamp) %>%
      mutate(RPQToken = 'BaselineRPQ_NA',
             GOATToken = 'BaselineGOAT_NA')
    
    earliest.window.idx <- curr.from.disch.tokens$WindowIdx[curr.from.disch.tokens$TimeStampStart == min(curr.from.disch.tokens$TimeStampStart)]
    curr.from.disch.tokens$Token[curr.from.disch.tokens$WindowIdx == earliest.window.idx] <- paste(curr.from.disch.tokens$Token[curr.from.disch.tokens$WindowIdx == earliest.window.idx],tf.baseline$Token[tf.baseline$GUPI == curr.GUPI])
    
    # Append tokens of time-sensitive baseline indicators
    if (curr.GUPI %in% tf.rpq.outcomes$GUPI){
      rpq.indices <- which(date(curr.from.adm.tokens$TimeStampEnd) >= date(tf.rpq.outcomes$BaselineRPQDate[tf.rpq.outcomes$GUPI == curr.GUPI]))
      curr.from.adm.tokens$RPQToken[rpq.indices] <- tf.rpq.outcomes$TokenStub[tf.rpq.outcomes$GUPI == curr.GUPI]
    } 
    
    if (curr.GUPI %in% tf.rpq.outcomes$GUPI){
      rpq.indices <- which(date(curr.from.disch.tokens$TimeStampEnd) >= date(tf.rpq.outcomes$BaselineRPQDate[tf.rpq.outcomes$GUPI == curr.GUPI]))
      curr.from.disch.tokens$RPQToken[rpq.indices] <- tf.rpq.outcomes$TokenStub[tf.rpq.outcomes$GUPI == curr.GUPI]
    } 
    
    if (curr.GUPI %in% tf.goat.outcomes$GUPI){
      goat.indices <- which(date(curr.from.adm.tokens$TimeStampEnd) >= date(tf.goat.outcomes$BaselineGOATDate[tf.goat.outcomes$GUPI == curr.GUPI]))
      curr.from.adm.tokens$GOATToken[goat.indices] <- tf.goat.outcomes$TokenStub[tf.goat.outcomes$GUPI == curr.GUPI]
    }
    
    if (curr.GUPI %in% tf.goat.outcomes$GUPI){
      goat.indices <- which(date(curr.from.disch.tokens$TimeStampEnd) >= date(tf.goat.outcomes$BaselineGOATDate[tf.goat.outcomes$GUPI == curr.GUPI]))
      curr.from.disch.tokens$GOATToken[goat.indices] <- tf.goat.outcomes$TokenStub[tf.goat.outcomes$GUPI == curr.GUPI]
    }
    
    curr.from.adm.tokens <- curr.from.adm.tokens %>%
      mutate(Token = paste(Token,RPQToken,GOATToken)) %>%
      select(-c(RPQToken,GOATToken))
    
    curr.from.disch.tokens <- curr.from.disch.tokens %>%
      mutate(Token = paste(Token,RPQToken,GOATToken)) %>%
      select(-c(RPQToken,GOATToken))
    
    # Add timestamped single events from admission
    curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,tf.biomarkers,curr.GUPI)
    curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,tf.central.haemostasis,curr.GUPI)
    curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,tf.ct.imaging,curr.GUPI)
    curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,tf.dh.values,curr.GUPI)
    curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,tf.daily.TIL,curr.GUPI)
    curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,tf.labs,curr.GUPI)
    curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,tf.mr.imaging,curr.GUPI)
    
    # Add timestamped single events from discharge
    curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,tf.biomarkers,curr.GUPI)
    curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,tf.central.haemostasis,curr.GUPI)
    curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,tf.ct.imaging,curr.GUPI)
    curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,tf.dh.values,curr.GUPI)
    curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,tf.daily.TIL,curr.GUPI)
    curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,tf.labs,curr.GUPI)
    curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,tf.mr.imaging,curr.GUPI)
    
    # Add dated single events from admission
    curr.from.adm.tokens <- add.date.event.tokens(curr.from.adm.tokens,tf.daily.vitals %>% rename(Date = DVDate),curr.GUPI)
    
    # Add dated single events from discharge
    curr.from.disch.tokens <- add.date.event.tokens(curr.from.disch.tokens,tf.daily.vitals %>% rename(Date = DVDate),curr.GUPI)
    
    # Add timestamped interval variables from admission
    curr.from.adm.tokens <- add.timestamp.interval.tokens(curr.from.adm.tokens,
                                                          tf.surgeries.cranial %>% rename(StartTimestamp = StartTimeStamp, StopTimestamp = EndTimeStamp),
                                                          c(''),
                                                          curr.GUPI)  
    
    # Add timestamped interval variables from discharge
    curr.from.disch.tokens <- add.timestamp.interval.tokens(curr.from.disch.tokens,
                                                            tf.surgeries.cranial %>% rename(StartTimestamp = StartTimeStamp, StopTimestamp = EndTimeStamp),
                                                            c(''),
                                                            curr.GUPI)  
    
    # Ensure all tokens are unique per study window
    curr.from.adm.tokens <- curr.from.adm.tokens %>%
      rowwise() %>%
      mutate(Token = str_trim(paste(unique(unlist(strsplit(Token,split =' '))),collapse = ' '))) %>%
      differentiate.tokens(adm.or.disch = 'adm')
    
    curr.from.disch.tokens <- curr.from.disch.tokens %>%
      rowwise() %>%
      mutate(Token = str_trim(paste(unique(unlist(strsplit(Token,split =' '))),collapse = ' '))) %>%
      differentiate.tokens(adm.or.disch = 'disch')
    
    # Save formatted tokens of the current patient
    write.csv(curr.from.adm.tokens,
              file.path(fold.dir,paste0('from_adm_GUPI_',curr.GUPI,'_strategy_',curr.strategy,'.csv')),
              row.names = F)
    
    write.csv(curr.from.disch.tokens,
              file.path(fold.dir,paste0('from_disch_GUPI_',curr.GUPI,'_strategy_',curr.strategy,'.csv')),
              row.names = F)
    
  }
}