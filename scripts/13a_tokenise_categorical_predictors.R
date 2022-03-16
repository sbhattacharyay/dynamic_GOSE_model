#### Master Script 13a: Tokenize categorical predictor tokens from CENTER-TBI dataset ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Prepare categorical CENTER-TBI predictors into tokens

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

# Set the number of parallel cores
NUM.CORES <- detectCores() - 2

# Initialize local cluster for parallel processing
registerDoParallel(cores = NUM.CORES)

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

### II. Prepare categorical CENTER-TBI predictors into tokens
## Load non-baseline categorical predictors
# Load timestamp variables
ct.imaging <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_ct_imaging.csv') %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT'))
mr.imaging <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_mr_imaging.csv') %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT'))
dh.values <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_daily_hourly.csv') %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT'))
daily.TIL <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_daily_TIL.csv') %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT'))
icp.catheter <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_icp_catheter.csv') %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT'))
labs <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_labs.csv') %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT'))
reintubation <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_reintubation.csv') %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT'))
remech.vent <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_remech_ventilation.csv') %>%
  mutate(Timestamp = as.POSIXct(Timestamp,tz = 'GMT'))

# Load date-based, single-event variables
daily.vitals <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_daily_vitals.csv') %>%
  mutate(DVDate = as.POSIXct(DVDate,tz = 'GMT'))
toc <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_transitions_of_care.csv') %>%
  mutate(DateEffectiveTransfer = as.POSIXct(DateEffectiveTransfer,tz = 'GMT'))

# Load timestamped interval variables
icp.monitoring <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_icp_monitoring.csv') %>%
  mutate(ICPInsTimestamp = as.POSIXct(ICPInsTimestamp,tz = 'GMT'),
         ICPRemTimestamp = as.POSIXct(ICPRemTimestamp,tz = 'GMT')) %>%
  select(-ICUCatheterICP)
intubation <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_intubation.csv') %>%
  mutate(IntubationStartTimestamp = as.POSIXct(IntubationStartTimestamp,tz = 'GMT'),
         IntubationStopTimestamp = as.POSIXct(IntubationStopTimestamp,tz = 'GMT'))
mech.vent <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_mech_ventilation.csv') %>%
  mutate(MechVentilationStartTimestamp = as.POSIXct(MechVentilationStartTimestamp,tz = 'GMT'),
         MechVentilationStopTimestamp = as.POSIXct(MechVentilationStopTimestamp,tz = 'GMT')) %>%
  mutate(MechVentilationStop = 'MechVentilationStop_1')
surgeries.cranial <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_surgeries_cranial.csv') %>%
  mutate(StartTimeStamp = as.POSIXct(StartTimeStamp,tz = 'GMT'),
         EndTimeStamp = as.POSIXct(EndTimeStamp,tz = 'GMT')) %>%
  mutate(SurgeryCranialStop = 'SurgeryCranialStop_1')
surgeries.extra.cranial <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_surgeries_extra_cranial.csv') %>%
  mutate(StartTimeStamp = as.POSIXct(StartTimeStamp,tz = 'GMT'),
         EndTimeStamp = as.POSIXct(EndTimeStamp,tz = 'GMT')) %>%
  mutate(SurgeryExtraCranialStop = 'SurgeryExtraCranialStop_1')

# Load dated interval variables
dvt.mech <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_dvt_mech.csv') %>%
  mutate(DVTProphylaxisMechStartDate = as.POSIXct(DVTProphylaxisMechStartDate,tz = 'GMT'),
         DVTProphylaxisMechStopDate = as.POSIXct(DVTProphylaxisMechStopDate,tz = 'GMT')) %>%
  mutate(DVTProphylaxisMechStop = 'DVTProphylaxisMechStop_1')
dvt.pharm <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_dvt_pharm.csv') %>%
  mutate(DVTProphylaxisStartDate = as.POSIXct(DVTProphylaxisStartDate,tz = 'GMT'),
         DVTProphylaxisStopDate = as.POSIXct(DVTProphylaxisStopDate,tz = 'GMT')) %>%
  mutate(DVTProphylaxisStop = 'DVTProphylaxisStop_1')
enteral <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_enteral_nutrition.csv') %>%
  mutate(EnteralNutritionStartDate = as.POSIXct(EnteralNutritionStartDate,tz = 'GMT'),
         EnteralNutritionStopDate = as.POSIXct(EnteralNutritionStopDate,tz = 'GMT')) %>%
  mutate(EnteralNutritionStop = 'EnteralNutritionStop_1')
parenteral <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_parenteral_nutrition.csv') %>%
  mutate(ParenteralNutritionStartDate = as.POSIXct(ParenteralNutritionStartDate,tz = 'GMT'),
         ParenteralNutritionStopDate = as.POSIXct(ParenteralNutritionStopDate,tz = 'GMT')) %>%
  mutate(ParenteralNutritionStop = 'ParenteralNutritionStop_1')
meds <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_meds.csv') %>%
  mutate(StartDate = as.POSIXct(StartDate,tz = 'GMT'),
         StopDate = as.POSIXct(StopDate,tz = 'GMT')) %>%
  mutate(MedicationStop = 'MedicationStop_1')
nasogastric <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_nasogastric.csv') %>%
  mutate(NasogastricStartDate = as.POSIXct(NasogastricStartDate,tz = 'GMT'),
         NasogastricStopDate = as.POSIXct(NasogastricStopDate,tz = 'GMT')) %>%
  mutate(NasogastricStop = 'NasogastricStop_1')
oxy.admin <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_oxygen_administration.csv') %>%
  mutate(OxygenAdmStartDate = as.POSIXct(OxygenAdmStartDate,tz = 'GMT'),
         OxygenAdmStopDate = as.POSIXct(OxygenAdmStopDate,tz = 'GMT')) %>%
  mutate(OxygenAdmStop = 'OxygenAdmStop_1')
peg.tube <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_peg_tube.csv') %>%
  mutate(PEGTubeStartDate = as.POSIXct(PEGTubeStartDate,tz = 'GMT'),
         PEGTubeStopDate = as.POSIXct(PEGTubeStopDate,tz = 'GMT')) %>%
  mutate(PEGTubeStop = 'PEGTubeStop_1')
tracheostomy <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_tracheostomy.csv') %>%
  mutate(TracheostomyStartDate = as.POSIXct(TracheostomyStartDate,tz = 'GMT'),
         TracheostomyStopDate = as.POSIXct(TracheostomyStopDate,tz = 'GMT')) %>%
  mutate(TracheostomyStop = 'TracheostomyStop_1')
urine.cath <- read.csv('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors/categorical_urine_catheter.csv') %>%
  mutate(UrineCathStartDate = as.POSIXct(UrineCathStartDate,tz = 'GMT'),
         UrineCathStopDate = as.POSIXct(UrineCathStopDate,tz = 'GMT')) %>%
  mutate(UrineCathStop = 'UrineCathStop_1')

#In parallel, iterate through each GUPI and format categorical predictor tokens 
foreach(curr.GUPI = study.GUPIs,.inorder = F) %dopar%{
  
  # Load baseline categorical predictors of current GUPI
  curr.baseline.categorical.predictors <- read.csv(file.path('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors',curr.GUPI,'categorical_baseline_predictors.csv'))
  curr.baseline.categorical.predictors <- curr.baseline.categorical.predictors$Token[1]
  
  # Create token dataframes for both from-admission indexing and from-discharge indexing
  curr.from.adm.tokens <- from.adm.timestamps %>%
    filter(GUPI == curr.GUPI) %>%
    mutate(Token = '')
  
  curr.from.disch.tokens <- from.disch.timestamps %>%
    filter(GUPI == curr.GUPI) %>%
    mutate(Token = '')
  
  # Add baseline tokens to first window
  curr.from.adm.tokens$Token[curr.from.adm.tokens$TimeStampStart == min(curr.from.adm.tokens$TimeStampStart)] <- curr.baseline.categorical.predictors
  curr.from.disch.tokens$Token[curr.from.disch.tokens$TimeStampStart == min(curr.from.disch.tokens$TimeStampStart)] <- curr.baseline.categorical.predictors
  
  # Add timestamped single events from admission
  curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,ct.imaging,curr.GUPI)
  curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,mr.imaging,curr.GUPI)
  curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,dh.values,curr.GUPI)
  curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,daily.TIL,curr.GUPI)
  curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,icp.catheter,curr.GUPI)
  curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,labs,curr.GUPI)
  curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,reintubation,curr.GUPI)
  curr.from.adm.tokens <- add.timestamp.event.tokens(curr.from.adm.tokens,remech.vent,curr.GUPI)
  
  # Add timestamped single events from discharge
  curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,ct.imaging,curr.GUPI)
  curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,mr.imaging,curr.GUPI)
  curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,dh.values,curr.GUPI)
  curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,daily.TIL,curr.GUPI)
  curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,icp.catheter,curr.GUPI)
  curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,labs,curr.GUPI)
  curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,reintubation,curr.GUPI)
  curr.from.disch.tokens <- add.timestamp.event.tokens(curr.from.disch.tokens,remech.vent,curr.GUPI)
  
  # Add dated single events from admission
  curr.from.adm.tokens <- add.date.event.tokens(curr.from.adm.tokens,daily.vitals %>% rename(Date = DVDate),curr.GUPI)
  curr.from.adm.tokens <- add.date.event.tokens(curr.from.adm.tokens,toc %>% rename(Date = DateEffectiveTransfer),curr.GUPI)
  
  # Add dated single events from discharge
  curr.from.disch.tokens <- add.date.event.tokens(curr.from.disch.tokens,daily.vitals %>% rename(Date = DVDate),curr.GUPI)
  curr.from.disch.tokens <- add.date.event.tokens(curr.from.disch.tokens,toc %>% rename(Date = DateEffectiveTransfer),curr.GUPI)
  
  # Add timestamped interval variables from admission
  curr.from.adm.tokens <- add.timestamp.interval.tokens(curr.from.adm.tokens,
                                                        icp.monitoring %>% rename(StartTimestamp = ICPInsTimestamp, StopTimestamp = ICPRemTimestamp),
                                                        c('ICPMonitorStop','ICPMonitorStopReasonOther'),
                                                        curr.GUPI)
  curr.from.adm.tokens <- add.timestamp.interval.tokens(curr.from.adm.tokens,
                                                        intubation %>% rename(StartTimestamp = IntubationStartTimestamp, StopTimestamp = IntubationStopTimestamp),
                                                        c('IntubationStopReason'),
                                                        curr.GUPI)
  curr.from.adm.tokens <- add.timestamp.interval.tokens(curr.from.adm.tokens,
                                                        mech.vent %>% rename(StartTimestamp = MechVentilationStartTimestamp, StopTimestamp = MechVentilationStopTimestamp),
                                                        c('MechVentilationStop'),
                                                        curr.GUPI)
  curr.from.adm.tokens <- add.timestamp.interval.tokens(curr.from.adm.tokens,
                                                        surgeries.cranial %>% rename(StartTimestamp = StartTimeStamp, StopTimestamp = EndTimeStamp),
                                                        c('SurgeryCranialStop'),
                                                        curr.GUPI)
  curr.from.adm.tokens <- add.timestamp.interval.tokens(curr.from.adm.tokens,
                                                        surgeries.extra.cranial %>% rename(StartTimestamp = StartTimeStamp, StopTimestamp = EndTimeStamp),
                                                        c('SurgeryExtraCranialStop'),
                                                        curr.GUPI)
  
  # Add timestamped interval variables from discharge
  curr.from.disch.tokens <- add.timestamp.interval.tokens(curr.from.disch.tokens,
                                                          icp.monitoring %>% rename(StartTimestamp = ICPInsTimestamp, StopTimestamp = ICPRemTimestamp),
                                                          c('ICPMonitorStop','ICPMonitorStopReasonOther'),
                                                          curr.GUPI)
  curr.from.disch.tokens <- add.timestamp.interval.tokens(curr.from.disch.tokens,
                                                          intubation %>% rename(StartTimestamp = IntubationStartTimestamp, StopTimestamp = IntubationStopTimestamp),
                                                          c('IntubationStopReason'),
                                                          curr.GUPI)
  curr.from.disch.tokens <- add.timestamp.interval.tokens(curr.from.disch.tokens,
                                                          mech.vent %>% rename(StartTimestamp = MechVentilationStartTimestamp, StopTimestamp = MechVentilationStopTimestamp),
                                                          c('MechVentilationStop'),
                                                          curr.GUPI)
  curr.from.disch.tokens <- add.timestamp.interval.tokens(curr.from.disch.tokens,
                                                          surgeries.cranial %>% rename(StartTimestamp = StartTimeStamp, StopTimestamp = EndTimeStamp),
                                                          c('SurgeryCranialStop'),
                                                          curr.GUPI)
  curr.from.disch.tokens <- add.timestamp.interval.tokens(curr.from.disch.tokens,
                                                          surgeries.extra.cranial %>% rename(StartTimestamp = StartTimeStamp, StopTimestamp = EndTimeStamp),
                                                          c('SurgeryExtraCranialStop'),
                                                          curr.GUPI)
  
  # Add dated interval variables from admission
  curr.from.adm.tokens <- add.date.interval.tokens(curr.from.adm.tokens,
                                                   dvt.mech %>% rename(StartDate = DVTProphylaxisMechStartDate,StopDate = DVTProphylaxisMechStopDate),
                                                   c('DVTProphylaxisMechStop'),
                                                   curr.GUPI)
  curr.from.adm.tokens <- add.date.interval.tokens(curr.from.adm.tokens,
                                                   dvt.pharm %>% rename(StartDate = DVTProphylaxisStartDate,StopDate = DVTProphylaxisStopDate),
                                                   c('DVTProphylaxisStop'),
                                                   curr.GUPI)
  curr.from.adm.tokens <- add.date.interval.tokens(curr.from.adm.tokens,
                                                   enteral %>% rename(StartDate = EnteralNutritionStartDate,StopDate = EnteralNutritionStopDate),
                                                   c('EnteralNutritionStop'),
                                                   curr.GUPI)
  curr.from.adm.tokens <- add.date.interval.tokens(curr.from.adm.tokens,
                                                   parenteral %>% rename(StartDate = ParenteralNutritionStartDate,StopDate = ParenteralNutritionStopDate),
                                                   c('ParenteralNutritionStop'),
                                                   curr.GUPI)  
  curr.from.adm.tokens <- add.date.interval.tokens(curr.from.adm.tokens,
                                                   meds,
                                                   c('MedicationStop'),
                                                   curr.GUPI)  
  curr.from.adm.tokens <- add.date.interval.tokens(curr.from.adm.tokens,
                                                   nasogastric %>% rename(StartDate = NasogastricStartDate,StopDate = NasogastricStopDate),
                                                   c('NasogastricStop'),
                                                   curr.GUPI)  
  curr.from.adm.tokens <- add.date.interval.tokens(curr.from.adm.tokens,
                                                   oxy.admin %>% rename(StartDate = OxygenAdmStartDate,StopDate = OxygenAdmStopDate),
                                                   c('OxygenAdmStop'),
                                                   curr.GUPI)  
  curr.from.adm.tokens <- add.date.interval.tokens(curr.from.adm.tokens,
                                                   peg.tube %>% rename(StartDate = PEGTubeStartDate,StopDate = PEGTubeStopDate),
                                                   c('PEGTubeStop'),
                                                   curr.GUPI)  
  curr.from.adm.tokens <- add.date.interval.tokens(curr.from.adm.tokens,
                                                   tracheostomy %>% rename(StartDate = TracheostomyStartDate,StopDate = TracheostomyStopDate),
                                                   c('TracheostomyStop'),
                                                   curr.GUPI)  
  curr.from.adm.tokens <- add.date.interval.tokens(curr.from.adm.tokens,
                                                   urine.cath %>% rename(StartDate = UrineCathStartDate,StopDate = UrineCathStopDate),
                                                   c('UrineCathStop'),
                                                   curr.GUPI)
  
  # Add dated interval variables from discharge
  curr.from.disch.tokens <- add.date.interval.tokens(curr.from.disch.tokens,
                                                     dvt.mech %>% rename(StartDate = DVTProphylaxisMechStartDate,StopDate = DVTProphylaxisMechStopDate),
                                                     c('DVTProphylaxisMechStop'),
                                                     curr.GUPI)
  curr.from.disch.tokens <- add.date.interval.tokens(curr.from.disch.tokens,
                                                     dvt.pharm %>% rename(StartDate = DVTProphylaxisStartDate,StopDate = DVTProphylaxisStopDate),
                                                     c('DVTProphylaxisStop'),
                                                     curr.GUPI)
  curr.from.disch.tokens <- add.date.interval.tokens(curr.from.disch.tokens,
                                                     enteral %>% rename(StartDate = EnteralNutritionStartDate,StopDate = EnteralNutritionStopDate),
                                                     c('EnteralNutritionStop'),
                                                     curr.GUPI)
  curr.from.disch.tokens <- add.date.interval.tokens(curr.from.disch.tokens,
                                                     parenteral %>% rename(StartDate = ParenteralNutritionStartDate,StopDate = ParenteralNutritionStopDate),
                                                     c('ParenteralNutritionStop'),
                                                     curr.GUPI)  
  curr.from.disch.tokens <- add.date.interval.tokens(curr.from.disch.tokens,
                                                     meds,
                                                     c('MedicationStop'),
                                                     curr.GUPI)  
  curr.from.disch.tokens <- add.date.interval.tokens(curr.from.disch.tokens,
                                                     nasogastric %>% rename(StartDate = NasogastricStartDate,StopDate = NasogastricStopDate),
                                                     c('NasogastricStop'),
                                                     curr.GUPI)  
  curr.from.disch.tokens <- add.date.interval.tokens(curr.from.disch.tokens,
                                                     oxy.admin %>% rename(StartDate = OxygenAdmStartDate,StopDate = OxygenAdmStopDate),
                                                     c('OxygenAdmStop'),
                                                     curr.GUPI)  
  curr.from.disch.tokens <- add.date.interval.tokens(curr.from.disch.tokens,
                                                     peg.tube %>% rename(StartDate = PEGTubeStartDate,StopDate = PEGTubeStopDate),
                                                     c('PEGTubeStop'),
                                                     curr.GUPI)  
  curr.from.disch.tokens <- add.date.interval.tokens(curr.from.disch.tokens,
                                                     tracheostomy %>% rename(StartDate = TracheostomyStartDate,StopDate = TracheostomyStopDate),
                                                     c('TracheostomyStop'),
                                                     curr.GUPI)  
  curr.from.disch.tokens <- add.date.interval.tokens(curr.from.disch.tokens,
                                                     urine.cath %>% rename(StartDate = UrineCathStartDate,StopDate = UrineCathStopDate),
                                                     c('UrineCathStop'),
                                                     curr.GUPI)
  
  # Ensure each window only contains unique tokens and convert token dataframes to differential token dataframes
  curr.from.adm.tokens <- curr.from.adm.tokens %>%
    rowwise() %>%
    mutate(Token = str_trim(paste(unique(unlist(strsplit(Token,split =' '))),collapse = ' '))) %>%
    differentiate.tokens(adm.or.disch = 'adm')
  
  curr.from.disch.tokens <- curr.from.disch.tokens %>%
    rowwise() %>%
    mutate(Token = str_trim(paste(unique(unlist(strsplit(Token,split =' '))),collapse = ' '))) %>%
    differentiate.tokens(adm.or.disch = 'disch')
  
  # Save formatted categorical tokens into patient-specific directory
  write.csv(curr.from.adm.tokens,
            file.path('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors',curr.GUPI,'from_admission_categorical_tokens.csv'),
            row.names = F)
  
  write.csv(curr.from.disch.tokens,
            file.path('/home/sb2406/rds/hpc-work/CENTER-TBI/formatted_predictors',curr.GUPI,'from_discharge_categorical_tokens.csv'),
            row.names = F)
}