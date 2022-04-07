#### Master Script ##: Visualise study results for manuscript ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation

### I. Initialisation
# Import necessary libraries
library(tidyverse)
library(readxl)
library(RColorBrewer)
library(rvg)
library(svglite)
library(viridis)
library(lemon)
library(VIM)
library(latex2exp)
library(cowplot)

###
# Load token characteristics and summarise across timepoints from admission
adm.token.characteristics <- read.csv('../tokens/repeat01/fold1/from_adm_token_characteristics.csv') %>%
  mutate(DynamicTokens = TotalTokens - Baseline) %>%
  group_by(WindowIdx) %>%
  summarise(meanDynamicTokens = mean(DynamicTokens,na.rm=T),
            sdDynamicTokens = sd(DynamicTokens,na.rm=T),
            countPatients = n()) %>%
  rowwise() %>%
  mutate(lowerBound = max(meanDynamicTokens - sdDynamicTokens,0),
         upperBound = meanDynamicTokens + sdDynamicTokens,
         DaysAfterICUAdmission = WindowIdx/12)

# Create plot of number of non-missing dynamic tokens per patient
dynamic.token.plot <- ggplot(adm.token.characteristics,aes(x=DaysAfterICUAdmission,y=meanDynamicTokens)) +
  geom_line(color='red', alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(ymin = lowerBound, ymax = upperBound),alpha=.2,fill='red',size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,30),ylim=c(0,50)) + 
  scale_x_continuous(breaks=seq(0,30,by=1),expand = expansion(mult = c(0, .01)))+
  scale_y_continuous(breaks=seq(0,75,by=25))+
  geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab(expression(bold(paste('Dynamic tokens per patient (', italic('n'), ')'))))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

# Create plot of proportion of patients remaining after certain point in ICU stay
patient.count.plot <- ggplot(adm.token.characteristics,aes(x=DaysAfterICUAdmission,y=countPatients/1550)) +
  geom_line(color='blue',alpha = 1, size=1.3/.pt)+
  coord_cartesian(xlim=c(0,30)) + 
  scale_x_continuous(breaks=seq(0,30,by=1),expand = expansion(mult = c(0, .01))) +
  ylab(expression(bold(paste('Patients remaining (', italic('p'), ')'))))+
  xlab('Days from ICU admission')+
  geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

# Vertically align admission token characterisation plots
adm.token.counts <- plot_grid(dynamic.token.plot, patient.count.plot, ncol = 1, align = 'v', rel_heights = c(2,1))

# Create directory for current date and save admission characteristic plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'admission_tokens.svg'),adm.token.counts,device= svg,units='in',dpi=600,width=15,height = 10)

###
# Load token characteristics and summarise across timepoints from discharge
disch.token.characteristics <- read.csv('../tokens/repeat01/fold1/from_disch_token_characteristics.csv') %>%
  mutate(DynamicTokens = TotalTokens - Baseline) %>%
  group_by(WindowIdx) %>%
  summarise(meanDynamicTokens = mean(DynamicTokens,na.rm=T),
            sdDynamicTokens = sd(DynamicTokens,na.rm=T),
            countPatients = n()) %>%
  rowwise() %>%
  mutate(lowerBound = max(meanDynamicTokens - sdDynamicTokens,0),
         upperBound = meanDynamicTokens + sdDynamicTokens,
         DaysBeforeICUDischarge = WindowIdx/12)

# Create plot of number of non-missing dynamic tokens per patient
dynamic.token.plot <- ggplot(disch.token.characteristics,aes(x=DaysBeforeICUDischarge,y=meanDynamicTokens)) +
  geom_line(color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(ymin = lowerBound, ymax = upperBound),alpha=.2,fill='red',size=.75/.pt,color=NA) + 
  scale_x_reverse(breaks=seq(0,30,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  coord_cartesian(xlim=c(30,0),ylim=c(0,150)) +
  scale_y_continuous(breaks=seq(0,150,by=25))+
  geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab(expression(bold(paste('Dynamic tokens per patient (', italic('n'), ')'))))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

# Create plot of proportion of patients remaining after certain point in ICU stay
patient.count.plot <- ggplot(disch.token.characteristics,aes(x=DaysBeforeICUDischarge,y=countPatients/1550)) +
  geom_line(color='blue',alpha = 1, size=1.3/.pt)+
  coord_cartesian(xlim=c(30,0)) +
  scale_x_reverse(breaks=seq(0,30,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  ylab(expression(bold(paste('Patients remaining (', italic('p'), ')'))))+
  xlab('Days before ICU discharge')+
  geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

# Vertically align admission token characterisation plots
disch.token.counts <- plot_grid(dynamic.token.plot, patient.count.plot, ncol = 1, align = 'v', rel_heights = c(2,1))

# Create directory for current date and save admission characteristic plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'discharge_tokens.svg'),disch.token.counts,device= svg,units='in',dpi=600,width=15,height = 10)


###
# Load 95% confidence interval for performance metrics
overall.metrics <- read.csv('../model_performance/CI_overall_metrics.csv')

# 
overall.metrics %>%
  filter(METRIC == 'ORC',
         ADM_OR_DISCH == 'adm') %>%
  mutate(Days = WINDOW_IDX/12) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean),color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi),alpha=.2,fill='red',size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim = c(0.5,1)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.77, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.76, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.74, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('Days from ICU admission')+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

overall.metrics %>%
  filter(METRIC == 'Entropy',
         ADM_OR_DISCH == 'adm') %>%
  mutate(Days = WINDOW_IDX/12) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean),color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi),alpha=.2,fill='red',size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  geom_hline(yintercept =sqrt(7), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Shannon\'s Entropy')+
  xlab('Days from ICU admission')+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

overall.metrics %>%
  filter(METRIC == 'Somers D',
         ADM_OR_DISCH == 'adm') %>%
  mutate(Days = WINDOW_IDX/12) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean),color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi),alpha=.2,fill='red',size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim = c(0,1)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.60, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.57, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.54, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Somers D')+
  xlab('Days from ICU admission')+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )



overall.metrics %>%
  filter(METRIC == 'ORC',
         ADM_OR_DISCH == 'disch') %>%
  mutate(Days = WINDOW_IDX/12) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean),color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi),alpha=.2,fill='red',size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0),ylim=c(.5,1)) +
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.77, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.76, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.74, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('Days before ICU discharge')+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

overall.metrics %>%
  filter(METRIC == 'Entropy',
         ADM_OR_DISCH == 'disch') %>%
  mutate(Days = WINDOW_IDX/12) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean),color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi),alpha=.2,fill='red',size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0)) +
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  # geom_hline(yintercept = 0.77, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  # geom_hline(yintercept = 0.76, color='black',alpha = 1, size=1.3/.pt)+
  # geom_hline(yintercept = 0.74, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('Days before ICU discharge')+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )


overall.metrics %>%
  filter(METRIC == 'Somers D',
         ADM_OR_DISCH == 'disch') %>%
  mutate(Days = WINDOW_IDX/12) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean),color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi),alpha=.2,fill='red',size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0),ylim=c(0,1)) +
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.60, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.57, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.54, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Somers D')+
  xlab('Days before ICU discharge')+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

#### Create plots for threshold-level calibration
threshold.calib.slope <- read.csv('../model_performance/CI_threshold_metrics.csv') %>%
  filter(METRIC == 'Calib_Slope',
         ADM_OR_DISCH == 'adm') %>%
  mutate(Days = WINDOW_IDX/12) %>%
  ggplot() +
  geom_line(aes(x=WINDOW_IDX,y=mean),color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=WINDOW_IDX,ymin = lo, ymax = hi),alpha=.2,fill='red',size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim = c(0,1)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  # geom_hline(yintercept = 0.77, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  # geom_hline(yintercept = 0.76, color='black',alpha = 1, size=1.3/.pt)+
  # geom_hline(yintercept = 0.74, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('WINDOW_IDX from ICU admission')+
  facet_wrap(~THRESHOLD)+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

###### v2-0
# Load 95% confidence interval for performance metrics
tuning.grid <- read.csv('../model_outputs/v2-0/tuning_grid.csv') %>%
  select(tune_idx,WINDOW_DURATION) %>%
  distinct() %>%
  rename(TUNE_IDX = tune_idx) 
overall.metrics <- read.csv('../model_performance/v2-0/CI_overall_metrics.csv') %>%
  left_join(tuning.grid,by = 'TUNE_IDX')
thresh.metrics <- read.csv('../model_performance/v2-0/CI_threshold_metrics.csv') %>%
  left_join(tuning.grid,by = 'TUNE_IDX') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*WINDOW_DURATION)/24,
         WindowLabel = paste(WINDOW_DURATION,'hrs'))

### ORC
adm.orc <- overall.metrics %>%
  filter(METRIC == 'ORC',
         ADM_OR_DISCH == 'adm') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*WINDOW_DURATION)/24,
         WindowLabel = paste(WINDOW_DURATION,'hrs')) %>%
  mutate(WindowLabel = factor(WindowLabel,levels = c('2 hrs','8 hrs','12 hrs','24 hrs'))) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=WindowLabel),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=WindowLabel),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim = c(0.6,.8)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7741341, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7573236, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.7400537, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

disch.orc <- overall.metrics %>%
  filter(METRIC == 'ORC',
         ADM_OR_DISCH == 'disch')  %>%
  rowwise() %>%
  mutate(Days = ((WINDOW_IDX-1)*WINDOW_DURATION)/24,
         WindowLabel = paste(WINDOW_DURATION,'hrs')) %>%
  mutate(WindowLabel = factor(WindowLabel,levels = c('2 hrs','8 hrs','12 hrs','24 hrs'))) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=WindowLabel),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=WindowLabel),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0),ylim = c(0.6,.8)) + 
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7741341, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7573236, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.7400537, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('Days before ICU discharge')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

### Somers D
adm.somers <- overall.metrics %>%
  filter(METRIC == 'Somers D',
         ADM_OR_DISCH == 'adm') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*WINDOW_DURATION)/24,
         WindowLabel = paste(WINDOW_DURATION,'hrs')) %>%
  mutate(WindowLabel = factor(WindowLabel,levels = c('2 hrs','8 hrs','12 hrs','24 hrs'))) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=WindowLabel),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=WindowLabel),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim = c(0.3,.6)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.5982590, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.5673806, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.5358931, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Somers D')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

disch.somers <- overall.metrics %>%
  filter(METRIC == 'Somers D',
         ADM_OR_DISCH == 'disch') %>%
  rowwise() %>%
  mutate(Days = ((WINDOW_IDX-1)*WINDOW_DURATION)/24,
         WindowLabel = paste(WINDOW_DURATION,'hrs')) %>%
  mutate(WindowLabel = factor(WindowLabel,levels = c('2 hrs','8 hrs','12 hrs','24 hrs'))) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=WindowLabel),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=WindowLabel),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0),ylim = c(0.3,.6)) + 
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.5982590, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.5673806, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.5358931, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Somers D')+
  xlab('Days before ICU discharge')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

### Entropy
adm.entropy <- overall.metrics %>%
  filter(METRIC == 'Entropy',
         ADM_OR_DISCH == 'adm') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*WINDOW_DURATION)/24,
         WindowLabel = paste(WINDOW_DURATION,'hrs')) %>%
  mutate(WindowLabel = factor(WindowLabel,levels = c('2 hrs','8 hrs','12 hrs','24 hrs'))) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=WindowLabel),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=WindowLabel),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim = c(1.6,2.8)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Shannon\'s Entropy')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

disch.entropy <- overall.metrics %>%
  filter(METRIC == 'Entropy',
         ADM_OR_DISCH == 'disch')  %>%
  rowwise() %>%
  mutate(Days = ((WINDOW_IDX-1)*WINDOW_DURATION)/24,
         WindowLabel = paste(WINDOW_DURATION,'hrs')) %>%
  mutate(WindowLabel = factor(WindowLabel,levels = c('2 hrs','8 hrs','12 hrs','24 hrs'))) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=WindowLabel),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=WindowLabel),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0),ylim = c(1.6,2.8)) + 
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Shannon\'s Entropy')+
  xlab('Days before ICU discharge')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

### AUC
baseline.aucs <- read.csv('../../ordinal_GOSE_prediction/model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(METRIC == 'AUC',
         Threshold != 'Average',
         MODEL == 'APM_DeepMN') %>%
  rename(THRESHOLD=Threshold)

dynamic.auc <- thresh.metrics %>%
  filter(METRIC == 'AUC',
         ADM_OR_DISCH == 'adm',
         THRESHOLD != 'Average') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*WINDOW_DURATION)/24,
         WindowLabel = paste(WINDOW_DURATION,'hrs')) %>%
  mutate(WindowLabel = factor(WindowLabel,levels = c('2 hrs','8 hrs','12 hrs','24 hrs')))

ggplot() +
  geom_line(data =dynamic.auc, aes(x=Days,y=mean,color=WindowLabel),alpha = 1, size=1.3/.pt)+
  geom_ribbon(data =dynamic.auc, aes(x=Days,ymin = lo, ymax = hi,fill=WindowLabel),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  geom_hline(data=baseline.aucs, aes(yintercept = hi), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(data=baseline.aucs, aes(yintercept = mean), color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(data=baseline.aucs, aes(yintercept = lo), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  facet_wrap(~THRESHOLD,ncol = 2,scales = 'free')+
  ylab('AUC')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

disch.auc <- thresh.metrics %>%
  filter(METRIC == 'AUC',
         ADM_OR_DISCH == 'disch',
         THRESHOLD != 'Average') %>%
  rowwise() %>%
  mutate(Days = ((WINDOW_IDX-1)*WINDOW_DURATION)/24,
         WindowLabel = paste(WINDOW_DURATION,'hrs')) %>%
  mutate(WindowLabel = factor(WindowLabel,levels = c('2 hrs','8 hrs','12 hrs','24 hrs')))

ggplot() +
  geom_line(data =disch.auc, aes(x=Days,y=mean,color=WindowLabel),alpha = 1, size=1.3/.pt)+
  geom_ribbon(data =disch.auc, aes(x=Days,ymin = lo, ymax = hi,fill=WindowLabel),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0)) + 
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  geom_hline(data=baseline.aucs, aes(yintercept = hi), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(data=baseline.aucs, aes(yintercept = mean), color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(data=baseline.aucs, aes(yintercept = lo), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  facet_wrap(~THRESHOLD,ncol = 2,scales = 'free')+
  ylab('AUC')+
  xlab('Days before ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

### Calibration Slope
baseline.calib.slope <- read.csv('../../ordinal_GOSE_prediction/model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(METRIC == 'Calib_Slope',
         Threshold != 'Average',
         MODEL == 'APM_DeepMN') %>%
  rename(THRESHOLD=Threshold)

dynamic.calib.slope <- thresh.metrics %>%
  filter(METRIC == 'Calib_Slope',
         ADM_OR_DISCH == 'adm',
         THRESHOLD != 'Average') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*WINDOW_DURATION)/24,
         WindowLabel = paste(WINDOW_DURATION,'hrs')) %>%
  mutate(WindowLabel = factor(WindowLabel,levels = c('2 hrs','8 hrs','12 hrs','24 hrs')))

ggplot() +
  geom_line(data =dynamic.calib.slope, aes(x=Days,y=mean,color=WindowLabel),alpha = 1, size=1.3/.pt)+
  geom_ribbon(data =dynamic.calib.slope, aes(x=Days,ymin = lo, ymax = hi,fill=WindowLabel),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  geom_hline(data=baseline.calib.slope, aes(yintercept = hi), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(data=baseline.calib.slope, aes(yintercept = mean), color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(data=baseline.calib.slope, aes(yintercept = lo), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 1, color='orange',alpha = 1, size=2/.pt)+
  facet_wrap(~THRESHOLD,ncol = 2,scales = 'free')+
  ylab('Calibration Slope')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

disch.calib.slope <- thresh.metrics %>%
  filter(METRIC == 'Calib_Slope',
         ADM_OR_DISCH == 'disch',
         THRESHOLD != 'Average') %>%
  rowwise() %>%
  mutate(Days = ((WINDOW_IDX-1)*WINDOW_DURATION)/24,
         WindowLabel = paste(WINDOW_DURATION,'hrs')) %>%
  mutate(WindowLabel = factor(WindowLabel,levels = c('2 hrs','8 hrs','12 hrs','24 hrs')))

ggplot() +
  geom_line(data =disch.calib.slope, aes(x=Days,y=mean,color=WindowLabel),alpha = 1, size=1.3/.pt)+
  geom_ribbon(data =disch.calib.slope, aes(x=Days,ymin = lo, ymax = hi,fill=WindowLabel),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0)) + 
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  geom_hline(data=baseline.calib.slope, aes(yintercept = hi), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(data=baseline.calib.slope, aes(yintercept = mean), color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(data=baseline.calib.slope, aes(yintercept = lo), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 1, color='orange',alpha = 1, size=2/.pt)+
  facet_wrap(~THRESHOLD,ncol = 2,scales = 'free')+
  ylab('Calibration Slope')+
  xlab('Days before ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

#### Individual trajectory example
compiled.test.predictions <- read.csv('../model_outputs/v2-0/compiled_test_predictions_from_adm.csv') %>%
  filter(tune_idx == 1)

individual.trajectories <- compiled.test.predictions %>%
  filter(GUPI == '2JwX739') %>%
  mutate(Days = (WindowIdx*2)/24,
         PrGOSE.gt.1 = 1 - Pr.GOSE.1.,
         PrGOSE.gt.3 = 1 - (Pr.GOSE.1.+Pr.GOSE.2.3.),
         PrGOSE.gt.4 = 1 - (Pr.GOSE.1.+Pr.GOSE.2.3.+Pr.GOSE.4.),
         PrGOSE.gt.5 = 1 - (Pr.GOSE.1.+Pr.GOSE.2.3.+Pr.GOSE.4.+Pr.GOSE.5.),
         PrGOSE.gt.6 = 1 - (Pr.GOSE.1.+Pr.GOSE.2.3.+Pr.GOSE.4.+Pr.GOSE.5.+Pr.GOSE.6.),
         PrGOSE.gt.7 = Pr.GOSE.8.) %>%
  select(Days,starts_with('PrGOSE.gt.'),TrueLabel) %>%
  pivot_longer(cols = starts_with('PrGOSE.gt.'),names_to = 'Threshold',values_to = 'Probability') %>%
  mutate(LINETYPE = 'solid',
         Threshold = plyr::mapvalues(Threshold,
                                     to = c('Pr(GOSE>1)','Pr(GOSE>3)','Pr(GOSE>4)','Pr(GOSE>5)','Pr(GOSE>6)','Pr(GOSE>7)'),
                                     from = c('PrGOSE.gt.1','PrGOSE.gt.3','PrGOSE.gt.4','PrGOSE.gt.5','PrGOSE.gt.6','PrGOSE.gt.7')))
individual.trajectories$LINETYPE[individual.trajectories$Threshold %in% c('Pr(GOSE>5)','Pr(GOSE>6)','Pr(GOSE>7)')] = 'dashed'

individual.trajectories %>%
  ggplot(aes(x=Days,y=Probability,color=Threshold,linetype=LINETYPE)) +
  geom_hline(yintercept = .5, color='orange',size=1.3/.pt)+
  geom_line(alpha = 1, size=2/.pt) + 
  scale_x_continuous(breaks=seq(0,7,by=.5),expand = expansion(mult = c(0, .01)))+
  xlab('Days after ICU Admission') +
  ylab('Predicted Probability')+
  scale_linetype_identity()+
  theme_bw() +
  theme(
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    axis.line.x = element_blank(),
    axis.line.y = element_blank(),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold'))

icu.timestamps <- read.csv('../timestamps/ICU_adm_disch_timestamps.csv') %>%
  filter(GUPI == '2JwX739')

demo.info <- read.csv('../CENTER-TBI/DemoInjHospMedHx/data.csv') %>%
  filter(GUPI == '2JwX739')

biomarkers <- read.csv('../CENTER-TBI/Biomarkers/data.csv') %>%
  filter(GUPI == '2JwX739')

meds <- read.csv('../CENTER-TBI/Medication/data.csv') %>%
  filter(GUPI == '2JwX739')

central.haemo <- read.csv('../CENTER-TBI/CentralHaemostasis/data.csv') %>%
  filter(GUPI == '2JwX739')

dh.values <- read.csv('../CENTER-TBI/DailyHourlyValues/data.csv') %>%
  filter(GUPI == '2JwX739')

dtil.values <- read.csv('../CENTER-TBI/DailyTIL/data.csv') %>%
  filter(GUPI == '2JwX739')

daily.values <- read.csv('../CENTER-TBI/DailyVitals/data.csv') %>%
  filter(GUPI == '2JwX739')

###### v2-1
# Load 95% confidence interval for performance metrics
tuning.grid <- read.csv('../model_outputs/v2-0/tuning_grid.csv') %>%
  select(tune_idx,WINDOW_DURATION) %>%
  distinct() %>%
  rename(TUNE_IDX = tune_idx) 
day.2.overall.metrics <- read.csv('../model_performance/v2-1/Day2/CI_overall_metrics.csv') %>%
  left_join(tuning.grid,by = 'TUNE_IDX')
day.2.thresh.metrics <- read.csv('../model_performance/v2-1/Day2/CI_threshold_metrics.csv') %>%
  left_join(tuning.grid,by = 'TUNE_IDX') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*WINDOW_DURATION)/24,
         WindowLabel = paste(WINDOW_DURATION,'hrs'))

day.3.overall.metrics <- read.csv('../model_performance/v2-1/Day3/CI_overall_metrics.csv') %>%
  left_join(tuning.grid,by = 'TUNE_IDX')
day.3.thresh.metrics <- read.csv('../model_performance/v2-1/Day3/CI_threshold_metrics.csv') %>%
  left_join(tuning.grid,by = 'TUNE_IDX') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*WINDOW_DURATION)/24,
         WindowLabel = paste(WINDOW_DURATION,'hrs'))

### ORC - Day2
day.2.adm.orc <- day.2.overall.metrics %>%
  filter(METRIC == 'ORC',
         ADM_OR_DISCH == 'adm') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*WINDOW_DURATION)/24,
         WindowLabel = paste(WINDOW_DURATION,'hrs')) %>%
  mutate(WindowLabel = factor(WindowLabel,levels = c('2 hrs','8 hrs','12 hrs','24 hrs'))) %>%
  filter(Days <= 2) %>%
  drop_na(lo) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=WindowLabel),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=WindowLabel),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,2),ylim = c(0.6,.8)) + 
  scale_x_continuous(breaks=seq(0,2,by=.5),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7741341, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7573236, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.7400537, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

day.2.disch.orc <- day.2.overall.metrics %>%
  filter(METRIC == 'ORC',
         ADM_OR_DISCH == 'disch')  %>%
  rowwise() %>%
  mutate(Days = ((WINDOW_IDX-1)*WINDOW_DURATION)/24,
         WindowLabel = paste(WINDOW_DURATION,'hrs')) %>%
  mutate(WindowLabel = factor(WindowLabel,levels = c('2 hrs','8 hrs','12 hrs','24 hrs'))) %>%
  filter(Days <= 2) %>%
  drop_na(lo) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=WindowLabel),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=WindowLabel),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(2,0),ylim = c(0.6,.8)) + 
  scale_x_reverse(breaks=seq(0,2,by=.5),expand = expansion(mult = c(0.01, 0.005)))+ 
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7741341, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7573236, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.7400537, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('Days before ICU discharge')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

### ORC - Day3
day.3.adm.orc <- day.3.overall.metrics %>%
  filter(METRIC == 'ORC',
         ADM_OR_DISCH == 'adm') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*WINDOW_DURATION)/24,
         WindowLabel = paste(WINDOW_DURATION,'hrs')) %>%
  mutate(WindowLabel = factor(WindowLabel,levels = c('2 hrs','8 hrs','12 hrs','24 hrs'))) %>%
  filter(Days <= 3) %>%
  drop_na(lo) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=WindowLabel),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=WindowLabel),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,3),ylim = c(0.6,.8)) + 
  scale_x_continuous(breaks=seq(0,3,by=.5),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7741341, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7573236, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.7400537, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

day.3.disch.orc <- day.3.overall.metrics %>%
  filter(METRIC == 'ORC',
         ADM_OR_DISCH == 'disch')  %>%
  rowwise() %>%
  mutate(Days = ((WINDOW_IDX-1)*WINDOW_DURATION)/24,
         WindowLabel = paste(WINDOW_DURATION,'hrs')) %>%
  mutate(WindowLabel = factor(WindowLabel,levels = c('2 hrs','8 hrs','12 hrs','24 hrs'))) %>%
  filter(Days <= 3) %>%
  drop_na(lo) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=WindowLabel),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=WindowLabel),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(3,0),ylim = c(0.6,.8)) + 
  scale_x_reverse(breaks=seq(0,3,by=.5),expand = expansion(mult = c(0.01, 0.005)))+ 
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7741341, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7573236, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.7400537, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('Days before ICU discharge')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )


###### v3-0
# Load 95% confidence interval for performance metrics
overall.metrics <- read.csv('../model_performance/v3-0/CI_overall_metrics.csv')
thresh.metrics <- read.csv('../model_performance/v3-0/CI_threshold_metrics.csv')

### ORC
adm.orc <- overall.metrics %>%
  filter(METRIC == 'ORC',
         ADM_OR_DISCH == 'adm') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean),color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi),fill='red',alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim = c(0.6,.8)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7741341, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7573236, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.7400537, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('Days from ICU admission')+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

disch.orc <- overall.metrics %>%
  filter(METRIC == 'ORC',
         ADM_OR_DISCH == 'disch')  %>%
  rowwise() %>%
  mutate(Days = ((WINDOW_IDX-1)*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean),color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi),fill='red',alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0),ylim = c(0.6,.8)) + 
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7741341, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7573236, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.7400537, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('Days before ICU discharge')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

### Somers D
adm.somers <- overall.metrics %>%
  filter(METRIC == 'Somers D',
         ADM_OR_DISCH == 'adm') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean),color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi),fill='red',alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim = c(0.3,.6)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.5982590, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.5673806, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.5358931, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Somers D')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

disch.somers <- overall.metrics %>%
  filter(METRIC == 'Somers D',
         ADM_OR_DISCH == 'disch') %>%
  rowwise() %>%
  mutate(Days = ((WINDOW_IDX-1)*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean),color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi),fill='red',alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0),ylim = c(0.3,.6)) + 
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.5982590, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.5673806, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.5358931, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Somers D')+
  xlab('Days before ICU discharge')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

### Entropy
adm.entropy <- overall.metrics %>%
  filter(METRIC == 'Entropy',
         ADM_OR_DISCH == 'adm') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean),color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi),fill='red',alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim = c(1.25,2.8)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Shannon\'s Entropy')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

disch.entropy <- overall.metrics %>%
  filter(METRIC == 'Entropy',
         ADM_OR_DISCH == 'disch')  %>%
  rowwise() %>%
  mutate(Days = ((WINDOW_IDX-1)*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean),color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi),fill='red',alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0),ylim = c(1.25,2.8)) + 
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Shannon\'s Entropy')+
  xlab('Days before ICU discharge')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

### AUC
baseline.aucs <- read.csv('../../ordinal_GOSE_prediction/model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(METRIC == 'AUC',
         Threshold != 'Average',
         MODEL == 'APM_DeepMN') %>%
  rename(THRESHOLD=Threshold)

dynamic.auc <- thresh.metrics %>%
  filter(METRIC == 'AUC',
         ADM_OR_DISCH == 'adm',
         THRESHOLD != 'Average') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24)

ggplot() +
  geom_line(data =dynamic.auc, aes(x=Days,y=mean),color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(data =dynamic.auc, aes(x=Days,ymin = lo, ymax = hi),fill='red',alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  geom_hline(data=baseline.aucs, aes(yintercept = hi), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(data=baseline.aucs, aes(yintercept = mean), color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(data=baseline.aucs, aes(yintercept = lo), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  facet_wrap(~THRESHOLD,ncol = 2,scales = 'free')+
  ylab('AUC')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

disch.auc <- thresh.metrics %>%
  filter(METRIC == 'AUC',
         ADM_OR_DISCH == 'disch',
         THRESHOLD != 'Average') %>%
  rowwise() %>%
  mutate(Days = ((WINDOW_IDX-1)*2)/24)

ggplot() +
  geom_line(data =disch.auc, aes(x=Days,y=mean),color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(data =disch.auc, aes(x=Days,ymin = lo, ymax = hi),fill='red',alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0)) + 
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  geom_hline(data=baseline.aucs, aes(yintercept = hi), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(data=baseline.aucs, aes(yintercept = mean), color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(data=baseline.aucs, aes(yintercept = lo), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  facet_wrap(~THRESHOLD,ncol = 2,scales = 'free')+
  ylab('AUC')+
  xlab('Days before ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

### Calibration Slope
baseline.calib.slope <- read.csv('../../ordinal_GOSE_prediction/model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(METRIC == 'Calib_Slope',
         Threshold != 'Average',
         MODEL == 'APM_DeepMN') %>%
  rename(THRESHOLD=Threshold)

dynamic.calib.slope <- thresh.metrics %>%
  filter(METRIC == 'Calib_Slope',
         ADM_OR_DISCH == 'adm',
         THRESHOLD != 'Average') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24)

ggplot() +
  geom_line(data =dynamic.calib.slope, aes(x=Days,y=mean),color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(data =dynamic.calib.slope, aes(x=Days,ymin = lo, ymax = hi),fill='red',alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  geom_hline(data=baseline.calib.slope, aes(yintercept = hi), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(data=baseline.calib.slope, aes(yintercept = mean), color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(data=baseline.calib.slope, aes(yintercept = lo), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 1, color='orange',alpha = 1, size=2/.pt)+
  facet_wrap(~THRESHOLD,ncol = 2,scales = 'free')+
  ylab('Calibration Slope')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

disch.calib.slope <- thresh.metrics %>%
  filter(METRIC == 'Calib_Slope',
         ADM_OR_DISCH == 'disch',
         THRESHOLD != 'Average') %>%
  rowwise() %>%
  mutate(Days = ((WINDOW_IDX-1)*2)/24)

ggplot() +
  geom_line(data=disch.calib.slope, aes(x=Days,y=mean),color='red',alpha = 1, size=1.3/.pt)+
  geom_ribbon(data =disch.calib.slope, aes(x=Days,ymin = lo, ymax = hi),fill='red',alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0)) + 
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  geom_hline(data=baseline.calib.slope, aes(yintercept = hi), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(data=baseline.calib.slope, aes(yintercept = mean), color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(data=baseline.calib.slope, aes(yintercept = lo), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 1, color='orange',alpha = 1, size=2/.pt)+
  facet_wrap(~THRESHOLD,ncol = 2,scales = 'free')+
  ylab('Calibration Slope')+
  xlab('Days before ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

###### v4-0
# Load 95% confidence interval for performance metrics
overall.metrics <- read.csv('../model_performance/v4-0/CI_overall_metrics.csv')
thresh.metrics <- read.csv('../model_performance/v4-0/CI_threshold_metrics.csv')

### ORC
adm.orc <- overall.metrics %>%
  filter(METRIC == 'ORC',
         ADM_OR_DISCH == 'adm',
         TUNE_IDX %in% c(1,2)) %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim = c(0.6,.8)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7741341, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7573236, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.7400537, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('Days from ICU admission')+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

### Somers D
adm.somers <- overall.metrics %>%
  filter(METRIC == 'Somers D',
         ADM_OR_DISCH == 'adm') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim = c(0.3,.6)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.5982590, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.5673806, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.5358931, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Somers D')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

### Entropy
adm.entropy <- overall.metrics %>%
  filter(METRIC == 'Entropy',
         ADM_OR_DISCH == 'adm') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim = c(1.25,2.8)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Shannon\'s Entropy')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

### AUC
baseline.aucs <- read.csv('../../ordinal_GOSE_prediction/model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(METRIC == 'AUC',
         Threshold != 'Average',
         MODEL == 'APM_DeepMN') %>%
  rename(THRESHOLD=Threshold)

dynamic.auc <- thresh.metrics %>%
  filter(METRIC == 'AUC',
         ADM_OR_DISCH == 'adm',
         THRESHOLD != 'Average') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24)

ggplot() +
  geom_line(data =dynamic.auc, aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(data =dynamic.auc, aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  geom_hline(data=baseline.aucs, aes(yintercept = hi), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(data=baseline.aucs, aes(yintercept = mean), color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(data=baseline.aucs, aes(yintercept = lo), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  facet_wrap(~THRESHOLD,ncol = 2,scales = 'free')+
  ylab('AUC')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

### Calibration Slope
baseline.calib.slope <- read.csv('../../ordinal_GOSE_prediction/model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(METRIC == 'Calib_Slope',
         Threshold != 'Average',
         MODEL == 'APM_DeepMN') %>%
  rename(THRESHOLD=Threshold)

dynamic.calib.slope <- thresh.metrics %>%
  filter(METRIC == 'Calib_Slope',
         ADM_OR_DISCH == 'adm',
         THRESHOLD != 'Average',
         TUNE_IDX %in% c(1,2)) %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24)

ggplot() +
  geom_line(data =dynamic.calib.slope, aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(data =dynamic.calib.slope, aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  geom_hline(data=baseline.calib.slope, aes(yintercept = hi), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(data=baseline.calib.slope, aes(yintercept = mean), color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(data=baseline.calib.slope, aes(yintercept = lo), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 1, color='orange',alpha = 1, size=2/.pt)+
  facet_wrap(~THRESHOLD,ncol = 2,scales = 'free')+
  ylab('Calibration Slope')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

##### Compare performance across model versions and configurations
v1.overall.metrics <- read.csv('../model_performance/v1-0/CI_overall_metrics.csv') %>%
  filter(ADM_OR_DISCH == 'adm') %>%
  mutate(VERSION = 'v1') %>%
  select(-ADM_OR_DISCH)
v1.thresh.metrics <- read.csv('../model_performance/v1-0/CI_threshold_metrics.csv') %>%
  filter(ADM_OR_DISCH == 'adm') %>%
  mutate(VERSION = 'v1') %>%
  select(-ADM_OR_DISCH)
v2.overall.metrics <- read.csv('../model_performance/v2-0/CI_overall_metrics.csv') %>%
  filter(ADM_OR_DISCH == 'adm',
         TUNE_IDX == 1) %>%
  mutate(VERSION = 'v2') %>%
  select(-c(ADM_OR_DISCH,TUNE_IDX))
v2.thresh.metrics <- read.csv('../model_performance/v2-0/CI_threshold_metrics.csv') %>%
  filter(ADM_OR_DISCH == 'adm',
         TUNE_IDX == 1) %>%
  mutate(VERSION = 'v2') %>%
  select(-c(ADM_OR_DISCH,TUNE_IDX))
v3.overall.metrics <- read.csv('../model_performance/v3-0/CI_overall_metrics.csv') %>%
  filter(ADM_OR_DISCH == 'adm',
         TUNE_IDX == 1) %>%
  mutate(VERSION = 'v3') %>%
  select(-c(ADM_OR_DISCH,TUNE_IDX))
v3.thresh.metrics <- read.csv('../model_performance/v3-0/CI_threshold_metrics.csv') %>%
  filter(ADM_OR_DISCH == 'adm',
         TUNE_IDX == 1) %>%
  mutate(VERSION = 'v3') %>%
  select(-c(ADM_OR_DISCH,TUNE_IDX))
v4.overall.metrics <- read.csv('../model_performance/v4-0/CI_overall_metrics.csv') %>%
  filter(ADM_OR_DISCH == 'adm',
         TUNE_IDX == 1) %>%
  mutate(VERSION = 'v4') %>%
  select(-c(ADM_OR_DISCH,TUNE_IDX))
v4.thresh.metrics <- read.csv('../model_performance/v4-0/CI_threshold_metrics.csv') %>%
  filter(ADM_OR_DISCH == 'adm',
         TUNE_IDX == 1) %>%
  mutate(VERSION = 'v4') %>%
  select(-c(ADM_OR_DISCH,TUNE_IDX))

compiled.overall.metrics <- rbind(v1.overall.metrics,v2.overall.metrics,v3.overall.metrics,v4.overall.metrics)
compiled.thresh.metrics <- rbind(v1.thresh.metrics,v2.thresh.metrics,v3.thresh.metrics,v4.thresh.metrics)

### ORC
orc.plot <- compiled.overall.metrics %>%
  filter(METRIC == 'ORC',
         VERSION == 'v4') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=(VERSION)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=(VERSION)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim = c(0.6,.8)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7741341, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7573236, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.7400537, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('Days from ICU admission')+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

### Entropy
entropy.plot <- compiled.overall.metrics %>%
  filter(METRIC == 'Entropy') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=factor(VERSION)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=factor(VERSION)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Entropy')+
  xlab('Days from ICU admission')+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

### Calibration slope
baseline.calib.slope <- read.csv('../../ordinal_GOSE_prediction/model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(METRIC == 'Calib_Slope',
         Threshold != 'Average',
         MODEL == 'APM_DeepMN') %>%
  rename(THRESHOLD=Threshold)

calib.slopes <- compiled.thresh.metrics %>%
  filter(METRIC == 'Calib_Slope',
         THRESHOLD != 'Average') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24)

calib.slope.plot <- ggplot() +
  geom_line(data =calib.slopes, aes(x=Days,y=mean,color=factor(VERSION)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(data =calib.slopes, aes(x=Days,ymin = lo, ymax = hi,fill=factor(VERSION)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim=c(0,1.5)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  geom_hline(data=baseline.calib.slope, aes(yintercept = hi), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(data=baseline.calib.slope, aes(yintercept = mean), color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(data=baseline.calib.slope, aes(yintercept = lo), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 1, color='orange',alpha = 1, size=2/.pt)+
  facet_wrap(~THRESHOLD,ncol = 2,scales = 'free')+
  ylab('Calibration Slope')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

###### v5-0
# Load 95% confidence interval for performance metrics
v5.tuning.grid <- read.csv('../model_outputs/v5-0/tuning_grid.csv') %>%
  rename(TUNE_IDX = tune_idx) %>%
  select(-c(repeat.,fold)) %>%
  unique() %>%
  select(TUNE_IDX,WINDOW_LIMIT,STRATEGY,NUM_EPOCHS) %>%
  mutate(TIME_TOKENS = 'None')
v2.tuning.grid <- read.csv('../model_outputs/v2-0/tuning_grid.csv') %>%
  rename(TUNE_IDX = tune_idx) %>%
  filter(WINDOW_DURATION == 2) %>%
  select(TUNE_IDX,NUM_EPOCHS) %>%
  unique() %>%
  mutate(WINDOW_LIMIT = 84,
         STRATEGY = 'abs',
         TIME_TOKENS = 'TOD_only')
v2.abs.orc <- read.csv('../model_performance/v2-0/CI_overall_metrics.csv') %>%
  filter(METRIC == 'ORC',
         TUNE_IDX == 1,
         ADM_OR_DISCH == 'adm') %>%
  left_join(v2.tuning.grid,by = 'TUNE_IDX') %>%
  select(-ADM_OR_DISCH) %>%
  mutate(TUNE_IDX = plyr::mapvalues(TUNE_IDX,from=1,to=11))
v4.tuning.grid <- read.csv('../model_outputs/v4-0/tuning_grid.csv') %>%
  rename(TUNE_IDX = tune_idx) %>%
  filter(TUNE_IDX == 1) %>%
  select(TUNE_IDX,WINDOW_LIMIT,NUM_EPOCHS) %>%
  unique() %>%
  mutate(STRATEGY = 'diff',
         TIME_TOKENS = 'TFA_only')
v4.diff.orc <- read.csv('../model_performance/v4-0/CI_overall_metrics.csv') %>%
  filter(METRIC == 'ORC',
         TUNE_IDX == 1) %>%
  left_join(v4.tuning.grid,by = 'TUNE_IDX') %>%
  select(-ADM_OR_DISCH) %>%
  mutate(TUNE_IDX = plyr::mapvalues(TUNE_IDX,from=1,to=12))

v5.overall.metrics <- read.csv('../model_performance/v5-0/test_CI_overall_metrics.csv') %>%
  left_join(v5.tuning.grid,by = 'TUNE_IDX')
v5.thresh.metrics <- read.csv('../model_performance/v5-0/test_CI_threshold_metrics.csv') %>%
  left_join(v5.tuning.grid,by = 'TUNE_IDX')

abs.orc <- v5.overall.metrics %>%
  filter(METRIC == 'ORC',
         STRATEGY == 'abs',
         SET == 'test') %>%
  select(-c(SET,ADM_OR_DISCH)) %>%
  rbind(v2.abs.orc)

diff.orc <- overall.metrics %>%
  filter(METRIC == 'ORC',
         STRATEGY == 'diff',
         SET == 'test') %>%
  select(-c(SET,ADM_OR_DISCH)) %>%
  rbind(v4.diff.orc)

### ORC
abs.orc.curve <- abs.orc %>%
  filter(METRIC == 'ORC') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim = c(0.6,1)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7741341, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7573236, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.7400537, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('Days from ICU admission')+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

diff.orc.curve <- diff.orc %>%
  filter(METRIC == 'ORC') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim = c(0.6,1)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7741341, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7573236, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.7400537, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('Days from ICU admission')+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )
### Somers D
adm.somers <- overall.metrics %>%
  filter(METRIC == 'Somers D',
         ADM_OR_DISCH == 'adm') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim = c(0.3,.6)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.5982590, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.5673806, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.5358931, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Somers D')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

### Entropy
adm.entropy <- overall.metrics %>%
  filter(METRIC == 'Entropy',
         ADM_OR_DISCH == 'adm') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim = c(1.25,2.8)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Shannon\'s Entropy')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

### AUC
baseline.aucs <- read.csv('../../ordinal_GOSE_prediction/model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(METRIC == 'AUC',
         Threshold != 'Average',
         MODEL == 'APM_DeepMN') %>%
  rename(THRESHOLD=Threshold)

dynamic.auc <- thresh.metrics %>%
  filter(METRIC == 'AUC',
         ADM_OR_DISCH == 'adm',
         THRESHOLD != 'Average') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24)

ggplot() +
  geom_line(data =dynamic.auc, aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(data =dynamic.auc, aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  geom_hline(data=baseline.aucs, aes(yintercept = hi), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(data=baseline.aucs, aes(yintercept = mean), color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(data=baseline.aucs, aes(yintercept = lo), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  facet_wrap(~THRESHOLD,ncol = 2,scales = 'free')+
  ylab('AUC')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

### Calibration Slope
baseline.calib.slope <- read.csv('../../ordinal_GOSE_prediction/model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(METRIC == 'Calib_Slope',
         Threshold != 'Average',
         MODEL == 'APM_DeepMN') %>%
  rename(THRESHOLD=Threshold)

dynamic.calib.slope <- thresh.metrics %>%
  filter(METRIC == 'Calib_Slope',
         ADM_OR_DISCH == 'adm',
         THRESHOLD != 'Average') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24)

ggplot() +
  geom_line(data =dynamic.calib.slope, aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(data =dynamic.calib.slope, aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7),ylim=c(0,2)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  geom_hline(data=baseline.calib.slope, aes(yintercept = hi), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(data=baseline.calib.slope, aes(yintercept = mean), color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(data=baseline.calib.slope, aes(yintercept = lo), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 1, color='orange',alpha = 1, size=2/.pt)+
  facet_wrap(~THRESHOLD,ncol = 2,scales = 'free')+
  ylab('Calibration Slope')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Tune idx"),
         color=guide_legend(title="Tune idx"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

###### v6-0
# Load 95% confidence interval for performance metrics
overall.metrics <- read.csv('../model_performance/v6-0/test_CI_overall_metrics.csv')
thresh.metrics <- read.csv('../model_performance/v6-0/test_CI_threshold_metrics.csv')

### ORC
orc.plot <- overall.metrics %>%
  filter(METRIC == 'ORC') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  geom_hline(yintercept = 0.7741341, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7573236, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.7400537, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('Days from ICU admission')+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

### Somers D
somers.plot <- overall.metrics %>%
  filter(METRIC == 'Somers D') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.5982590, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.5673806, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.5358931, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Somers D')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

### Entropy
entropy.plot <- overall.metrics %>%
  filter(METRIC == 'Entropy') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  ylab('Shannon\'s Entropy')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

### AUC
baseline.aucs <- read.csv('../../ordinal_GOSE_prediction/model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(METRIC == 'AUC',
         Threshold != 'Average',
         MODEL == 'APM_DeepMN') %>%
  rename(THRESHOLD=Threshold)

dynamic.auc <- thresh.metrics %>%
  filter(METRIC == 'AUC',
         THRESHOLD != 'Average') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24)

auc.plot <- ggplot() +
  geom_line(data =dynamic.auc, aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(data =dynamic.auc, aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  geom_hline(data=baseline.aucs, aes(yintercept = hi), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(data=baseline.aucs, aes(yintercept = mean), color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(data=baseline.aucs, aes(yintercept = lo), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  facet_wrap(~THRESHOLD,ncol = 2,scales = 'free')+
  ylab('AUC')+
  xlab('Days from ICU admission')+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

### Calibration Slope
baseline.calib.slope <- read.csv('../../ordinal_GOSE_prediction/model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(METRIC == 'Calib_Slope',
         Threshold != 'Average',
         MODEL == 'APM_DeepMN') %>%
  rename(THRESHOLD=Threshold)

dynamic.calib.slope <- thresh.metrics %>%
  filter(METRIC == 'CALIB_SLOPE',
         THRESHOLD != 'Average',
         TUNE_IDX == 135) %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24)

calib.slope.plot <- ggplot() +
  geom_line(data =dynamic.calib.slope, aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(data =dynamic.calib.slope, aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(0,7)) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0, .01)))+
  geom_hline(data=baseline.calib.slope, aes(yintercept = hi), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(data=baseline.calib.slope, aes(yintercept = mean), color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(data=baseline.calib.slope, aes(yintercept = lo), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 1, color='orange',alpha = 1, size=2/.pt)+
  facet_wrap(~THRESHOLD,ncol = 2,scales = 'free')+
  ylab('Calibration Slope')+
  xlab('Days from ICU admission')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

###################
# Load 95% confidence interval for performance metrics
disch.overall.metrics <- read.csv('../model_performance/v6-0/test_CI_overall_metrics_from_disch.csv')
disch.thresh.metrics <- read.csv('../model_performance/v6-0/test_CI_threshold_metrics_from_disch.csv')

### ORC
orc.plot <- disch.overall.metrics %>%
  filter(METRIC == 'ORC') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0)) + 
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  geom_hline(yintercept = 0.7741341, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.7573236, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.7400537, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Ordinal c-index')+
  xlab('Days before ICU discharge')+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

### Somers D
somers.plot <- disch.overall.metrics %>%
  filter(METRIC == 'Somers D') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0)) + 
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  #geom_vline(xintercept = 7, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.5982590, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0.5673806, color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(yintercept = 0.5358931, color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  ylab('Somers D')+
  xlab('Days before ICU discharge')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

### Entropy
entropy.plot <- disch.overall.metrics %>%
  filter(METRIC == 'Entropy') %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24) %>%
  ggplot() +
  geom_line(aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0)) + 
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  ylab('Shannon\'s Entropy')+
  xlab('Days before ICU discharge')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

### AUC
baseline.aucs <- read.csv('../../ordinal_GOSE_prediction/model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(METRIC == 'AUC',
         Threshold != 'Average',
         MODEL == 'APM_DeepMN') %>%
  rename(THRESHOLD=Threshold)

dynamic.auc <- disch.thresh.metrics %>%
  filter(METRIC == 'AUC',
         THRESHOLD != 'Average',
         TUNE_IDX == 135) %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24)

auc.plot <- ggplot() +
  geom_line(data =dynamic.auc, aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(data =dynamic.auc, aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0)) + 
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  geom_hline(data=baseline.aucs, aes(yintercept = hi), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(data=baseline.aucs, aes(yintercept = mean), color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(data=baseline.aucs, aes(yintercept = lo), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  facet_wrap(~THRESHOLD,ncol = 2,scales = 'free')+
  ylab('AUC')+
  xlab('Days before ICU discharge')+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

### Calibration Slope
baseline.calib.slope <- read.csv('../../ordinal_GOSE_prediction/model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(METRIC == 'Calib_Slope',
         Threshold != 'Average',
         MODEL == 'APM_DeepMN') %>%
  rename(THRESHOLD=Threshold)

dynamic.calib.slope <- disch.thresh.metrics %>%
  filter(METRIC == 'CALIB_SLOPE',
         THRESHOLD != 'Average',
         TUNE_IDX == 135) %>%
  rowwise() %>%
  mutate(Days = (WINDOW_IDX*2)/24)

calib.slope.plot <- ggplot() +
  geom_line(data =dynamic.calib.slope, aes(x=Days,y=mean,color=factor(TUNE_IDX)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(data =dynamic.calib.slope, aes(x=Days,ymin = lo, ymax = hi,fill=factor(TUNE_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  coord_cartesian(xlim=c(7,0)) + 
  scale_x_reverse(breaks=seq(0,7,by=1),expand = expansion(mult = c(0.01, 0.005)))+ 
  geom_hline(data=baseline.calib.slope, aes(yintercept = hi), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(data=baseline.calib.slope, aes(yintercept = mean), color='black',alpha = 1, size=1.3/.pt)+
  geom_hline(data=baseline.calib.slope, aes(yintercept = lo), color='black',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 1, color='orange',alpha = 1, size=2/.pt)+
  facet_wrap(~THRESHOLD,ncol = 2,scales = 'free')+
  ylab('Calibration Slope')+
  xlab('Days before ICU discharge')+
  guides(fill=guide_legend(title="Data resampling window"),
         color=guide_legend(title="Data resampling window"))+
  theme_bw()+
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(color = "black", face = 'bold')
  )

###### CALIBRATION CURVES
#### accuracy/confidence curves
acc.conf.calib.curves <- read.csv('../model_performance/v6-0/CI_acc_conf_calib_curves.csv')

# admission
adm.acc.conf.plot <- acc.conf.calib.curves %>%
  filter(ADM_OR_DISCH == 'adm',
         TUNE_IDX == 135) %>%
  ggplot(aes(x=CONFIDENCE)) +
  geom_line(aes(y = ACCURACY_mean,color=factor(WINDOW_IDX))) +
  geom_ribbon(aes(ymin = ACCURACY_lo, ymax = ACCURACY_hi,fill=factor(WINDOW_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  geom_segment(x = 0, y = 0, xend = 1, yend = 1,alpha = 0.5,linetype = "dashed",size=.75/.pt, color = 'gray')+
  xlab("Confidence") +
  ylab("Accuracy") +
  coord_cartesian(ylim = c(0,1),xlim = c(0,1))+
  facet_wrap(~factor(WINDOW_IDX),
             ncol=3,
             scales = 'free')+
  theme_classic()+
  theme(strip.text = element_text(size=11, color = "black",face = 'bold'), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.text.x = element_text(size = 10, color = "black"),
        axis.text.y = element_text(size = 10, color = "black"),
        axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
        axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
        strip.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
        aspect.ratio = 1,
        plot.margin=grid::unit(c(0,0,0,0), "mm"),
        legend.position = 'none',
        legend.title = element_blank(),
        legend.text=element_blank(),
        axis.line = element_blank())

# discharge
disch.acc.conf.plot <- acc.conf.calib.curves %>%
  filter(ADM_OR_DISCH == 'disch',
         TUNE_IDX == 135) %>%
  ggplot(aes(x=CONFIDENCE)) +
  geom_line(aes(y = ACCURACY_mean,color=factor(WINDOW_IDX))) +
  geom_ribbon(aes(ymin = ACCURACY_lo, ymax = ACCURACY_hi,fill=factor(WINDOW_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  geom_segment(x = 0, y = 0, xend = 1, yend = 1,alpha = 0.5,linetype = "dashed",size=.75/.pt, color = 'gray')+
  xlab("Confidence") +
  ylab("Accuracy") +
  coord_cartesian(ylim = c(0,1),xlim = c(0,1))+
  facet_wrap(~factor(WINDOW_IDX),
             ncol=3,
             scales = 'free')+
  theme_classic()+
  theme(strip.text = element_text(size=11, color = "black",face = 'bold'), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.text.x = element_text(size = 10, color = "black"),
        axis.text.y = element_text(size = 10, color = "black"),
        axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
        axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
        strip.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
        aspect.ratio = 1,
        plot.margin=grid::unit(c(0,0,0,0), "mm"),
        legend.position = 'none',
        legend.title = element_blank(),
        legend.text=element_blank(),
        axis.line = element_blank())

#### threshold curves
thresh.calib.curves <- read.csv('../model_performance/v6-0/CI_thresh_calibration_curves.csv')

# admission
adm.thresh.plot <- thresh.calib.curves %>%
  filter(ADM_OR_DISCH == 'adm',
         TUNE_IDX == 135,
         WINDOW_IDX <= 24) %>%
  ggplot(aes(x=PRED_PROB)) +
  geom_line(aes(y = TRUE_PROB_mean,color=factor(WINDOW_IDX))) +
  geom_ribbon(aes(ymin = TRUE_PROB_lo, ymax = TRUE_PROB_hi,fill=factor(WINDOW_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  geom_segment(x = 0, y = 0, xend = 1, yend = 1,alpha = 0.5,linetype = "dashed",size=.75/.pt, color = 'gray')+
  xlab("Predicted Probability") +
  ylab("Observed Probability") +
  coord_cartesian(ylim = c(0,1),xlim = c(0,1))+
  facet_wrap(~THRESHOLD,
             ncol=3,
             scales = 'free')+
  theme_classic()+
  theme(strip.text = element_text(size=11, color = "black",face = 'bold'), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.text.x = element_text(size = 10, color = "black"),
        axis.text.y = element_text(size = 10, color = "black"),
        axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
        axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
        strip.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
        aspect.ratio = 1,
        plot.margin=grid::unit(c(0,0,0,0), "mm"),
        legend.position = 'bottom',
        legend.title = element_text(size = 11, color = "black",face = 'bold'),
        legend.text=element_text(size = 10, color = "black"),
        axis.line = element_blank())

# discharge
disch.thresh.plot <- thresh.calib.curves %>%
  filter(ADM_OR_DISCH == 'disch',
         TUNE_IDX == 135,
         WINDOW_IDX <= 24) %>%
  ggplot(aes(x=PRED_PROB)) +
  geom_line(aes(y = TRUE_PROB_mean,color=factor(WINDOW_IDX))) +
  geom_ribbon(aes(ymin = TRUE_PROB_lo, ymax = TRUE_PROB_hi,fill=factor(WINDOW_IDX)),alpha=.2,size=.75/.pt,color=NA) + 
  geom_segment(x = 0, y = 0, xend = 1, yend = 1,alpha = 0.5,linetype = "dashed",size=.75/.pt, color = 'gray')+
  xlab("Predicted Probability") +
  ylab("Observed Probability") +
  coord_cartesian(ylim = c(0,1),xlim = c(0,1))+
  facet_wrap(~THRESHOLD,
             ncol=3,
             scales = 'free')+
  theme_classic()+
  theme(strip.text = element_text(size=11, color = "black",face = 'bold'), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.text.x = element_text(size = 10, color = "black"),
        axis.text.y = element_text(size = 10, color = "black"),
        axis.title.x = element_text(size = 12, color = "black",face = 'bold'),
        axis.title.y = element_text(size = 12, color = "black",face = 'bold'),
        strip.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
        aspect.ratio = 1,
        plot.margin=grid::unit(c(0,0,0,0), "mm"),
        legend.position = 'bottom',
        legend.title = element_text(size = 11, color = "black",face = 'bold'),
        legend.text=element_text(size = 10, color = "black"),
        axis.line = element_blank())