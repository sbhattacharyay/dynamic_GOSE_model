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
  coord_cartesian(xlim=c(0,30)) + 
  scale_x_continuous(breaks=seq(0,30,by=1),expand = expansion(mult = c(0, .01)))+
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