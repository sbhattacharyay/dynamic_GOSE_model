#### Master Script 06: Visualise study results for manuscript ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Figure 1
# III. Figure 2
# IV. Figure 3
# V. Figure 4
# VI. Supplementary Figures 5 and 6
# VII. Supplementary Figures 8 and 9
# VIII. Supplementary Figure 7
# IX. Supplementary Appendix
# X. Unused
# XI. Addendum to figure 2
# XII. Supplementary Figures 2 and 3
# XIII. Supplementary Figure 4
# XIV. Appendix variable lists
# XV. Create table of cutoffs defining significant transitions
# XVI. Stacked proportion barplots of characteristics over time
# XVII. Sensitivity analysis difference plots

### I. Initialisation
# Import necessary libraries
library(tidyverse)
library(readxl)
library(plotly)
library(ggbeeswarm)
library(cowplot)
library(rvg)
library(svglite)
library(openxlsx)
library(gridExtra)
library(extrafont)

# Import custom plotting functions
source('functions/plotting.R')

### II. Figure 1
## Prepare overall calibration dataframes
# Calibration curve results
calib.curves.CIs <- read.csv('../model_performance/v6-0/test_set_calib_curves_CI.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  mutate(VERSION = 6,
         SinceAdmission = WINDOW_IDX > 0)
calib.curves.CIs$WINDOW_IDX[!calib.curves.CIs$SinceAdmission] <- calib.curves.CIs$WINDOW_IDX[!calib.curves.CIs$SinceAdmission] + 1
calib.curves.CIs <- calib.curves.CIs %>%
  mutate(DaysAfterICUAdmission = WINDOW_IDX/12,
         hi = pmax(0,hi),
         lo = pmax(0,lo),
         median = pmax(0,median)) %>%
  mutate(hi = pmin(1,hi),
         lo = pmin(1,lo),
         median = pmin(1,median))

# Calibration metrics results
calibration.CIs <- read.csv('../model_performance/v6-0/test_set_calibration_CI.csv',
                            na.strings = c("NA","NaN","", " ")) %>%
  mutate(VERSION = 6,
         SinceAdmission = WINDOW_IDX > 0)
calibration.CIs$WINDOW_IDX[!calibration.CIs$SinceAdmission] <- calibration.CIs$WINDOW_IDX[!calibration.CIs$SinceAdmission] + 1
calibration.CIs <- calibration.CIs %>%
  mutate(DaysAfterICUAdmission = WINDOW_IDX/12)

# Static calibration metric results
static.calibration.CIs <- read.csv('../model_performance/v6-0/static_set_calibration_CI.csv',
                                   na.strings = c("NA","NaN","", " ")) %>%
  mutate(VERSION = 'Static',
         SinceAdmission = WINDOW_IDX > 0)
static.calibration.CIs$WINDOW_IDX[!static.calibration.CIs$SinceAdmission] <- static.calibration.CIs$WINDOW_IDX[!static.calibration.CIs$SinceAdmission] + 1
static.calibration.CIs <- static.calibration.CIs %>%
  mutate(DaysAfterICUAdmission = WINDOW_IDX/12)

# Baseline calibration metric results
baseline.calibration.CIs <- read.csv('../model_performance/BaselineComparison/test_set_calibration_CI.csv',
                                     na.strings = c("NA","NaN","", " ")) %>%
  mutate(VERSION = 'Baseline',
         SinceAdmission = WINDOW_IDX > 0)
baseline.calibration.CIs$WINDOW_IDX[!baseline.calibration.CIs$SinceAdmission] <- baseline.calibration.CIs$WINDOW_IDX[!baseline.calibration.CIs$SinceAdmission] + 1
baseline.calibration.CIs <- baseline.calibration.CIs %>%
  mutate(DaysAfterICUAdmission = WINDOW_IDX/12)

## Create threshold-level calibration slope plot
# Since admission calibration slope plot
since.adm.calib.slope <- rbind(calibration.CIs,baseline.calibration.CIs) %>%
  filter(SinceAdmission,
         METRIC == 'CALIB_SLOPE',
         THRESHOLD=='Average') %>%
  ggplot() +
  coord_cartesian(xlim=c(0,7),ylim=c(0,1.5)) +
  geom_vline(xintercept = 0.33333333, color='dark gray',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_vline(xintercept = 1, color='#bc5090',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 1, color='#ffa600',alpha = 1, size=2/.pt)+
  # geom_line(data=old.calibration.CIs,aes(x=DaysAfterICUAdmission,y=median),alpha = 1, size=1.3/.pt,color='dark gray')+
  # geom_line(data=old.calibration.CIs,aes(x=DaysAfterICUAdmission,y=lo),alpha = 1, size=1.3/.pt,color='dark gray',linetype = "dashed")+
  # geom_line(data=old.calibration.CIs,aes(x=DaysAfterICUAdmission,y=hi),alpha = 1, size=1.3/.pt,color='dark gray',linetype = "dashed")+
  geom_line(aes(x=DaysAfterICUAdmission,y=median,color=VERSION),alpha = 1, size=1.3/.pt) + 
  geom_ribbon(aes(x=DaysAfterICUAdmission,ymin=lo,ymax=hi,fill=VERSION),alpha=.2,size=.75/.pt) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0,0))) +
  ylab('Calibration slope')+
  xlab('Days since ICU admission') + 
  scale_fill_manual(values = c("#003f5c", "#bc5090"))+
  scale_color_manual(values = c("#003f5c", "#bc5090"))+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'none'
  )

# Before discharge calibration slope plot
before.disch.calib.slope <- rbind(calibration.CIs,baseline.calibration.CIs) %>%
  filter(!SinceAdmission,
         METRIC == 'CALIB_SLOPE',
         THRESHOLD=='Average') %>%
  mutate(DaysBeforeICUDischarge = abs(DaysAfterICUAdmission)) %>%
  ggplot() +
  coord_cartesian(xlim=c(7,0), ylim = c(0,1.5)) +
  geom_hline(yintercept = 1, color='#ffa600',alpha = 1, size=2/.pt) +
  # geom_line(data=old.calibration.CIs,aes(x=DaysBeforeICUDischarge,y=median),alpha = 1, size=1.3/.pt,color='dark gray')+
  # geom_line(data=old.calibration.CIs,aes(x=DaysBeforeICUDischarge,y=lo),alpha = 1, size=1.3/.pt,color='dark gray',linetype = "dashed")+
  # geom_line(data=old.calibration.CIs,aes(x=DaysBeforeICUDischarge,y=hi),alpha = 1, size=1.3/.pt,color='dark gray',linetype = "dashed")+
  geom_line(aes(x=DaysBeforeICUDischarge,y=median,color=VERSION),alpha = 1, size=1.3/.pt) + 
  geom_ribbon(aes(x=DaysBeforeICUDischarge,ymin=lo,ymax=hi,fill=VERSION),alpha=.2,size=.75/.pt) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0,0))) +
  ylab('Calibration slope') +
  xlab('Days before ICU discharge') +
  scale_fill_manual(values = c("#003f5c", "#bc5090"))+
  scale_color_manual(values = c("#003f5c", "#bc5090"))+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'none'
  )

# Create directory for current date and save post-admission and pre-discharge calibration slope plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'since_adm_calib_slope.svg'),since.adm.calib.slope,device=svglite,units='in',dpi=600,width=3.7,height=1.38)
ggsave(file.path('../plots',Sys.Date(),'before_disch_calib_slope.svg'),before.disch.calib.slope,device=svglite,units='in',dpi=600,width=3.7,height=1.38)

## Create threshold-level calibration curve plot
# Since admission calibration curve plot
since.adm.calib.curves <- calib.curves.CIs %>%
  filter(WINDOW_IDX %in% c(1,4,12,24)) %>%
  mutate(WINDOW_IDX = plyr::mapvalues(WINDOW_IDX,
                                      from=c(1,4,12,24),
                                      to=c('2 hrs.','8 hrs.','1 day','2 days'))) %>%
  mutate(WINDOW_IDX = fct_relevel(WINDOW_IDX,'2 hrs.','8 hrs.','1 day','2 days')) %>%
  ggplot(aes(x=100*PREDPROB)) +
  facet_wrap( ~ THRESHOLD,
              scales = 'free',
              ncol = 4) +
  coord_cartesian(ylim = c(0,100),xlim = c(0,100))+
  geom_segment(x = 0, y = 0, xend = 100, yend = 100,alpha = 0.5,linetype = "dashed",size=.75/.pt, color = 'gray')+
  geom_ribbon(aes(ymin = 100*lo, ymax = 100*hi, fill = WINDOW_IDX), alpha = 0.3,size=.75/.pt,color=NA) +
  geom_line(aes(y = 100*median, color = WINDOW_IDX), alpha = 1, size=1.3/.pt) +
  scale_x_continuous(expand = expansion(mult = c(.01, .01))) +
  scale_y_continuous(expand = expansion(mult = c(.01, .01))) +
  guides(fill=guide_legend(nrow=2,byrow=TRUE),color=guide_legend(nrow=2,byrow=TRUE)) +
  scale_fill_manual(name = "Time since ICU admission",
                    values = c("#003f5c", "#7a5195", "#ef5675",'#ffa600'))+
  scale_color_manual(name = "Time since ICU admission",
                     values = c("#003f5c", "#7a5195", "#ef5675",'#ffa600'))+
  xlab("Predicted probability") +
  ylab("Observed probability") +
  theme_classic(base_family = 'Roboto Condensed') +
  theme(
    strip.text = element_text(size=6, color = "black",face = 'bold',margin = margin(b = .5)), 
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    panel.spacing = unit(0.05, "lines"),
    axis.text.x = element_text(size = 5, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 5, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    strip.background = element_blank(),
    aspect.ratio = 1,
    panel.border = element_rect(colour = 'black', fill=NA, size = 1/.pt),
    #plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    axis.line = element_blank(),
    legend.key.size = unit(1.3/.pt,"line")
  )

# Create directory for current date and save post-admission calibration curve plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'since_adm_calib_curves.svg'),since.adm.calib.curves,device=svglite,units='in',dpi=600,width=7.5,height=4.46)

# Calculate integrated calibration index (ICI) for plot
since.adm.ICI <- calibration.CIs %>%
  filter(THRESHOLD != 'Average',
         METRIC == 'ICI',
         WINDOW_IDX %in% c(1,4,12,24)) %>% 
  mutate(WINDOW_IDX = plyr::mapvalues(WINDOW_IDX,
                                      from=c(1,4,12,24),
                                      to=c('2 hrs.','8 hrs.','1 day','2 days'))) %>%
  mutate(WINDOW_IDX = fct_relevel(WINDOW_IDX,'2 hrs.','8 hrs.','1 day','2 days')) %>%
  mutate(formatted = sprintf('%s: %.2f (%.2f–%.2f)',WINDOW_IDX,median,lo,hi)) %>%
  arrange(THRESHOLD,WINDOW_IDX)

### III. Figure 2
## Prepare overall performance dataframes
# Discrimination results
discrimination.CIs <- read.csv('../model_performance/v6-0/test_set_discrimination_CI.csv',
                               na.strings = c("NA","NaN","", " ")) %>%
  mutate(VERSION = 6,
         SinceAdmission = WINDOW_IDX > 0)
discrimination.CIs$WINDOW_IDX[!discrimination.CIs$SinceAdmission] <- discrimination.CIs$WINDOW_IDX[!discrimination.CIs$SinceAdmission] + 1
discrimination.CIs <- discrimination.CIs %>%
  mutate(DaysAfterICUAdmission = WINDOW_IDX/12)

# Static discrimination results
static.discrimination.CIs <- read.csv('../model_performance/v6-0/static_set_discrimination_CI.csv',
                                      na.strings = c("NA","NaN","", " "))  %>%
  mutate(VERSION = 'Static',
         SinceAdmission = WINDOW_IDX > 0)
static.discrimination.CIs$WINDOW_IDX[!static.discrimination.CIs$SinceAdmission] <- static.discrimination.CIs$WINDOW_IDX[!static.discrimination.CIs$SinceAdmission] + 1
static.discrimination.CIs <- static.discrimination.CIs %>%
  mutate(DaysAfterICUAdmission = WINDOW_IDX/12)

# Baseline discrimination results
baseline.discrimination.CIs <- read.csv('../model_performance/BaselineComparison/test_set_discrimination_CI.csv',
                                        na.strings = c("NA","NaN","", " ")) %>%
  mutate(VERSION = 'Baseline',
         SinceAdmission = WINDOW_IDX > 0)
baseline.discrimination.CIs$WINDOW_IDX[!baseline.discrimination.CIs$SinceAdmission] <- baseline.discrimination.CIs$WINDOW_IDX[!baseline.discrimination.CIs$SinceAdmission] + 1
baseline.discrimination.CIs <- baseline.discrimination.CIs %>%
  mutate(DaysAfterICUAdmission = WINDOW_IDX/12)

# Difference in discrimination between full-model and static variables
static.difference.CIs <- read.csv('../model_performance/v6-0/static_test_set_discrimination_difference_CI.csv',
                                  na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC == 'Somers D') %>%
  mutate(SinceAdmission = WINDOW_IDX > 0,
         VERSION = 'Static') %>%
  drop_na(median)
static.difference.CIs$WINDOW_IDX[!static.difference.CIs$SinceAdmission] <- static.difference.CIs$WINDOW_IDX[!static.difference.CIs$SinceAdmission] + 1
static.difference.CIs <- static.difference.CIs %>%
  mutate(DaysAfterICUAdmission = WINDOW_IDX/12)

# Difference in discrimination between full-model and IMPACT
baseline.difference.CIs <- read.csv('../model_performance/BaselineComparison/test_set_discrimination_difference_CI.csv',
                                    na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC == 'Somers D') %>%
  mutate(SinceAdmission = WINDOW_IDX > 0,
         VERSION = 'Baseline') %>%
  drop_na(median)
baseline.difference.CIs$WINDOW_IDX[!baseline.difference.CIs$SinceAdmission] <- baseline.difference.CIs$WINDOW_IDX[!baseline.difference.CIs$SinceAdmission] + 1
baseline.difference.CIs <- baseline.difference.CIs %>%
  mutate(DaysAfterICUAdmission = WINDOW_IDX/12)

## Create overall discrimination performance plots
# Since admission Somers' D plot
since.adm.somers <- rbind(discrimination.CIs,baseline.discrimination.CIs) %>%
  filter(SinceAdmission,
         METRIC == 'Somers D') %>%
  ggplot() +
  coord_cartesian(xlim=c(0,7), ylim = c(27.5,55)) +
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(.0, .0)))+
  geom_vline(xintercept = 1, color='#bc5090',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  # geom_line(data=old.discrimination.CIs,aes(x=DaysAfterICUAdmission,y=100*median),alpha = 1, size=1.3/.pt,color='dark gray')+
  # geom_line(data=old.discrimination.CIs,aes(x=DaysAfterICUAdmission,y=100*lo),alpha = 1, size=1.3/.pt,color='dark gray',linetype = "dashed")+
  # geom_line(data=old.discrimination.CIs,aes(x=DaysAfterICUAdmission,y=100*hi),alpha = 1, size=1.3/.pt,color='dark gray',linetype = "dashed")+
  geom_line(aes(x=DaysAfterICUAdmission,y=100*median,color=VERSION),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=DaysAfterICUAdmission,ymin=lo*100,ymax=hi*100,fill=VERSION),alpha=.2) +
  ylab('Explanation of ordinal GOSE (%)')+
  xlab('Days since ICU admission')+
  scale_fill_manual(values = c("#003f5c", "#bc5090"))+
  scale_color_manual(values = c("#003f5c", "#bc5090"))+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'none'
  )

# Before discharge Somers' D plot
before.disch.somers <- rbind(discrimination.CIs,baseline.discrimination.CIs) %>%
  filter(!SinceAdmission,
         METRIC == 'Somers D') %>%
  mutate(DaysBeforeICUDischarge = abs(DaysAfterICUAdmission)) %>%
  ggplot() +
  scale_x_reverse(expand = expansion(mult = c(.0, .0)),breaks=seq(0,7,by=1))+
  coord_cartesian(xlim=c(7,0), ylim = c(27.5,55)) +
  # geom_line(data=old.discrimination.CIs,aes(x=DaysBeforeICUDischarge,y=100*median),alpha = 1, size=1.3/.pt,color='dark gray')+
  # geom_line(data=old.discrimination.CIs,aes(x=DaysBeforeICUDischarge,y=100*lo),alpha = 1, size=1.3/.pt,color='dark gray',linetype = "dashed")+
  # geom_line(data=old.discrimination.CIs,aes(x=DaysBeforeICUDischarge,y=100*hi),alpha = 1, size=1.3/.pt,color='dark gray',linetype = "dashed")+
  geom_line(aes(x=DaysBeforeICUDischarge,y=100*median,color=VERSION),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=DaysBeforeICUDischarge,ymin=lo*100,ymax=hi*100,fill=VERSION),alpha=.2) +
  ylab('Explanation of ordinal GOSE (%)')+
  xlab('Days before ICU discharge')+
  scale_fill_manual(values = c("#003f5c", "#bc5090"))+
  scale_color_manual(values = c("#003f5c", "#bc5090"))+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'none'
  )

# Create directory for current date and save post-admission and pre-discharge Somers' D plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'since_adm_somers.svg'),since.adm.somers,device= svglite,units='in',dpi=600,width=3.7,height = 1.38)
ggsave(file.path('../plots',Sys.Date(),'before_disch_somers.svg'),before.disch.somers,device= svglite,units='in',dpi=600,width=3.7,height = 1.38)

## Create overall discrimination difference over baseline plots
# Since admission difference in Somers' D plot
since.adm.baseline.diff.somers <- rbind(baseline.difference.CIs,static.difference.CIs) %>%
  filter(SinceAdmission) %>%
  ggplot() +
  coord_cartesian(xlim=c(0,7), ylim = c(0,15)) +
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(.0, .0)))+
  geom_vline(xintercept = 1, color='#bc5090',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_line(aes(x=DaysAfterICUAdmission,y=100*median,color=VERSION),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=DaysAfterICUAdmission,ymin=lo*100,ymax=hi*100,fill=VERSION),alpha=.2) +
  ylab('Added explanation of ordinal GOSE (d%)')+
  xlab('Days since ICU admission')+
  scale_fill_manual(values = c("#bc5090","#003f5c"))+
  scale_color_manual(values = c("#bc5090","#003f5c"))+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'none'
  )

# Before discharge difference in Somers' D plot
before.disch.baseline.diff.somers <- rbind(baseline.difference.CIs,static.difference.CIs) %>%
  filter(!SinceAdmission) %>%
  mutate(DaysBeforeICUDischarge = abs(DaysAfterICUAdmission)) %>%
  ggplot() +
  scale_x_reverse(expand = expansion(mult = c(.0, .0)),breaks=seq(0,7,by=1))+
  coord_cartesian(xlim=c(7,0), ylim = c(0,15)) +
  geom_line(aes(x=DaysBeforeICUDischarge,y=100*median,color=VERSION),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=DaysBeforeICUDischarge,ymin=lo*100,ymax=hi*100,fill=VERSION),alpha=.2) +
  ylab('Added explanation of ordinal GOSE (d%)')+
  xlab('Days before ICU discharge')+
  scale_fill_manual(values = c("#bc5090","#003f5c"))+
  scale_color_manual(values = c("#bc5090","#003f5c"))+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'none'
  )

# Create directory for current date and save post-admission and pre-discharge Somers' D plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'since_adm_baseline_diff_somers.svg'),since.adm.baseline.diff.somers,device= svglite,units='in',dpi=600,width=3.7,height = 1.38)
ggsave(file.path('../plots',Sys.Date(),'before_disch_baseline_diff_somers.svg'),before.disch.baseline.diff.somers,device= svglite,units='in',dpi=600,width=3.7,height = 1.38)

## Discharge cutoff sensitivity analysis
# Load and prepare cutoff mean analysis dataframe
cutoff.mean.analysis <- read.csv('../model_performance/v6-0/sensitivity_cutoff_mean_difference_CI.csv',
                                 na.strings = c("NA","NaN","", " ")) %>%
  mutate(CutoffDays = CUTOFF_IDX/12)

# Load and prepare cutoff discrimination difference dataframe
cutoff.discrimination <- read.csv('../model_performance/v6-0/sensitivity_cutoff_discrimination_CI.csv',
                                  na.strings = c("NA","NaN","", " ")) %>%
  mutate(VERSION = 6,
         SinceAdmission = WINDOW_IDX > 0)
cutoff.discrimination$WINDOW_IDX[!cutoff.discrimination$SinceAdmission] <- cutoff.discrimination$WINDOW_IDX[!cutoff.discrimination$SinceAdmission] + 1
cutoff.discrimination <- cutoff.discrimination %>%
  mutate(DaysAfterICUAdmission = WINDOW_IDX/12)

# Trajectory of mean difference in Somers' D vs. discharge cutoff
cutoff.mean.somers.diff.plot <- cutoff.mean.analysis %>%
  filter(METRIC == 'Somers D',
         CutoffDays>1) %>%
  ggplot() +
  coord_cartesian(xlim=c(0,7), ylim = c(-15,19)) +
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(.0, .0)))+
  geom_vline(xintercept = 13/12, color='dark gray',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 0, color='#ffa600',alpha = 1, size=2/.pt) +
  geom_line(aes(x=CutoffDays,y=100*median),alpha = 1, size=1.3/.pt,color="#003f5c")+
  geom_ribbon(aes(x=CutoffDays,ymin=lo*100,ymax=hi*100),alpha=.2,fill="#003f5c") +
  ylab('Mean difference in ordinal GOSE explanation (%)')+
  xlab('ICU stay duration cutoff (days)')+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'none'
  )

# Create directory for current date and save trajectory of mean difference in Somers' D vs. discharge cutoff
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'cutoff_diff_Somers_plot.svg'),cutoff.mean.somers.diff.plot,device= svglite,units='in',dpi=600,width=3.7,height = 1.53)




cutoff.discrimination %>%
  filter(CUTOFF_IDX == 73,
         variable != 'CUTOFF_DIFFERENCE',
         METRIC == 'Somers D') %>%
  ggplot() +
  #coord_cartesian(xlim=c(0,7), ylim = c(0,10)) +
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(.0, .0)))+
  geom_line(aes(x=DaysAfterICUAdmission,y=100*median,color=variable),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=DaysAfterICUAdmission,ymin=lo*100,ymax=hi*100,fill=variable),alpha=.2) +
  ylab('Explanation of ordinal GOSE (%)')+
  xlab('Days since ICU admission')+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold')
  )



## Create bi-directional distribution plot of significant transitions
# Load dataframe of significant transitions
sig.transitions.df <- read.csv('../model_interpretations/v6-0/timeSHAP/significant_transition_points.csv',
                               na.strings = c("NA","NaN","", " ")) %>%
  # filter(WindowIdx > 4) %>%
  mutate(DaysAfterICUAdmission = WindowIdx/12)

# Partition dataframe by above and below threshold
above.transitions.df <- sig.transitions.df %>%
  filter(Above=='True')
below.transitions.df <- sig.transitions.df %>%
  filter(Above=='False')

# Create bidirectional bar plot
transition.distrib.plot <- ggplot() +
  annotate('rect',xmin=(1/12),xmax=(4.5/12),ymin=-1,ymax=1,alpha=0.3,fill='#de425b') + 
  geom_density(data = above.transitions.df, aes(x=DaysAfterICUAdmission, y=..scaled..),fill=NA,color='#003f5c',alpha=0.2,trim=TRUE, size=1.3/.pt) + 
  geom_density(data = below.transitions.df, aes(x=DaysAfterICUAdmission, y=-..scaled..),fill=NA,color='#76488b',alpha=0.2,trim=TRUE, size=1.3/.pt) +
  geom_histogram(data = above.transitions.df, aes(x=DaysAfterICUAdmission, y=..ndensity..),alpha=0.2,fill='#003f5c',color='#003f5c',binwidth = 1/12, size=1/.pt) + 
  geom_histogram(data = below.transitions.df, aes(x=DaysAfterICUAdmission, y=-..ndensity..),alpha=0.2,fill='#76488b',color='#76488b',binwidth = 1/12, size=1/.pt) +
  geom_segment(data = above.transitions.df, aes(x = median(DaysAfterICUAdmission),xend = median(DaysAfterICUAdmission)),y = 0,yend = 1.1,colour = "orange",alpha = 1, size=1.3/.pt,linetype = "dashed") +
  geom_segment(data = below.transitions.df, aes(x = median(DaysAfterICUAdmission),xend = median(DaysAfterICUAdmission)),y = 0,yend = -1.1,colour = "orange",alpha = 1, size=1.3/.pt,linetype = "dashed") +
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(.0, .0))) +
  coord_cartesian(xlim=c(0,7), ylim = c(-1,1)) +
  xlab('Days since ICU admission')+
  ggtitle('Scaled density of significant transitions')+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.minor.x = element_blank(),
        axis.text.y = element_blank(),
        axis.title.y = element_blank(),
        plot.title = element_text(size = 7, color = "black",face = 'bold',hjust = 0.5,margin = margin(b = 0,r = 0)),
        axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
        axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)))

# Create directory for current date and save post-admission and pre-discharge Somers' D plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'transition_distributions.svg'),transition.distrib.plot,device= svglite,units='in',dpi=600,width=3.7,height = 1.53)

### IV. Figure 3
## Prepare dataframe of filtered TimeSHAP values for plotting
# Load TimeSHAP value dataframe
filt.timeSHAP.df <- read.csv('../model_interpretations/v6-0/timeSHAP/filtered_plotting_timeSHAP_values.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  filter(Threshold == 'ExpectedValue') %>%
  mutate(Baseline = as.logical(Baseline),
         Numeric = as.logical(Numeric),
         GROUPS = case_when((RankIdx >= 11) ~ 'Top',
                            (RankIdx <= 10) ~'Bottom')) %>%
  mutate(GROUPS = factor(GROUPS,levels=c('Top','Middle','Bottom')))

# Isolate and save unique `BaseTokens` to manually create labels and designate ordered variables
exp.GOSE.var.df <- filt.timeSHAP.df %>%
  select(Baseline,Numeric,RankIdx,BaseToken) %>%
  unique() %>%
  mutate(PLOT_LABEL = '',
         ORDERED = case_when(Numeric ~ TRUE,
                             !Numeric ~ NA))
write.xlsx(exp.GOSE.var.df,'../model_interpretations/v6-0/timeSHAP/expected_GOSE_timeSHAP_labels.xlsx') 

# Load manually created labels of unique `BaseTokens` in plotting datafame
exp.GOSE.var.df <- read_xlsx('../model_interpretations/v6-0/timeSHAP/expected_GOSE_timeSHAP_labels_filled.xlsx')

# Merge manually created labels to filtered TimeSHAP plotting dataframe
filt.timeSHAP.df <- filt.timeSHAP.df %>%
  left_join(exp.GOSE.var.df)

# Isolate and save unique `Tokens` to manually verify and fill variable order (if applicable)
exp.GOSE.token.df <- filt.timeSHAP.df %>%
  select(RankIdx,PLOT_LABEL,BaseToken,Token,Baseline,Numeric,Missing,ORDERED,TokenRankIdx) %>%
  unique() %>%
  mutate(OrderIdx = case_when((!ORDERED)|Numeric ~ (TokenRankIdx-1)))
write.xlsx(exp.GOSE.token.df,'../model_interpretations/v6-0/timeSHAP/expected_GOSE_timeSHAP_orders.xlsx') 

# Load manually inspected orders of unique `Tokens` in plotting datafame
exp.GOSE.token.df <- read_xlsx('../model_interpretations/v6-0/timeSHAP/expected_GOSE_timeSHAP_orders_filled.xlsx') 

# Calculate the number of unique known values per predictor
max.order.indices <- exp.GOSE.token.df %>%
  group_by(BaseToken) %>%
  summarise(MaxOrderIdx = max(OrderIdx))

# Merge manually created labels to filtered TimeSHAP plotting dataframe
filt.timeSHAP.df <- filt.timeSHAP.df %>%
  left_join(exp.GOSE.token.df) %>%
  left_join(max.order.indices) %>%
  arrange(TUNE_IDX,Threshold,RankIdx,OrderIdx) %>%
  mutate(ColorScale = OrderIdx/MaxOrderIdx) %>%
  mutate(ColorScale = case_when(is.na(ColorScale)~1,
                                ((!is.na(ColorScale))&(ColorScale>=0))~ColorScale))

# Complete formatting dataframe prior to plotting
filt.timeSHAP.df <- filt.timeSHAP.df %>%
  mutate(Baseline = recode(as.character(Baseline),'TRUE'='Static','FALSE'='Dynamic')) %>%
  mutate(Baseline = fct_relevel(Baseline, 'Static', 'Dynamic')) %>%
  mutate(PLOT_LABEL = fct_reorder(PLOT_LABEL, RankIdx))

# Create feature importance beeswarm plot for static predictors
static.timeshap.plot <- filt.timeSHAP.df %>%
  filter(Baseline=='Static',
         abs(SHAP) <= 0.5) %>%
  ggplot() +
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_quasirandom(aes(y=PLOT_LABEL,x=SHAP,color=ColorScale),groupOnX=FALSE,varwidth = FALSE,alpha = 0.6,stroke = 0,size=.75) + 
  scale_color_gradient2(na.value='#488f31',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=.5,limits = c(0,1),breaks = c(0.05,.95), labels = c('Low','High')) +
  theme_minimal(base_family = 'Roboto Condensed') +
  guides(color = guide_colourbar(title='Feature Value',title.vjust=1,barwidth = 10, barheight = .25,ticks = FALSE))+
  facet_grid(rows = vars(GROUPS), scales = 'free_y', switch = 'y', space = 'free_y') +
  theme(
    strip.background = element_blank(),
    strip.text = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(size = 5, color = 'black'),
    # axis.text.y = element_text(size = 6, color = 'black',angle = 30, hjust=1),
    axis.text.y = element_blank(),
    axis.title.x = element_blank(),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    # legend.position = 'bottom',
    legend.position = 'none',
    panel.grid.major.y = element_blank(),
    panel.spacing = unit(10, 'points'),
    #legend.key.size = unit(1.3/.pt,'line'),
    # legend.title = element_text(size = 7, color = 'black',face = 'bold'),
    # legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm")
  )

# Create feature importance beeswarm plot for dynamic predictors
dynamic.timeshap.plot <- filt.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         SHAP >= -0.25,
         SHAP <= .375) %>%
  ggplot() +
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_quasirandom(aes(y=PLOT_LABEL,x=SHAP,color=ColorScale),groupOnX=FALSE,varwidth = FALSE,alpha = 0.6,stroke = 0,size=.75) + 
  scale_color_gradient2(na.value='#488f31',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=.5,limits = c(0,1),breaks = c(0.05,.95), labels = c('Low','High')) +
  theme_minimal(base_family = 'Roboto Condensed') +
  guides(color = guide_colourbar(title='Feature Value',title.vjust=1,barwidth = 10, barheight = .25,ticks = FALSE))+
  facet_grid(rows = vars(GROUPS), scales = 'free_y', switch = 'y', space = 'free_y') +
  theme(
    strip.background = element_blank(),
    strip.text = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(size = 5, color = 'black'),
    # axis.text.y = element_text(size = 6, color = 'black',angle = 30, hjust=1),
    axis.text.y = element_blank(),
    axis.title.x = element_blank(),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'none',
    panel.grid.major.y = element_blank(),
    panel.spacing = unit(10, 'points'),
    plot.margin=grid::unit(c(0,2,0,0), "mm")
  )

# Extract color bar legend
plot.legend <- filt.timeSHAP.df %>%
  filter(Baseline=='Static',
         abs(SHAP) <= 0.5) %>%
  ggplot() +
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_quasirandom(aes(y=PLOT_LABEL,x=SHAP,color=ColorScale),groupOnX=FALSE,varwidth = FALSE,alpha = 0.6,stroke = 0,size=.75) +
  scale_color_gradient2(na.value='black',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=.5,limits = c(0,1),breaks = c(0.05,.95), labels = c('Low','High')) +
  theme_minimal(base_family = 'Roboto Condensed') +
  guides(color = guide_colourbar(title='Feature Value',title.vjust=1,barwidth = 10, barheight = .25,ticks = FALSE))+
  facet_grid(rows = vars(GROUPS), scales = 'free_y', switch = 'y', space = 'free_y') +
  theme(
    strip.background = element_blank(),
    strip.text = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(size = 5, color = 'black'),
    # axis.text.y = element_text(size = 6, color = 'black',angle = 30, hjust=1),
    axis.text.y = element_blank(),
    axis.title.x = element_blank(),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    # legend.position = 'none',
    panel.grid.major.y = element_blank(),
    panel.spacing = unit(10, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(size = 7, color = 'black',face = 'bold'),
    legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

# Create directory for current date and save feature-level TimeSHAP plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'static_timeshap.png'),static.timeshap.plot,units='in',dpi=600,height=3.38,width=2.47)
ggsave(file.path('../plots',Sys.Date(),'dynamic_timeshap.png'),dynamic.timeshap.plot,units='in',dpi=600,height=3.38,width=2.47)

## Prepare dataframe of event-level TimeSHAP values for plotting
# Load and format event TimeSHAP value dataframe
event.timeSHAP.df <- read.csv('../model_interpretations/v6-0/timeSHAP/filtered_plotting_event_timeSHAP_values.csv',
                              na.strings = c("NA","NaN","", " ")) %>%
  filter(Threshold == 'ExpectedValue') %>%
  mutate(HoursBeforeTransition = sprintf('%d–%d',abs(2*(TimePreTransition+1)),abs(2*TimePreTransition))) %>%
  mutate(HoursBeforeTransition = fct_reorder(HoursBeforeTransition,TimePreTransition))

# Create event importance violin plots for points before transition
event.timeshap.violin.plot <- event.timeSHAP.df %>%
  ggplot(aes(x = HoursBeforeTransition, y = absSHAP)) +
  geom_violin(scale = "width",trim=TRUE,fill='#9cc3dc',lwd=1.3/.pt) +
  geom_quasirandom(varwidth = TRUE,alpha = 0.15,stroke = 0,size=.5) +
  geom_boxplot(width=0.1,outlier.shape = NA,lwd=1.3/.pt) +
  coord_cartesian(ylim = c(0,1.25)) +
  ylab('Absolute effect on average prognosis (|TimeSHAP|)') +
  xlab('Hours before significant transition') +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold')
  )

# Create directory for current date and save event-level TimeSHAP plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'event_timeshap.svg'),event.timeshap.violin.plot,device= svglite,units='in',dpi=600,width=7.5,height = 2.3)

### V. Figure 4
## Prepare dataframe of filtered test set predictions for plotting
# Load filtered test set predictions
plotting.test.preds <- read.csv('../model_outputs/v6-0/plotting_test_predictions.csv',
                                na.strings = c("NA","NaN","", " ")) %>%
  select(WindowIdx,REPEAT,FOLD,Pr.GOSE.1.,Pr.GOSE.3.,Pr.GOSE.4.,Pr.GOSE.5.,Pr.GOSE.6.,Pr.GOSE.7.) %>%
  pivot_longer(cols=c(Pr.GOSE.1.,Pr.GOSE.3.,Pr.GOSE.4.,Pr.GOSE.5.,Pr.GOSE.6.,Pr.GOSE.7.),
               names_to = "Threshold",
               values_to = "Probability") %>%
  mutate(Threshold = plyr::mapvalues(Threshold,
                                     from = c("Pr.GOSE.1.","Pr.GOSE.3.","Pr.GOSE.4.","Pr.GOSE.5.","Pr.GOSE.6.","Pr.GOSE.7."),
                                     to = c("GOSE>1","GOSE>3","GOSE>4","GOSE>5","GOSE>6","GOSE>7")))

# Calculate mean and variance of prediction values
summ.plotting.test.preds <- plotting.test.preds %>%
  group_by(WindowIdx,Threshold) %>%
  summarise(meanProb = 100*mean(Probability),
            stdProb = 100*sd(Probability)) %>%
  rowwise() %>%
  mutate(hiProb = min(meanProb+stdProb,100),
         loProb = max(meanProb-stdProb,0),
         DaysAfterICUAdmission = WindowIdx/12)

# Create line plots for individual patient trajectory
indiv.pt.trajectory <- summ.plotting.test.preds %>%
  ggplot() +
  coord_cartesian(ylim = c(0,100)) +
  scale_x_continuous(breaks=seq(0,9,by=1),expand = expansion(mult = c(.01, .01)))+
  scale_fill_manual(values=c('#003f5c','#444e86','#955196','#dd5182','#ff6e54','#ffa600'))+
  scale_color_manual(values=c('#003f5c','#444e86','#955196','#dd5182','#ff6e54','#ffa600'))+
  geom_hline(yintercept = 50, color='dark gray',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_vline(xintercept = max(summ.plotting.test.preds$DaysAfterICUAdmission),
             color='orange', 
             alpha = 1,
             size=1.3/.pt,
             linetype = "twodash")+
  geom_line(aes(x=DaysAfterICUAdmission,y=meanProb,color=Threshold),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=DaysAfterICUAdmission,ymin=loProb,ymax=hiProb,fill=Threshold),alpha=.2) +
  annotate('rect',xmin=(1/12),xmax=(3/12),ymin=0,ymax=100,alpha=0.3,fill='#de425b') +
  annotate('rect',xmin=(3/12),xmax=(5/12),ymin=0,ymax=100,alpha=0.3,fill='#488f31') +
  annotate('rect',xmin=(40/12),xmax=(46/12),ymin=0,ymax=100,alpha=0.3,fill='#488f31') +
  annotate('rect',xmin=(76/12),xmax=(78/12),ymin=0,ymax=100,alpha=0.3,fill='#488f31') +
  annotate('rect',xmin=(88/12),xmax=(90/12),ymin=0,ymax=100,alpha=0.3,fill='#488f31') +
  ylab('Probability (%)')+
  xlab('Days since ICU admission')+
  guides(fill=guide_legend(byrow=TRUE),
         color=guide_legend(byrow=TRUE))+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'bottom',
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(size = 7, color = 'black',face = 'bold'),
    legend.text=element_text(size=6)
  )

# Create directory for current date and save individual patient trajectory
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'indiv_trajectory.svg'),indiv.pt.trajectory,device= svglite,units='in',dpi=600,width=7.5,height = 2.3)

## Create individual SHAP plots
# Load individual feature-level SHAP values
indiv.timeSHAP.df <- read.csv('../model_interpretations/v6-0/timeSHAP/individual_plotting_timeSHAP_values.csv',
                              na.strings = c("NA","NaN","", " "))

# Save Excel dataframe of tokens for manual labelling
indiv.timeSHAP.var.df <- indiv.timeSHAP.df %>%
  select(Baseline,RankIdx,Token) %>%
  mutate(PLOT_LABEL='')
write.xlsx(indiv.timeSHAP.var.df,'../model_interpretations/v6-0/timeSHAP/individual_timeSHAP_labels.xlsx') 

# Load manually created labels of unique `Tokens` in plotting datafame
indiv.timeSHAP.var.df <- read_xlsx('../model_interpretations/v6-0/timeSHAP/individual_timeSHAP_labels_filled.xlsx')

# Merge manually created labels to filtered TimeSHAP plotting dataframe
indiv.timeSHAP.df <- indiv.timeSHAP.df %>%
  left_join(indiv.timeSHAP.var.df)

# Complete formatting dataframe prior to plotting
indiv.timeSHAP.df <- indiv.timeSHAP.df %>%
  mutate(Baseline = recode(as.character(Baseline),'True'='Static','False'='Dynamic')) %>%
  mutate(Baseline = fct_relevel(Baseline, 'Static', 'Dynamic')) %>%
  mutate(PLOT_LABEL = fct_reorder(PLOT_LABEL, -RankIdx))
indiv.timeSHAP.df$SHAP[(indiv.timeSHAP.df$Token=='Others')&(indiv.timeSHAP.df$Baseline=='Static')] = indiv.timeSHAP.df$SHAP[(indiv.timeSHAP.df$Token=='Others')&(indiv.timeSHAP.df$Baseline=='Static')]/253
indiv.timeSHAP.df$SHAP[(indiv.timeSHAP.df$Token=='Others')&(indiv.timeSHAP.df$Baseline=='Dynamic')] = indiv.timeSHAP.df$SHAP[(indiv.timeSHAP.df$Token=='Others')&(indiv.timeSHAP.df$Baseline=='Dynamic')]/138

# Create local feature importance bar plot for static predictors
indiv.static.timeshap.plot <- indiv.timeSHAP.df %>%
  filter(Baseline=='Static') %>%
  ggplot() +
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_col(aes(y=PLOT_LABEL,x=SHAP),width=.6,fill='#003f5c') + 
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    strip.background = element_blank(),
    strip.text = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(size = 5, color = 'black'),
    axis.text.y = element_text(size = 6, color = 'black',angle = 30, hjust=1),
    # axis.text.y = element_blank(),
    axis.title.x = element_blank(),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    # legend.position = 'bottom',
    legend.position = 'none',
    panel.grid.major.y = element_blank(),
    panel.spacing = unit(10, 'points'),
    #legend.key.size = unit(1.3/.pt,'line'),
    # legend.title = element_text(size = 7, color = 'black',face = 'bold'),
    # legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

# Create local feature importance bar plot for dynamic predictors
indiv.dynamic.timeshap.plot <- indiv.timeSHAP.df %>%
  filter(Baseline=='Dynamic') %>%
  ggplot() +
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_col(aes(y=PLOT_LABEL,x=SHAP),width=.6,fill='#003f5c') + 
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    strip.background = element_blank(),
    strip.text = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(size = 5, color = 'black'),
    axis.text.y = element_text(size = 6, color = 'black',angle = 30, hjust=1),
    # axis.text.y = element_blank(),
    axis.title.x = element_blank(),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    # legend.position = 'bottom',
    legend.position = 'none',
    panel.grid.major.y = element_blank(),
    panel.spacing = unit(10, 'points'),
    #legend.key.size = unit(1.3/.pt,'line'),
    # legend.title = element_text(size = 7, color = 'black',face = 'bold'),
    # legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,0,0,0), "mm")
  )

# Create directory for current date and save individual feature-level bar plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'indiv_static_barplots.svg'),indiv.static.timeshap.plot,device= svglite,units='in',dpi=600,width=2.34,height = 1.72)
ggsave(file.path('../plots',Sys.Date(),'indiv_dynamic_barplots.svg'),indiv.dynamic.timeshap.plot,device= svglite,units='in',dpi=600,width=2.34,height = 1.72)

## Create individual eventSHAP heatmaps
# Load individual event-level SHAP values
indiv.event.timeSHAP.df <- read.csv('../model_interpretations/v6-0/timeSHAP/individual_plotting_event_timeSHAP_values.csv',
                                    na.strings = c("NA","NaN","", " ")) %>%
  mutate(HoursBeforeTransition = plyr::mapvalues(TimePreTransition,
                                                 from=c('-1','-2','-3','Pruned Events'),
                                                 to=c('0–2','2–4','4–6','Previous windows'))) %>%
  mutate(HoursBeforeTransition = factor(HoursBeforeTransition,levels=c('Previous windows','4–6','2–4','0–2')),
         SHAPLabel = sprintf('%.2f',SHAP))

# Create local event importance heat plot
indiv.event.timeshap.plot <- indiv.event.timeSHAP.df %>%
  mutate(x=0) %>%
  ggplot(aes(x=x)) +
  geom_tile(aes(y=HoursBeforeTransition,fill=SHAP)) + 
  geom_text(aes(y=HoursBeforeTransition,
                label=SHAPLabel,
                color = as.factor(as.integer(abs(SHAP)>.35))),family = 'Roboto Condensed',size=7/.pt) +
  scale_fill_gradient2(na.value='black',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=0,limits = c(-.7,.7),breaks=seq(-0.6,0.6,by=.15)) +
  scale_color_manual(values = c('black','white'),guide='none') +
  theme_minimal(base_family = 'Roboto Condensed') +
  guides(fill = guide_colourbar(title='Feature Value',title.vjust=1,barwidth = .5, barheight = 5,ticks = FALSE))+
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        legend.title = element_blank(),
        axis.text.y = element_text(size = 7, color = 'black',margin = margin(r = 0)),
        legend.text=element_text(size=6,color = 'black',margin = margin(r = 0)),
        axis.title.y = element_blank())

# Create directory for current date and save individual feature-level bar plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'indiv_event_heatmaps.svg'),indiv.event.timeshap.plot,device= svglite,units='in',dpi=600,width=2.17,height = 1)

### VI. Supplementary Figures 5 and 6
## Prepare dataframes
physician.difference.CIs <- read.csv('../model_performance/v6-0/test_set_discrimination_difference_CI.csv',
                                     na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC == 'Somers D') %>%
  mutate(SinceAdmission = WINDOW_IDX > 0) %>%
  drop_na(median)
physician.difference.CIs$WINDOW_IDX[!physician.difference.CIs$SinceAdmission] <- physician.difference.CIs$WINDOW_IDX[!physician.difference.CIs$SinceAdmission] + 1
physician.difference.CIs <- physician.difference.CIs %>%
  mutate(DaysAfterICUAdmission = WINDOW_IDX/12)

discrimination.CIs <- read.csv('../model_performance/v6-0/test_set_discrimination_CI.csv',
                               na.strings = c("NA","NaN","", " ")) %>%
  mutate(VERSION = 'All variables',
         SinceAdmission = WINDOW_IDX > 0) %>%
  filter(SinceAdmission,
         METRIC == 'Somers D')
discrimination.CIs$WINDOW_IDX[!discrimination.CIs$SinceAdmission] <- discrimination.CIs$WINDOW_IDX[!discrimination.CIs$SinceAdmission] + 1
discrimination.CIs <- discrimination.CIs %>%
  mutate(DaysAfterICUAdmission = WINDOW_IDX/12)

v7.discrimination.CIs <- read.csv('../model_performance/v7-0/test_set_discrimination_CI.csv',
                                  na.strings = c("NA","NaN","", " ")) %>%
  mutate(VERSION = 'Without physician impressions',
         SinceAdmission = WINDOW_IDX > 0) %>%
  filter(SinceAdmission,
         METRIC == 'Somers D',
         TUNE_IDX==171)
v7.discrimination.CIs$WINDOW_IDX[!v7.discrimination.CIs$SinceAdmission] <- v7.discrimination.CIs$WINDOW_IDX[!v7.discrimination.CIs$SinceAdmission] + 1
v7.discrimination.CIs <- v7.discrimination.CIs %>%
  mutate(DaysAfterICUAdmission = WINDOW_IDX/12)

discrimination.CIs <- rbind(discrimination.CIs,v7.discrimination.CIs)

## Create overall discrimination added with physician impression plots
# Since admission Somers' D plot with both models
since.adm.both.models.somers <- discrimination.CIs %>%
  ggplot() +
  coord_cartesian(xlim=c(0,7), ylim = c(35,55)) +
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(.0, .0)))+
  geom_line(aes(x=DaysAfterICUAdmission,y=100*median,color=VERSION),alpha = 1, size=1.3/.pt)+
  geom_ribbon(aes(x=DaysAfterICUAdmission,ymin=lo*100,ymax=hi*100,fill=VERSION),alpha=.2) +
  scale_fill_manual(values=c('#bc5090','#ffa600'))+
  scale_color_manual(values=c('#bc5090','#ffa600'))+
  ylab('Explanation of ordinal GOSE (%)')+
  xlab('Days since ICU admission')+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'bottom',
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_blank(),
    legend.text=element_text(size = 7, color = 'black',face = 'bold')
  )

# Since admission difference in Somers' D plot
since.adm.physician.diff.somers <- physician.difference.CIs %>%
  filter(SinceAdmission) %>%
  ggplot() +
  coord_cartesian(xlim=c(0,7), ylim = c(-5,10)) +
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(.0, .0)))+
  geom_hline(yintercept = 0, color='dark gray',alpha = 1, size=1.3/.pt)+
  geom_line(aes(x=DaysAfterICUAdmission,y=100*median),alpha = 1, size=1.3/.pt,color='#003f5c')+
  geom_ribbon(aes(x=DaysAfterICUAdmission,ymin=lo*100,ymax=hi*100),alpha=.2,fill='#003f5c') +
  ylab('Added explanation of ordinal GOSE (d%)')+
  xlab('Days since ICU admission')+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold')
  )

# Create directory for current date and save post-admission and pre-discharge Somers' D plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'since_adm_both_models_somers.svg'),since.adm.both.models.somers,device= svglite,units='in',dpi=600,width=3.7,height = 1.78)
ggsave(file.path('../plots',Sys.Date(),'since_adm_physician_diff_somers.svg'),since.adm.physician.diff.somers,device= svglite,units='in',dpi=600,width=3.7,height = 1.38)

### VII. Supplementary Figures 8 and 9
## Global feature-level TimeSHAP plots for missing variables
# Load missing token TimeSHAP values
missing.value.timeSHAP.df <- read.csv('../model_interpretations/v6-0/timeSHAP/filtered_plotting_missing_timeSHAP_values.csv',
                                      na.strings = c("NA","NaN","", " ")) %>%
  filter(Threshold == 'ExpectedValue') %>%
  mutate(Baseline = as.logical(Baseline),
         Numeric = as.logical(Numeric),
         GROUPS = case_when((RankIdx >= 11) ~ 'Top',
                            (RankIdx <= 10) ~'Bottom')) %>%
  mutate(GROUPS = factor(GROUPS,levels=c('Top','Middle','Bottom')))

# Complete formatting dataframe prior to plotting
missing.value.timeSHAP.df <- missing.value.timeSHAP.df %>%
  mutate(Baseline = recode(as.character(Baseline),'TRUE'='Static','FALSE'='Dynamic')) %>%
  mutate(Baseline = fct_relevel(Baseline, 'Static', 'Dynamic')) %>%
  mutate(BaseToken = fct_reorder(BaseToken, RankIdx))

# Create feature importance beeswarm plot for missing static predictors
missing.static.timeshap.plot <- missing.value.timeSHAP.df %>%
  filter(Baseline=='Static') %>%
  ggplot() +
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_quasirandom(aes(y=BaseToken,x=SHAP),groupOnX=FALSE,varwidth = FALSE,alpha = 0.6,stroke = 0,size=.75,color='#003f5c') + 
  theme_minimal(base_family = 'Roboto Condensed') +
  facet_grid(rows = vars(GROUPS), scales = 'free_y', switch = 'y', space = 'free_y') +
  theme(
    strip.background = element_blank(),
    strip.text = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(size = 5, color = 'black'),
    axis.text.y = element_text(size = 6, color = 'black'),
    axis.title.x = element_blank(),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'none',
    panel.grid.major.y = element_blank(),
    panel.spacing = unit(10, 'points'),
    plot.margin=grid::unit(c(0,2,0,0), "mm")
  )

# Create feature importance beeswarm plot for missing dynamic predictors
missing.dynamic.timeshap.plot <- missing.value.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         abs(SHAP) <= .1) %>%
  ggplot() +
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_quasirandom(aes(y=BaseToken,x=SHAP),groupOnX=FALSE,varwidth = FALSE,alpha = 0.6,stroke = 0,size=.75,color='#bc5090') + 
  theme_minimal(base_family = 'Roboto Condensed') +
  facet_grid(rows = vars(GROUPS), scales = 'free_y', switch = 'y', space = 'free_y') +
  theme(
    strip.background = element_blank(),
    strip.text = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(size = 5, color = 'black'),
    axis.text.y = element_text(size = 6, color = 'black'),
    axis.title.x = element_blank(),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'none',
    panel.grid.major.y = element_blank(),
    panel.spacing = unit(10, 'points'),
    plot.margin=grid::unit(c(0,2,0,0), "mm")
  )

# Create directory for current date and save missing feature-level TimeSHAP plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'missing_static_timeshap.png'),missing.static.timeshap.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'missing_dynamic_timeshap.png'),missing.dynamic.timeshap.plot,units='in',dpi=600,height=3.38,width=3.75)

## Global feature-level TimeSHAP plots for each category of variables
# Load category-based token TimeSHAP values
types.timeSHAP.df <- read.csv('../model_interpretations/v6-0/timeSHAP/filtered_plotting_types_timeSHAP_values.csv',
                              na.strings = c("NA","NaN","", " ")) %>%
  filter(Threshold == 'ExpectedValue') %>%
  mutate(Baseline = as.logical(Baseline),
         Numeric = as.logical(Numeric),
         Ordered = as.logical(Ordered),
         Binary = as.logical(Binary),
         GROUPS = case_when((RankIdx >= 11) ~ 'Top',
                            (RankIdx <= 10) ~'Bottom')) %>%
  mutate(GROUPS = factor(GROUPS,levels=c('Top','Middle','Bottom')))

# Isolate and save unique `Tokens` by type to manually verify and fill variable order (if applicable)
exp.GOSE.types.token.df <- types.timeSHAP.df %>%
  select(Type,RankIdx,BaseToken,Token,Baseline,Numeric,Missing,Ordered,Binary,TokenRankIdx) %>%
  unique() %>%
  mutate(OrderIdx = case_when((!Ordered&!Binary)|Numeric ~ (TokenRankIdx-1)))
missing.exp.GOSE.types.token.df <- exp.GOSE.types.token.df %>%
  filter(is.na(OrderIdx)) %>%
  select(BaseToken,Token,OrderIdx)
prev.filled.orders <- read_xlsx('../model_interpretations/v6-0/timeSHAP/expected_GOSE_timeSHAP_orders_filled.xlsx') %>%
  select(BaseToken,Token,OrderIdx) %>%
  filter(BaseToken %in% missing.exp.GOSE.types.token.df$BaseToken) %>%
  rename(PrevOrderIdx = OrderIdx)
exp.GOSE.types.token.df <- exp.GOSE.types.token.df %>%
  left_join(prev.filled.orders)
exp.GOSE.types.token.df$OrderIdx[(is.na(exp.GOSE.types.token.df$OrderIdx))&(!is.na(exp.GOSE.types.token.df$PrevOrderIdx))] <- exp.GOSE.types.token.df$PrevOrderIdx[(is.na(exp.GOSE.types.token.df$OrderIdx))&(!is.na(exp.GOSE.types.token.df$PrevOrderIdx))]
exp.GOSE.types.token.df <- exp.GOSE.types.token.df %>% select(-PrevOrderIdx)
write.xlsx(exp.GOSE.types.token.df,'../model_interpretations/v6-0/timeSHAP/types_expected_GOSE_timeSHAP_orders.xlsx') 

# Load manually inspected orders of unique `Tokens` in plotting datafame
exp.GOSE.types.token.df <- read_xlsx('../model_interpretations/v6-0/timeSHAP/types_expected_GOSE_timeSHAP_orders_filled.xlsx') 

# Calculate the number of unique known values per predictor
max.types.order.indices <- exp.GOSE.types.token.df %>%
  group_by(BaseToken) %>%
  summarise(MaxOrderIdx = max(OrderIdx))

# Merge manually created labels to filtered TimeSHAP plotting dataframe and complete formatting
types.timeSHAP.df <- types.timeSHAP.df %>%
  left_join(exp.GOSE.types.token.df) %>%
  left_join(max.types.order.indices) %>%
  arrange(TUNE_IDX,Type,Threshold,RankIdx,OrderIdx) %>%
  mutate(ColorScale = OrderIdx/MaxOrderIdx) %>%
  mutate(ColorScale = case_when(is.na(ColorScale)~1,
                                ((!is.na(ColorScale))&(ColorScale>=0))~ColorScale)) %>%
  mutate(Baseline = recode(as.character(Baseline),'TRUE'='Static','FALSE'='Dynamic')) %>%
  mutate(Baseline = fct_relevel(Baseline, 'Static', 'Dynamic')) %>%
  mutate(BaseToken = fct_reorder(BaseToken, RankIdx))

# Create feature importance beeswarm plot for static predictors of each type
static.Imaging.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Static',
         Type == 'Brain Imaging') %>%
  types.timeSHAP.plots()

static.DemoSES.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Static',
         Type == 'Demographics and Socioeconomic Status',
         abs(SHAP)<=.35) %>%
  types.timeSHAP.plots()

static.ERCare.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Static',
         Type == 'Emergency Care and ICU Admission',
         abs(SHAP)<=.5) %>%
  types.timeSHAP.plots()

static.ICUMedsMx.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Static',
         Type == 'ICU Medications and Management') %>%
  types.timeSHAP.plots()

static.Injury.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Static',
         Type == 'Injury Characteristics and Severity',
         abs(SHAP)<=.1) %>%
  types.timeSHAP.plots()

static.Labs.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Static',
         Type == 'Labs',
         abs(SHAP)<=.2) %>%
  types.timeSHAP.plots()

static.MedBehavHx.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Static',
         Type == 'Medical and Behavioural History',
         abs(SHAP)<=.1) %>%
  types.timeSHAP.plots()

static.SurgMonitor.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Static',
         Type == 'Surgery and Neuromonitoring') %>%
  types.timeSHAP.plots()

# Create feature importance beeswarm plot for dynamic predictors of each type
dynamic.Imaging.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         Type == 'Brain Imaging',
         abs(SHAP)<=.025) %>%
  types.timeSHAP.plots()

dynamic.ICUMedsMx.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         Type == 'ICU Medications and Management',
         abs(SHAP)<=.15) %>%
  types.timeSHAP.plots()

dynamic.ICUVitals.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         Type == 'ICU Vitals and Assessments',
         abs(SHAP)<=.2) %>%
  types.timeSHAP.plots()

dynamic.Labs.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         Type == 'Labs',
         abs(SHAP)<=.025) %>%
  types.timeSHAP.plots()

dynamic.SurgMonitor.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         Type == 'Surgery and Neuromonitoring',
         abs(SHAP)<=.2) %>%
  types.timeSHAP.plots()

# Create directory for current date and save missing feature-level TimeSHAP plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'imaging_static_timeshap.png'),static.Imaging.timeshap.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'demoSES_static_timeshap.png'),static.DemoSES.timeshap.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'ERcare_static_timeshap.png'),static.ERCare.timeshap.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'ICUmedsmx_static_timeshap.png'),static.ICUMedsMx.timeshap.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'injury_static_timeshap.png'),static.Injury.timeshap.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'labs_static_timeshap.png'),static.Labs.timeshap.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'medbehavhx_static_timeshap.png'),static.MedBehavHx.timeshap.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'surgmonitor_static_timeshap.png'),static.SurgMonitor.timeshap.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'imaging_dynamic_timeshap.png'),dynamic.Imaging.timeshap.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'ICUmedsmx_dynamic_timeshap.png'),dynamic.ICUMedsMx.timeshap.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'ICUvitals_dynamic_timeshap.png'),dynamic.ICUVitals.timeshap.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'labs_dynamic_timeshap.png'),dynamic.Labs.timeshap.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'surgmonitor_dynamic_timeshap.png'),dynamic.SurgMonitor.timeshap.plot,units='in',dpi=600,height=3.38,width=3.75)

### VIII. Supplementary Figure 7
## Prepare dataframe of filtered TimeSHAP values for plotting
# Load TimeSHAP value dataframe
filt.timeSHAP.df <- read.csv('../model_interpretations/v6-0/timeSHAP/filtered_plotting_timeSHAP_values.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  filter(Threshold != 'ExpectedValue') %>%
  mutate(Baseline = as.logical(Baseline),
         Numeric = as.logical(Numeric),
         Binary = as.logical(Binary),
         Ordered = as.logical(Ordered),
         GROUPS = case_when((RankIdx >= 11) ~ 'Top',
                            (RankIdx <= 10) ~'Bottom')) %>%
  mutate(GROUPS = factor(GROUPS,levels=c('Top','Middle','Bottom')))

# Isolate and save unique `Tokens` to manually verify and fill variable order (if applicable)
thresh.GOSE.token.df <- filt.timeSHAP.df %>%
  select(BaseToken,Token,Baseline,Numeric,Missing,TokenRankIdx,Ordered,Binary) %>%
  unique() %>%
  mutate(OrderIdx = case_when((!Ordered&!Binary)|Numeric ~ (TokenRankIdx-1)))
missing.thresh.GOSE.token.df <- thresh.GOSE.token.df %>%
  filter(is.na(OrderIdx)) %>%
  select(BaseToken,Token,OrderIdx)
prev.filled.orders <- read_xlsx('../model_interpretations/v6-0/timeSHAP/types_expected_GOSE_timeSHAP_orders_filled.xlsx') %>%
  select(BaseToken,Token,OrderIdx) %>%
  filter(BaseToken %in% missing.thresh.GOSE.token.df$BaseToken) %>%
  rename(PrevOrderIdx = OrderIdx)
thresh.GOSE.token.df <- thresh.GOSE.token.df %>%
  left_join(prev.filled.orders)
thresh.GOSE.token.df$OrderIdx[(is.na(thresh.GOSE.token.df$OrderIdx))&(!is.na(thresh.GOSE.token.df$PrevOrderIdx))] <- thresh.GOSE.token.df$PrevOrderIdx[(is.na(thresh.GOSE.token.df$OrderIdx))&(!is.na(thresh.GOSE.token.df$PrevOrderIdx))]
thresh.GOSE.token.df <- thresh.GOSE.token.df %>% select(-PrevOrderIdx)
write.xlsx(thresh.GOSE.token.df,'../model_interpretations/v6-0/timeSHAP/threshold_GOSE_timeSHAP_orders.xlsx') 

# Load manually inspected orders of unique `Tokens` in plotting datafame
thresh.GOSE.token.df <- read_xlsx('../model_interpretations/v6-0/timeSHAP/threshold_GOSE_timeSHAP_orders_filled.xlsx') 

# Calculate the number of unique known values per predictor
max.order.indices <- thresh.GOSE.token.df %>%
  group_by(BaseToken) %>%
  summarise(MaxOrderIdx = max(OrderIdx))

# Merge manually created labels to filtered TimeSHAP plotting dataframe and complete formatting
filt.timeSHAP.df <- filt.timeSHAP.df %>%
  left_join(thresh.GOSE.token.df) %>%
  left_join(max.order.indices) %>%
  arrange(TUNE_IDX,Threshold,RankIdx,OrderIdx) %>%
  mutate(ColorScale = OrderIdx/MaxOrderIdx) %>%
  mutate(ColorScale = case_when(is.na(ColorScale)~1,
                                ((!is.na(ColorScale))&(ColorScale>=0))~ColorScale)) %>%
  mutate(Baseline = recode(as.character(Baseline),'TRUE'='Static','FALSE'='Dynamic')) %>%
  mutate(Baseline = fct_relevel(Baseline, 'Static', 'Dynamic')) %>%
  mutate(BaseToken = fct_reorder(BaseToken, RankIdx))

# Create feature importance beeswarm plot for static predictors of each threshold
static.gose.gt.1.plot <- filt.timeSHAP.df %>%
  filter(Baseline=='Static',
         Threshold == 'GOSE>1') %>%
  types.timeSHAP.plots()

static.gose.gt.3.plot <- filt.timeSHAP.df %>%
  filter(Baseline=='Static',
         Threshold == 'GOSE>3') %>%
  types.timeSHAP.plots()

static.gose.gt.4.plot <- filt.timeSHAP.df %>%
  filter(Baseline=='Static',
         Threshold == 'GOSE>4') %>%
  types.timeSHAP.plots()

static.gose.gt.5.plot <- filt.timeSHAP.df %>%
  filter(Baseline=='Static',
         Threshold == 'GOSE>5') %>%
  types.timeSHAP.plots()

static.gose.gt.6.plot <- filt.timeSHAP.df %>%
  filter(Baseline=='Static',
         Threshold == 'GOSE>6') %>%
  types.timeSHAP.plots()

static.gose.gt.7.plot <- filt.timeSHAP.df %>%
  filter(Baseline=='Static',
         Threshold == 'GOSE>7') %>%
  types.timeSHAP.plots()

# Create feature importance beeswarm plot for dynamic predictors of each threshold
dynamic.gose.gt.1.plot <- filt.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         Threshold == 'GOSE>1',
         abs(SHAP)<.075) %>%
  types.timeSHAP.plots()

dynamic.gose.gt.3.plot <- filt.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         Threshold == 'GOSE>3',
         abs(SHAP)<.01) %>%
  types.timeSHAP.plots()

dynamic.gose.gt.4.plot <- filt.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         Threshold == 'GOSE>4',
         abs(SHAP)<.01) %>%
  types.timeSHAP.plots()

dynamic.gose.gt.5.plot <- filt.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         Threshold == 'GOSE>5') %>%
  types.timeSHAP.plots()

dynamic.gose.gt.6.plot <- filt.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         Threshold == 'GOSE>6',
         abs(SHAP)<.005) %>%
  types.timeSHAP.plots()

dynamic.gose.gt.7.plot <- filt.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         Threshold == 'GOSE>7') %>%
  types.timeSHAP.plots()

# Create directory for current date and save feature-level TimeSHAP plots at each threshold of GOSE
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'GOSE_gt_1_static_timeshap.png'),static.gose.gt.1.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'GOSE_gt_3_static_timeshap.png'),static.gose.gt.3.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'GOSE_gt_4_static_timeshap.png'),static.gose.gt.4.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'GOSE_gt_5_static_timeshap.png'),static.gose.gt.5.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'GOSE_gt_6_static_timeshap.png'),static.gose.gt.6.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'GOSE_gt_7_static_timeshap.png'),static.gose.gt.7.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'GOSE_gt_1_dynamic_timeshap.png'),dynamic.gose.gt.1.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'GOSE_gt_3_dynamic_timeshap.png'),dynamic.gose.gt.3.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'GOSE_gt_4_dynamic_timeshap.png'),dynamic.gose.gt.4.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'GOSE_gt_5_dynamic_timeshap.png'),dynamic.gose.gt.5.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'GOSE_gt_6_dynamic_timeshap.png'),dynamic.gose.gt.6.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'GOSE_gt_7_dynamic_timeshap.png'),dynamic.gose.gt.7.plot,units='in',dpi=600,height=3.38,width=3.75)

### IX. Supplementary Appendix
## Load and prepare all version tuning grids
# Version 0-0
v0.0.tuning.grid <- read.csv('../model_outputs/v0-0/tuning_grid.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  rename(TUNE_IDX = tune_idx) %>%
  mutate(WINDOW_LIMIT = 'None',
         TIME_TOKENS = 'None',
         WINDOW_DURATION = 2,
         STRATEGY = 'abs') %>%
  unique() %>%
  mutate(VERSION='v0-0')

# Version 0-1
v0.1.tuning.grid <- read.csv('../model_outputs/v0-1/tuning_grid.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  rename(TUNE_IDX = tune_idx) %>%
  mutate(WINDOW_LIMIT = 'None',
         TIME_TOKENS = 'None',
         WINDOW_DURATION = 2,
         STRATEGY = 'abs') %>%
  select(-Fold) %>%
  unique() %>%
  mutate(VERSION='v0-1')

# Version 0-2
v0.2.tuning.grid <- read.csv('../model_outputs/v0-2/tuning_grid.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  rename(TUNE_IDX = tune_idx) %>%
  mutate(WINDOW_LIMIT = 'None',
         TIME_TOKENS = 'None',
         WINDOW_DURATION = 2,
         STRATEGY = 'abs') %>%
  select(-Fold) %>%
  unique() %>%
  mutate(VERSION='v0-2')

# Version 0-3
v0.3.tuning.grid <- read.csv('../model_outputs/v0-3/tuning_grid.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  rename(TUNE_IDX = tune_idx) %>%
  mutate(WINDOW_LIMIT = 'None',
         TIME_TOKENS = 'None',
         WINDOW_DURATION = 2,
         STRATEGY = 'abs') %>%
  unique() %>%
  mutate(VERSION='v0-3')

# Version 1-0
v1.0.tuning.grid <- read.csv('../model_outputs/v1-0/tuning_grid.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  rename(TUNE_IDX = tune_idx) %>%
  mutate(WINDOW_LIMIT = 'None',
         TIME_TOKENS = 'None',
         WINDOW_DURATION = 2,
         STRATEGY = 'abs') %>%
  unique() %>%
  mutate(VERSION='v1-0')

# Version 2-0
v2.0.tuning.grid <- read.csv('../model_outputs/v2-0/tuning_grid.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  rename(TUNE_IDX = tune_idx) %>%
  mutate(WINDOW_LIMIT = 'None',
         TIME_TOKENS = 'None',
         TOKEN_CUTS = 20,
         STRATEGY = 'abs') %>%
  select(-c(repeat.,fold,adm_or_disch)) %>%
  unique() %>%
  mutate(VERSION='v2-0')

# Version 3-0
v3.0.tuning.grid <- read.csv('../model_outputs/v3-0/tuning_grid.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  rename(TUNE_IDX = tune_idx) %>%
  mutate(WINDOW_LIMIT = 'None',
         TIME_TOKENS = 'None',
         TOKEN_CUTS = 20,
         WINDOW_DURATION = 2,
         STRATEGY = 'abs') %>%
  select(-c(repeat.,fold,adm_or_disch)) %>%
  unique() %>%
  mutate(VERSION='v3-0')

# Version 4-0
v4.0.tuning.grid <- read.csv('../model_outputs/v4-0/tuning_grid.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  rename(TUNE_IDX = tune_idx) %>%
  mutate(TIME_TOKENS = 'None',
         TOKEN_CUTS = 20,
         WINDOW_DURATION = 2,
         STRATEGY = 'abs') %>%
  select(-c(repeat.,fold)) %>%
  unique() %>%
  mutate(VERSION='v4-0')

# Version 5-0
v5.0.tuning.grid <- read.csv('../model_outputs/v5-0/tuning_grid.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  rename(TUNE_IDX = tune_idx) %>%
  mutate(TIME_TOKENS = 'None',
         TOKEN_CUTS = 20,
         WINDOW_DURATION = 2) %>%
  select(-c(repeat.,fold)) %>%
  unique() %>%
  mutate(VERSION='v5-0')

# Version 6-0
v6.0.tuning.grid <- read.csv('../model_outputs/v6-0/tuning_grid.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  mutate(WINDOW_DURATION = 2) %>%
  select(-c(repeat.,fold)) %>%
  unique() %>%
  mutate(VERSION='v6-0')

# Version 7-0
v7.0.tuning.grid <- read.csv('../model_outputs/v7-0/tuning_grid.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  mutate(WINDOW_DURATION = 2,
         STRATEGY = 'abs') %>%
  select(-c(FOLD)) %>%
  unique() %>%
  mutate(VERSION='v7-0')

# Concatenate tuning grids from all iterations
compiled.tuning.grids <- rbind(v0.0.tuning.grid,
                               v0.1.tuning.grid,
                               v0.2.tuning.grid,
                               v0.3.tuning.grid,
                               v1.0.tuning.grid,
                               v2.0.tuning.grid,
                               v3.0.tuning.grid,
                               v4.0.tuning.grid,
                               v5.0.tuning.grid,
                               v6.0.tuning.grid,
                               v7.0.tuning.grid) %>%
  relocate(VERSION)

### X. Unused
## Load and prepare summarised test set predictions
# Load summarised testing set predictions
summarised.test.preds.df <- read.csv('../model_outputs/v6-0/summarised_test_predictions.csv',
                                     na.strings = c("NA","NaN","", " ")) %>%
  filter(WindowIdx <= 84) %>%
  mutate(SinceAdmission = WindowIdx > 0)
summarised.test.preds.df$WindowIdx[!summarised.test.preds.df$SinceAdmission] <- summarised.test.preds.df$WindowIdx[!summarised.test.preds.df$SinceAdmission] + 1
summarised.test.preds.df <- summarised.test.preds.df %>%
  mutate(DaysAfterICUAdmission = WindowIdx/12)

summarised.from.disch.test.preds.df <- read.csv('../model_outputs/v6-0/summarised_from_discharge_test_predictions.csv',
                                                na.strings = c("NA","NaN","", " ")) %>%
  filter(WindowIdx >= -85) %>%
  mutate(SinceAdmission = WindowIdx > 0)
summarised.from.disch.test.preds.df$WindowIdx[!summarised.from.disch.test.preds.df$SinceAdmission] <- summarised.from.disch.test.preds.df$WindowIdx[!summarised.from.disch.test.preds.df$SinceAdmission] + 1
summarised.from.disch.test.preds.df <- summarised.from.disch.test.preds.df %>%
  mutate(DaysBeforeICUDischarge = abs(WindowIdx/12))

# Calculate proportion of outcome at each threshold
study.GUPI.GOSE <- read.csv('../legacy_cross_validation_splits.csv',
                            na.strings = c("NA","NaN","", " ")) %>%
  select(GUPI,GOSE) %>%
  unique()
threshold.frequencies <- as.vector(1-cumsum(table(study.GUPI.GOSE$GOSE)/nrow(study.GUPI.GOSE))[1:6])*100

### XI. Addendum to figure 2
## Calculate true expected GOSE over time
# Create vector mapping of GOSE to index
gose.index <- seq(0,6)

# Load study set GUPIs
study.GUPI.GOSE <- read.csv('../legacy_cross_validation_splits.csv',
                            na.strings = c("NA","NaN","", " ")) %>%
  select(GUPI,GOSE) %>%
  unique()

# Convert explicit GOSE labels to indices
study.GUPI.GOSE <- study.GUPI.GOSE %>%
  mutate(GOSE.index = as.integer(plyr::mapvalues(GOSE,from=sort(unique(study.GUPI.GOSE$GOSE)),to=gose.index)))

# Load token type counts and merge GOSE index information
true.exp.GOSE.df <-  read.csv('../tokens/fold1/token_type_counts.csv',
                              na.strings = c("NA","NaN","", " ")) %>%
  left_join(study.GUPI.GOSE) %>%
  group_by(WindowIdx) %>%
  summarise(true.exp.GOSE = mean(GOSE.index,na.rm=T),
            sd.exp.GOSE = sd(GOSE.index,na.rm = T),
            q1.exp.GOSE = quantile(GOSE.index,.25,na.rm=T),
            q3.exp.GOSE = quantile(GOSE.index,.75,na.rm=T)) %>%
  mutate(DaysAfterICUAdmission = WindowIdx/12) %>%
  filter(DaysAfterICUAdmission<=7)

## Prepare average patient output in expected GOSE
# Load and prepare dataframe of summarised average event predictions
summ.ave.event.preds <- read.csv('../model_interpretations/v6-0/timeSHAP/summarised_average_event_predictions.csv',
                                 na.strings = c("NA","NaN","", " ")) %>%
  filter(Threshold=='ExpectedValue') %>%
  mutate(DaysAfterICUAdmission = WindowIdx/12)

## Create plot of average event expected GOSE over time
# Average-event expected GOSE plot
ave.event.output.plot <- ggplot() +
  coord_cartesian(xlim=c(0,7),ylim=c(1.5,4.5)) +
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(.0, .0)))+
  scale_y_continuous(breaks=seq(0,6,by=1),labels = c('1','2 or 3','4','5','6','7','8'))+
  geom_line(data=true.exp.GOSE.df,aes(x=DaysAfterICUAdmission,y=true.exp.GOSE),alpha = 1, size=1.3/.pt,color='dark gray',linetype = "dashed")+
  geom_line(data=summ.ave.event.preds,aes(x=DaysAfterICUAdmission,y=median),alpha = 1, size=1.3/.pt,color='#003f5c')+
  geom_ribbon(data=summ.ave.event.preds,aes(x=DaysAfterICUAdmission,ymin=Q1,ymax=Q3),alpha=.2,fill='#003f5c') +
  ylab('GOSE_E[i]')+
  xlab('Days since ICU admission')+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold')
  )

# Create directory for current date and save average-event expected GOSE plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'ave_event_output.svg'),ave.event.output.plot,device= svglite,units='in',dpi=600,width=3.7,height = 1.38)

### XII. Supplementary Figures 2 and 3
## Load token characteristics
# Token characteristics
token.type.counts <- read.csv('../tokens/fold1/token_type_counts.csv',
                              na.strings = c("NA","NaN","", " ")) %>%
  mutate(DynamicTokens = TotalTokens - Baseline) %>%
  group_by(WindowIdx) %>%
  summarise(q1DynamicTokens = quantile(DynamicTokens,.25,na.rm=T),
            medianDynamicTokens = median(DynamicTokens,na.rm=T),
            q3DynamicTokens = quantile(DynamicTokens,.75,na.rm=T),
            q1BaselineTokens = quantile(Baseline,.25,na.rm=T),
            medianBaselineTokens = median(Baseline,na.rm=T),
            q3BaselineTokens = quantile(Baseline,.75,na.rm=T),
            countPatients = n()) %>%
  mutate(DaysAfterICUAdmission = WindowIdx/12,
         PropRemaining = 100*countPatients/1552,
         SumTotalTokens = medianBaselineTokens+medianDynamicTokens)

# Create ggplot of median token count over the first month
median.tokens.per.pt <- token.type.counts %>%
  filter(DaysAfterICUAdmission<=30)%>%
  ggplot(aes(x=DaysAfterICUAdmission)) +
  geom_ribbon(aes(ymin=0,ymax=medianBaselineTokens,fill='Static'),alpha=.2) +
  geom_ribbon(aes(ymin=medianBaselineTokens,ymax=SumTotalTokens,fill='Dynamic'),alpha=.2) +
  geom_line(aes(y=medianBaselineTokens),alpha = 1, size=1.3/.pt,color='#003f5c')+
  geom_line(aes(y=SumTotalTokens),alpha = 1, size=1.3/.pt,color='#bc5090')+
  scale_x_continuous(breaks=seq(0,30,by=1),expand = expansion(mult = c(.00, .01)))+
  scale_y_continuous(expand = expansion(mult = c(.00, .00)))+
  scale_fill_manual(name = 'Variable type',values=c("Static" = "#003f5c", "Dynamic" = "#bc5090"))+
  coord_cartesian(xlim=c(0,30)) +
  ylab('Token count (median/patient)')+
  xlab('Days since ICU admission')+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    strip.text = element_blank(),
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    legend.key.size = unit(1.3/.pt,"line")
  )

# Create directory for current date and save token characteristic plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'month_token_characteristics.svg'),median.tokens.per.pt,device= svglite,units='in',dpi=600,width=3.7,height = 1.75)

# Create ggplot of proportion of remaining population over the first week
prop.remaining <- token.type.counts %>%
  filter(DaysAfterICUAdmission<=30)%>%
  ggplot(aes(x=DaysAfterICUAdmission)) +
  geom_ribbon(aes(ymin=0,ymax=PropRemaining,fill='All patients'),alpha=.2) +
  geom_line(aes(y=PropRemaining),alpha = 1, size=1.3/.pt,color='#ffa600')+
  scale_x_continuous(breaks=seq(0,30,by=1),expand = expansion(mult = c(.00, .01)))+
  scale_y_continuous(expand = expansion(mult = c(.00, .00)))+
  coord_cartesian(xlim=c(0,30),ylim = c(0,100)) +
  ylab('Proportion remaining (%)')+
  xlab('Days since ICU admission')+
  scale_fill_manual(name = 'Variable type',values=c("All patients" = "#ffa600"))+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    strip.text = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    legend.key.size = unit(1.3/.pt,"line")
  )

# Create directory for current date and save remaining proportion plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'month_prop_remaining.png'),prop.remaining,units='in',dpi=600,width=3.7,height = 1.75)

### XIII. Supplementary Figure 4
## Load performance metrics from v2-0
# Load overall metrics from v2-0
v2.overall.metrics <- read.csv('../model_performance/v2-0/CI_overall_metrics.csv',
                               na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='Somers D',
         ADM_OR_DISCH=='adm')

# Load threshold-level metrics from v2-0
v2.threshold.metrics <- read.csv('../model_performance/v2-0/CI_threshold_metrics.csv',
                                 na.strings = c("NA","NaN","", " ")) %>%
  filter(THRESHOLD=='Average',
         METRIC=='Calib_Slope',
         ADM_OR_DISCH=='adm')

# Load tuning grid from v2-0
v2.tuning.grid <- read.csv('../model_outputs/v2-0/tuning_grid.csv',
                           na.strings = c("NA","NaN","", " ")) %>%
  select(-c(repeat.,fold)) %>%
  filter(adm_or_disch=='adm') %>%
  unique() %>%
  rename(TUNE_IDX=tune_idx)

# Add window duration information to performance dataframes
v2.overall.metrics <- v2.overall.metrics %>%
  left_join(v2.tuning.grid %>% select(TUNE_IDX,WINDOW_DURATION))
v2.threshold.metrics <- v2.threshold.metrics %>%
  left_join(v2.tuning.grid %>% select(TUNE_IDX,WINDOW_DURATION))

## Load ultimate metrics from v6-0
# Load discrimination metrics from v6-0
v6.overall.metrics <- read.csv('../model_performance/v6-0/test_set_discrimination_CI.csv',
                               na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='Somers D') %>%
  mutate(WINDOW_DURATION=2)

# Load calibration metrics from v6-0
v6.threshold.metrics <- read.csv('../model_performance/v6-0/test_set_calibration_CI.csv',
                                 na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='CALIB_SLOPE',
         THRESHOLD=='Average') %>%
  mutate(WINDOW_DURATION=2)

## Combine metric dataframes
# Combine Somers' D dataframes
combined.somers.dataframes <- v2.overall.metrics %>%
  filter(WINDOW_DURATION!=2) %>%
  select(-ADM_OR_DISCH) %>%
  rbind(v6.overall.metrics) %>%
  filter(WINDOW_IDX>=1) %>%
  mutate(DaysAfterICUAdmission = WINDOW_IDX/(24/WINDOW_DURATION))

# Combine calibration slope dataframes
combined.calib.slope.dataframes <- v2.threshold.metrics %>%
  filter(WINDOW_DURATION!=2) %>%
  select(-ADM_OR_DISCH) %>%
  rbind(v6.threshold.metrics) %>%
  filter(WINDOW_IDX>=1) %>%
  mutate(DaysAfterICUAdmission = WINDOW_IDX/(24/WINDOW_DURATION))

## Load baseline performance dataframes
# Baseline discrimination results
baseline.discrimination.CIs <- read.csv('../model_performance/BaselineComparison/test_set_discrimination_CI.csv',
                                        na.strings = c("NA","NaN","", " ")) %>%
  mutate(WINDOW_DURATION = 'None',
         DaysAfterICUAdmission = WINDOW_IDX/12) %>%
  filter(METRIC=='Somers D',
         WINDOW_IDX>=1)

# Baseline calibration metric results
baseline.calibration.CIs <- read.csv('../model_performance/BaselineComparison/test_set_calibration_CI.csv',
                                     na.strings = c("NA","NaN","", " ")) %>%
  mutate(WINDOW_DURATION = 'None',
         DaysAfterICUAdmission = WINDOW_IDX/12) %>%
  filter(METRIC=='CALIB_SLOPE',
         WINDOW_IDX>=1,
         THRESHOLD=='Average')

## Create overall discrimination performance plots across time window lengths
# Since admission Somers' D plot
since.adm.somers <- ggplot() +
  coord_cartesian(xlim=c(0,7), ylim = c(27.5,55)) +
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(.0, .0)))+
  geom_line(data=baseline.discrimination.CIs,aes(x=DaysAfterICUAdmission,y=100*median),alpha = 1, size=1.3/.pt,color='dark gray')+
  geom_line(data=baseline.discrimination.CIs,aes(x=DaysAfterICUAdmission,y=100*lo),alpha = 1, size=1.3/.pt,color='dark gray',linetype = "dashed")+
  geom_line(data=baseline.discrimination.CIs,aes(x=DaysAfterICUAdmission,y=100*hi),alpha = 1, size=1.3/.pt,color='dark gray',linetype = "dashed")+
  geom_line(data=combined.somers.dataframes,aes(x=DaysAfterICUAdmission,y=100*median,color=factor(WINDOW_DURATION)),alpha = 1, size=1.3/.pt)+
  geom_ribbon(data=combined.somers.dataframes,aes(x=DaysAfterICUAdmission,ymin=lo*100,ymax=hi*100,fill=factor(WINDOW_DURATION)),alpha=.2) +
  ylab('Explanation of ordinal GOSE (%)')+
  xlab('Days since ICU admission')+
  theme_minimal(base_family = 'Roboto Condensed') +
  scale_fill_manual(values=c('#003f5c','#7a5195','#ef5675','#ffa600'))+
  scale_color_manual(values=c('#003f5c','#7a5195','#ef5675','#ffa600'))+
  guides(fill=guide_legend(title="Time window length (hrs)"),
         color=guide_legend(title="Time window length (hrs)"))+
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    legend.key.size = unit(1.3/.pt,"line")
  )

# Since admission calibration slope plot
since.adm.calib.slope <- ggplot() +
  coord_cartesian(xlim=c(0,7),ylim=c(0,1.5)) +
  geom_vline(xintercept = 0.33333333, color='dark gray',alpha = 1, size=1.3/.pt, linetype = "dashed")+
  geom_hline(yintercept = 1, color='#ffa600',alpha = 1, size=2/.pt)+
  geom_line(data=baseline.calibration.CIs,aes(x=DaysAfterICUAdmission,y=median),alpha = 1, size=1.3/.pt,color='dark gray')+
  geom_line(data=baseline.calibration.CIs,aes(x=DaysAfterICUAdmission,y=lo),alpha = 1, size=1.3/.pt,color='dark gray',linetype = "dashed")+
  geom_line(data=baseline.calibration.CIs,aes(x=DaysAfterICUAdmission,y=hi),alpha = 1, size=1.3/.pt,color='dark gray',linetype = "dashed")+
  geom_line(data=combined.calib.slope.dataframes,aes(x=DaysAfterICUAdmission,y=median,color=factor(WINDOW_DURATION)),alpha = 1, size=1.3/.pt) + 
  geom_ribbon(data=combined.calib.slope.dataframes,aes(x=DaysAfterICUAdmission,ymin=lo,ymax=hi,fill=factor(WINDOW_DURATION)),alpha=.2,size=.75/.pt) + 
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(0,0))) +
  ylab('Calibration slope')+
  xlab('Days since ICU admission') + 
  theme_minimal(base_family = 'Roboto Condensed') +
  scale_fill_manual(values=c('#003f5c','#7a5195','#ef5675','#ffa600'))+
  scale_color_manual(values=c('#003f5c','#7a5195','#ef5675','#ffa600'))+
  guides(fill=guide_legend(title="Time window length (hrs)"),
         color=guide_legend(title="Time window length (hrs)"))+
  theme(
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    legend.key.size = unit(1.3/.pt,"line")
  )

# Create directory for current date and save cross-window-length performance curves
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'since_adm_all_models_somers.svg'),since.adm.somers,device= svglite,units='in',dpi=600,width=3.7,height = 1.78)
ggsave(file.path('../plots',Sys.Date(),'since_adm_all_models_calib_slope.svg'),since.adm.calib.slope,device= svglite,units='in',dpi=600,width=3.7,height = 1.78)

### XIV. Appendix variable lists
## Load full token dictionaries to extract unique variable list
# Load v6-0 token dictionary
v6.token.dictionary <- read_xlsx('../tokens/legacy_full_token_keys.xlsx') %>%
  select(BaseToken,Baseline,ICUIntervention,ClinicianInput,Type) %>%
  unique()

# Load v7-0 token dictionary
v7.token.dictionary <- read_xlsx('../tokens/full_token_keys.xlsx') %>%
  select(BaseToken,Baseline,ICUIntervention,ClinicianInput,Type) %>%
  unique()

# Load full CENTER-TBI dictionary
CENTER.TBI.dictionary <- read_xlsx('../CENTER-TBI/CENTER-TBI_version3-dictionary.xlsx') %>%
  rename(BaseToken = name,
         VariableFormat = valueType,
         Description = description) %>%
  select(BaseToken,VariableFormat,Description) %>%
  unique() %>%
  mutate(BaseToken=str_remove_all(BaseToken,"_"))

# Extract lab values and append ER name
labs.CENTER.TBI.dictionary <- CENTER.TBI.dictionary %>%
  filter(str_starts(BaseToken,'DL')) %>%
  mutate(BaseToken = str_replace(BaseToken,'DL','ER'),
         Description = str_c('In emergency room: ',Description))

# Merge with full CENTER-TBI dictionary
CENTER.TBI.dictionary <- rbind(CENTER.TBI.dictionary,labs.CENTER.TBI.dictionary)

# Merge full dictionary information to v6-0 dictionary
v6.token.dictionary <- v6.token.dictionary %>%
  left_join(CENTER.TBI.dictionary)

# Identify missing-description ER imaging values
missing.desc.imgaing <- v6.token.dictionary %>%
  filter(is.na(Description),
         Type=='Brain Imaging',
         str_starts(BaseToken,'ER')) %>%
  mutate(BaseToken=str_remove(BaseToken,'ER')) %>%
  select(-c(VariableFormat,Description)) %>%
  left_join(CENTER.TBI.dictionary) %>%
  mutate(Description = str_c('In emergency room: ',Description),
         BaseToken=str_c('ER',BaseToken)) %>%
  rename(FillVariableFormat = VariableFormat,
         FillDescription = Description)

# Merge ER imaging dictionary information to v6-0 dictionary
v6.token.dictionary <- v6.token.dictionary %>%
  left_join(missing.desc.imgaing)
v6.token.dictionary$Description[(is.na(v6.token.dictionary$Description))&(!is.na(v6.token.dictionary$FillDescription))] <- v6.token.dictionary$FillDescription[(is.na(v6.token.dictionary$Description))&(!is.na(v6.token.dictionary$FillDescription))]
v6.token.dictionary$VariableFormat[(is.na(v6.token.dictionary$VariableFormat))&(!is.na(v6.token.dictionary$FillVariableFormat))] <- v6.token.dictionary$FillVariableFormat[(is.na(v6.token.dictionary$VariableFormat))&(!is.na(v6.token.dictionary$FillVariableFormat))]
v6.token.dictionary <- v6.token.dictionary %>%
  select(-c(FillDescription,FillVariableFormat)) %>% 
  group_by(BaseToken) %>% 
  slice(1)

# Load and merge category information per variable
ais.categories <- read.csv('../CENTER-TBI/AIS/variables.csv',
                           na.strings = c("NA","NaN","", " ")) %>%
  select(name,categories) %>%
  unique() %>%
  rename(BaseToken = name,
         PossibleValues = categories) %>%
  mutate(PossibleValues = replace_na(PossibleValues,'Not Applicable'))
biomarkers.categories <- read.csv('../CENTER-TBI/Biomarkers/variables.csv',
                                  na.strings = c("NA","NaN","", " ")) %>%
  select(name,categories) %>%
  unique() %>%
  rename(BaseToken = name,
         PossibleValues = categories) %>%
  mutate(PossibleValues = replace_na(PossibleValues,'Not Applicable'))
central.haemostasis.categories <- read.csv('../CENTER-TBI/CentralHaemostasis/variables.csv',
                                           na.strings = c("NA","NaN","", " ")) %>%
  select(name,categories) %>%
  unique() %>%
  rename(BaseToken = name,
         PossibleValues = categories) %>%
  mutate(BaseToken=str_remove_all(BaseToken,"_"),
         PossibleValues = replace_na(PossibleValues,'Not Applicable'))
daily.hourly.categories <- read.csv('../CENTER-TBI/DailyHourlyValues/variables.csv',
                                    na.strings = c("NA","NaN","", " ")) %>%
  select(name,categories) %>%
  unique() %>%
  rename(BaseToken = name,
         PossibleValues = categories) %>%
  mutate(PossibleValues = replace_na(PossibleValues,'Not Applicable'))
daily.TIL.categories <- read.csv('../CENTER-TBI/DailyTIL/variables.csv',
                                 na.strings = c("NA","NaN","", " ")) %>%
  select(name,categories) %>%
  unique() %>%
  rename(BaseToken = name,
         PossibleValues = categories) %>%
  mutate(PossibleValues = replace_na(PossibleValues,'Not Applicable'))
daily.vitals.categories <- read.csv('../CENTER-TBI/DailyVitals/variables.csv',
                                    na.strings = c("NA","NaN","", " ")) %>%
  select(name,categories) %>%
  unique() %>%
  rename(BaseToken = name,
         PossibleValues = categories) %>%
  mutate(PossibleValues = replace_na(PossibleValues,'Not Applicable'))
demo.categories <- read.csv('../CENTER-TBI/DemoInjHospMedHx/variables.csv',
                            na.strings = c("NA","NaN","", " ")) %>%
  select(name,categories) %>%
  unique() %>%
  rename(BaseToken = name,
         PossibleValues = categories) %>%
  mutate(PossibleValues = replace_na(PossibleValues,'Not Applicable'))
imaging.categories <- read.csv('../CENTER-TBI/Imaging/variables.csv',
                               na.strings = c("NA","NaN","", " ")) %>%
  select(name,categories) %>%
  unique() %>%
  rename(BaseToken = name,
         PossibleValues = categories) %>%
  mutate(PossibleValues = replace_na(PossibleValues,'Not Applicable'))
er.imaging.categories <- imaging.categories %>%
  mutate(BaseToken=str_c('ER',BaseToken))
labs.categories <- read.csv('../CENTER-TBI/Labs/variables.csv',
                            na.strings = c("NA","NaN","", " ")) %>%
  select(name,categories) %>%
  unique() %>%
  rename(BaseToken = name,
         PossibleValues = categories) %>%
  mutate(PossibleValues = replace_na(PossibleValues,'Not Applicable'))
er.labs.categories <- labs.categories %>%
  mutate(BaseToken=str_replace(BaseToken,'DL','ER'))
medications.categories <- read.csv('../CENTER-TBI/Medication/variables.csv',
                                   na.strings = c("NA","NaN","", " ")) %>%
  select(name,categories) %>%
  unique() %>%
  rename(BaseToken = name,
         PossibleValues = categories) %>%
  mutate(PossibleValues = replace_na(PossibleValues,'Not Applicable'))
outcomes.categories <- read.csv('../CENTER-TBI/Outcomes/variables.csv',
                                na.strings = c("NA","NaN","", " ")) %>%
  select(name,categories) %>%
  unique() %>%
  rename(BaseToken = name,
         PossibleValues = categories) %>%
  mutate(PossibleValues = replace_na(PossibleValues,'Not Applicable'))
prior.meds.categories <- read.csv('../CENTER-TBI/PriorMeds/variables.csv',
                                  na.strings = c("NA","NaN","", " ")) %>%
  select(name,categories) %>%
  unique() %>%
  rename(BaseToken = name,
         PossibleValues = categories) %>%
  mutate(PossibleValues = replace_na(PossibleValues,'Not Applicable'))
surgery.cranial.categories <- read.csv('../CENTER-TBI/SurgeriesCranial/variables.csv',
                                       na.strings = c("NA","NaN","", " ")) %>%
  select(name,categories) %>%
  unique() %>%
  rename(BaseToken = name,
         PossibleValues = categories) %>%
  mutate(PossibleValues = replace_na(PossibleValues,'Not Applicable'))
surgery.extra.cranial.categories <- read.csv('../CENTER-TBI/SurgeriesExtraCranial/variables.csv',
                                             na.strings = c("NA","NaN","", " ")) %>%
  select(name,categories) %>%
  unique() %>%
  rename(BaseToken = name,
         PossibleValues = categories) %>%
  mutate(PossibleValues = replace_na(PossibleValues,'Not Applicable'))
transitions.of.care.categories <- read.csv('../CENTER-TBI/TransitionsOfCare/variables.csv',
                                           na.strings = c("NA","NaN","", " ")) %>%
  select(name,categories) %>%
  unique() %>%
  rename(BaseToken = name,
         PossibleValues = categories) %>%
  mutate(PossibleValues = replace_na(PossibleValues,'Not Applicable'))

all.variable.categories <- rbind(ais.categories,biomarkers.categories,central.haemostasis.categories,daily.hourly.categories,daily.TIL.categories,daily.vitals.categories,demo.categories,imaging.categories,er.imaging.categories,labs.categories,er.labs.categories,medications.categories,outcomes.categories,prior.meds.categories,surgery.cranial.categories,surgery.extra.cranial.categories,transitions.of.care.categories)
v6.token.dictionary <- v6.token.dictionary %>%
  left_join(all.variable.categories)

## Manual inspection of dictionary
# Save dictionary for manual inspection and bespoke filling
write.csv(v6.token.dictionary,'../CENTER-TBI/full_variable_dictionary.csv',row.names = FALSE)

## Create list of physician-impression variables
# Load delta variable table
delta.variables <- read_xlsx('../delta_variables.xlsx',sheet = 'removed_from_old') %>%
  filter(`Keep?` == 'Y',ActuallyNotDropped == 'N') %>%
  select(BaseToken) %>%
  left_join(v6.token.dictionary)

# Save physician impression dictionary for manual inspection and bespoke filling
write.csv(delta.variables,'../CENTER-TBI/physician_impression_variables.csv',row.names = FALSE)

### XV. Create table of cutoffs defining significant transitions
# Load dataframe of significant transitions
sig.transitions.df <- read.csv('../model_interpretations/v6-0/timeSHAP/significant_transition_points.csv',
                               na.strings = c("NA","NaN","", " ")) %>%
  # filter(WindowIdx > 4) %>%
  mutate(DaysAfterICUAdmission = WindowIdx/12)

# Extract cutoffs
cutoff.table <- sig.transitions.df %>%
  select(Threshold,Cutoff) %>%
  unique()

### XVI. Stacked proportion barplots of characteristics over time
## Prepare dataframe
# Load dataframe of characteristics over time
char.over.time <- read.csv('../CENTER-TBI/characteristics_over_time.csv',
                           na.strings = c("NA","NaN","", " "))

# Focus on first 7 days
char.over.time <- char.over.time %>%
  mutate(DaysAfterICUAdmission = WindowIdx/12)


char.over.time %>%
  filter(DaysAfterICUAdmission<=7,
         Characteristic=='GOSE')%>%
  ggplot(aes(x=DaysAfterICUAdmission,y=Count,fill=forcats::fct_rev(as.factor(Value)))) +
  geom_bar(stat = "identity",
           position = "fill")


char.over.time %>%
  filter(DaysAfterICUAdmission<=7,
         Characteristic=='Severity')%>%
  ggplot(aes(x=DaysAfterICUAdmission,y=Count,fill=factor(Value))) +
  geom_bar(stat = "identity",
           position = "fill")

char.over.time %>%
  filter(DaysAfterICUAdmission<=7,
         Characteristic=='Intensity')%>%
  ggplot(aes(x=DaysAfterICUAdmission,y=Count,fill=factor(Value))) +
  geom_bar(stat = "identity",
           position = "fill")

geom_ribbon(aes(ymin=0,ymax=PropRemaining,fill='All patients'),alpha=.2) +
  geom_line(aes(y=PropRemaining),alpha = 1, size=1.3/.pt,color='#ffa600')+
  scale_x_continuous(breaks=seq(0,7,by=1),expand = expansion(mult = c(.00, .01)))+
  scale_y_continuous(expand = expansion(mult = c(.00, .00)))+
  coord_cartesian(xlim=c(0,7),ylim = c(0,100)) +
  ylab('Proportion remaining (%)')+
  xlab('Days since ICU admission')+
  scale_fill_manual(name = 'Variable type',values=c("All patients" = "#ffa600"))+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    strip.text = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    legend.key.size = unit(1.3/.pt,"line")
  )



barPlots <- ggplot(impact.dataframe.long.cat, aes(x = GOSE, y = count, fill = forcats::fct_rev(as.factor(value)))) +
  geom_bar(stat = "identity",
           position = "fill") +
  scale_fill_brewer(palette = "Set2") + 
  facet_wrap(~name, 
             scales = "free_y",
             strip.position = "left",
             labeller = as_labeller(c(GCSm = "Pr(GCSm)",marshall = "Pr(Marshall CT)", unreactive_pupils = "Pr(Unreactive Pupils)") )) +
  ylab('Proportion') +
  xlab('GOSE at 6 months post-injury') +
  theme_classic() +
  theme(strip.text = element_text(size=20), 
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 16, color = "black"),
        axis.text.y = element_text(size = 16, color = "black"),
        axis.title.x = element_text(size = 20),
        axis.title.y = element_blank(), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        strip.background = element_blank(),
        strip.placement = "outside",
        legend.position = "none",
        aspect.ratio = 1)