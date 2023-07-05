# Function to plot relevance layer boxplots
relevance.boxplots <- function(plot.df){
  curr.boxplot <- plot.df %>%
    mutate(
      GROUPS = fct_relevel(GROUPS, 'Top', 'Middle', 'Bottom'),
      BaseToken = fct_reorder(BaseToken, median)
    ) %>% 
    ggplot(aes(y = BaseToken,fill=Type)) +
    geom_boxplot(aes(xmin=min,xmax=max,xlower=Q1,xupper=Q3,xmiddle=median),stat='identity') +
    facet_grid(rows = vars(GROUPS), scales = 'free_y', switch = 'y', space = 'free_y') +
    scale_x_continuous(expand = c(0, 0.1)) +
    scale_y_discrete(expand = c(0,0)) + 
    #scale_fill_manual(values=c("#003f5c", "#444e86", "#955196",'#dd5182','#ff6e54','#ffa600')) +
    xlab('Learned relevance weight')+
    theme_minimal(base_family = 'Roboto Condensed') +
    theme(
      strip.background = element_blank(),
      strip.text.y = element_blank(),
      axis.title.y = element_blank(),
      axis.text.x = element_text(size = 10, color = 'black'),
      axis.text.y = element_text(size = 10, color = 'black',face = 'bold'),
      axis.title.x = element_text(size = 12, color = 'black',face = 'bold'),
      panel.border = element_blank(),
      axis.line.x = element_line(size=1/.pt),
      axis.text = element_text(color='black'),
      legend.position = 'bottom',
      panel.grid.major.y = element_blank(),
      panel.spacing = unit(10, 'points'),
      legend.key.size = unit(1.3/.pt,'line'),
      legend.title = element_text(size = 12, color = 'black',face = 'bold'),
      legend.text=element_text(size=10)
    )
  return(curr.boxplot)
}

# Function to plot category-specific feature-level TimeSHAP values
types.timeSHAP.plots <- function(plot.df){
  curr.timeSHAP.plot <- plot.df %>% 
    ggplot() +
    geom_vline(xintercept = 0, color = "darkgray") +
    geom_quasirandom(aes(y=BaseToken,x=SHAP,color=ColorScale),groupOnX=FALSE,varwidth = FALSE,alpha = 0.6,stroke = 0,size=.75) + 
    scale_color_gradient2(na.value='#488f31',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=.5,limits = c(0,1),breaks = c(0.05,.95), labels = c('Low','High')) +
    theme_minimal(base_family = 'Roboto Condensed') +
    guides(color = guide_colourbar(title='Feature Value',title.vjust=1,barwidth = 10, barheight = .25,ticks = FALSE))+
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
  return(curr.timeSHAP.plot)
}

# Function to plot individual trajectories
indiv.pt.trajectory.plot <- function(test.pred.df,curr.GUPI,axis.title.hjust=.5,legend.just='center'){
  
  filt.df <- test.pred.df %>%
    filter(GUPI==curr.GUPI)
  
  indiv.plot <- filt.df %>%
    ggplot() +
    coord_cartesian(ylim = c(0,100)) +
    scale_x_continuous(breaks=seq(0,max(filt.df$DaysAfterICUAdmission),by=1),expand = expansion(mult = c(.01, .01)))+
    scale_fill_manual(values=c('#003f5c','#444e86','#955196','#dd5182','#ff6e54','#ffa600'))+
    scale_color_manual(values=c('#003f5c','#444e86','#955196','#dd5182','#ff6e54','#ffa600'))+
    geom_hline(yintercept = 50, color='dark gray',alpha = 1, size=1.3/.pt, linetype = "dashed")+
    geom_vline(xintercept = max(filt.df$DaysAfterICUAdmission),
               color='orange',
               alpha = 1,
               size=1.3/.pt,
               linetype = "twodash")+
    geom_line(aes(x=DaysAfterICUAdmission,y=meanProb,color=Threshold),alpha = 1, size=1.3/.pt)+
    geom_ribbon(aes(x=DaysAfterICUAdmission,ymin=loProb,ymax=hiProb,fill=Threshold),alpha=.2) +
    annotate('rect',xmin=(1/12),xmax=(3/12),ymin=0,ymax=100,alpha=0.3,fill='#de425b') +
    ylab('Probability (%)')+
    xlab('Days since ICU admission')+
    guides(fill=guide_legend(title = 'Functional outcome at 6\nmonths post-injury',byrow=TRUE),
           color=guide_legend(title = 'Functional outcome at 6\nmonths post-injury', byrow=TRUE))+
    theme_minimal(base_family = 'Roboto Condensed') +
    theme(
      panel.grid.minor.x = element_blank(),
      axis.text.x = element_text(size = 6, color = "black",margin = margin(r = 0)),
      axis.text.y = element_text(size = 6, color = "black",margin = margin(r = 0)),
      axis.title.x = element_text(size = 7, color = "black",face = 'bold',hjust=axis.title.hjust,margin = margin(r = 0,t=0)),
      axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
      legend.position = 'bottom',
      legend.key.size = unit(1.3/.pt,'line'),
      legend.title = element_text(size = 7, color = 'black',face = 'bold', hjust=1),
      legend.text=element_text(size=6),
      legend.justification = legend.just,
      legend.margin = margin(t=0)
    )
  
  return(indiv.plot)
}

# Function to plot individual feature TimeSHAP barplots
indiv.pt.feature.barplot <- function(timeSHAP.df,curr.GUPI,curr.type,labels.log = F){
  
  filt.timeSHAP.df <- timeSHAP.df %>%
    filter(GUPI==curr.GUPI,
           Baseline==curr.type)
  
  if (labels.log) {
    
    # Load manually created labels of unique `Tokens` in plotting datafame
    indiv.timeSHAP.var.df <- read_xlsx('../model_interpretations/v6-0/timeSHAP/individual_timeSHAP_labels_filled.xlsx') %>%
      select(-Baseline)
    
    # Merge manually created labels to filtered TimeSHAP plotting dataframe
    filt.timeSHAP.df <- filt.timeSHAP.df %>%
      left_join(indiv.timeSHAP.var.df) %>%
      mutate(PLOT_LABEL = fct_reorder(PLOT_LABEL, -RankIdx),
             Token = fct_reorder(Token, -RankIdx)) %>%
      mutate(Token = PLOT_LABEL)
    
  } else {
    filt.timeSHAP.df <- filt.timeSHAP.df %>%
      mutate(Token = fct_reorder(Token, -RankIdx)) 
  }
  
  indiv.timeshap.barplot <- filt.timeSHAP.df %>%
    ggplot() +
    geom_vline(xintercept = 0, color = "darkgray") +
    geom_col(aes(y=Token,x=SHAP),width=.6,fill='#003f5c') + 
    theme_minimal(base_family = 'Roboto Condensed') +
    theme(
      strip.background = element_blank(),
      strip.text = element_blank(),
      axis.title.y = element_blank(),
      axis.text.x = element_text(size = 5, color = 'black'),
      axis.text.y = element_text(size = 6, color = 'black',angle = 30, hjust=1),
      axis.title.x = element_blank(),
      panel.border = element_blank(),
      axis.line.x = element_line(size=1/.pt),
      axis.text = element_text(color='black'),
      legend.position = 'none',
      panel.grid.major.y = element_blank(),
      panel.spacing = unit(10, 'points'),
      plot.margin=grid::unit(c(0,0,0,0), "mm")
    )
  
  return(indiv.timeshap.barplot)
}

# Function to plot individual event TimeSHAP heatmaps
indiv.pt.event.heatmap <- function(timeSHAP.event.df,curr.GUPI){
  
  filt.df <- timeSHAP.event.df %>%
    filter(GUPI==curr.GUPI)
  
  indiv.event.timeshap.plot <- filt.df %>%
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
  
  return(indiv.event.timeshap.plot)
}