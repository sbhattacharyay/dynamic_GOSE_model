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