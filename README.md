# FAMS-Dash
Dashboard for items processed using the functionality of the FAMS repository

Takes in a set of combined rankings for different metrics and provides (currently) two tabs for analyzing the data

## Tab 1: Polling Data
Exploring the expert-elicited data via the Ranking and Order classes from FAMS. For each metric :

### For each metric
In the case of models, this could be abstraction, resolution, scope, and/or fidelity WRT a particular response, for technologies or other more general items, this will just be the name of the Ranking

#### Visualize the item scores
i.e. probability that item is best in set WRT that attribute
- Bar chart with cumulative line
- Score distributions

### For each item
i.e. Technology or Model
- Scores for different metrics
- Histogram of rankings (currently assumes not allowed to mark similar, only ranked list)

### Parallel plot
At bottom of tab, follow any given technology score across all of the rankings

## Tab 2: Single-Item Decision-Making
Look at the relative importance of different rankngs in order to understand the multi-attribute space

### TOPSIS
Apply specific weightings for the importance of each given ranking and use TOPSIS to turn the multi-dimensional space into a ranked list of items

### Monte Carlo
Randomly generate a set of n different weightings and show which items fare best independently of weighting
