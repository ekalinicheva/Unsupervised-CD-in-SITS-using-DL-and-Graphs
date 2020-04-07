This folder contains scripts for graph clustering with GRU AE.
The main file is main_clustering_lstm_padding_graph_synopsys.py that performs the clustering and writes the rasters with clustering results.
Then you can inspire yourself with the following files adapted for your own data:
stats_cluster.py - produces statistics about clusters (number of graphs per cluster, average length, etc);
quality_stats.py - computed NMI and ARI for the produced clusters. You shoul adapt it for your own GT data.
