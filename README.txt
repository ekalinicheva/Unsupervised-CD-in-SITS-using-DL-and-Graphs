This repository is not complete. Yet.
The folder with segmentation codes will be loaded at the end of the lockdown.

This repository contains files relevant to the article "Unsupervised Change Detection Analysis in Satellite Image Time Series using Deep Learning Combined with Graph-Based Approaches"

The files GT_Classes_Rostov2.TIF and GT_Classes_Montpellier1.TIF correspond to the groung truth for change clustering in SITS.

The following folders containing different steps of the proposed algorithm are available:
  - SITS_encoding - encode each image if SITS in feature respresentation
  - Bitemporal_change_detection - contains files for bi-temporal change detection and their multi-temporal interpretation
  - Set_covering - contains scripts for graph creation and synopsis computation
  - Outliers_clustering_LSTM_GRU_RNN -  performs clustering of extracted graphs
