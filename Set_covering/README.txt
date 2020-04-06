This folder contains the source codes to construct graphs and compute their synopsys

Required files:
- Rasterized segmentation of SITS (with changes mask)
- Encoded SITS

The script should be used in following order:
1.  set_covering.py - BB selection with alpha parameter
2.  final_bb_constraint.py - graph creation with corresponding tau1 and tau3
3.  synopsys.py - code to compute a synopsys for every graph that was created
4.  plot_graph_optimized_new.py - code to plot a graph, demands the id ([img_id, seg_id]) of the corresponding BB
5.  analysis_final_graphs.py - optional script to compute graph compactness and overall overlapping