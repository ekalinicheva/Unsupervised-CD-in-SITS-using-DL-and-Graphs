import numpy as np
import os

# This is an optional code to get some stats from the obtained clusters (such as nbr of graphs per cluster and their length.


# The parameters from the article
alpha = 0.5
t1 = 0.4    # tau1
t2 = 0
t3 = 0.2    # tau3

folder_clustering = "LSTM_linear_mean/Synopsys_padding_feat_200_lr_0.001.2019-10-04_1157/"

# Montpellier
path_results_seg_series = os.path.expanduser(
    '~/Desktop/Results/Segmentation_outliers_upd_filled/Montpellier_SPOT5_graph_cut_series_2D/')
seg_folder_series = "series_sigma_0.1_k_7_min_10_bands_3_threshold_int_0.4/"
folder_encoded = "patch_9_feat_5.2019-09-03_1627_mean_std"


path_results = path_results_seg_series + seg_folder_series + "Graph_coverage_filtered/"
path_results_final = path_results + "alpha_" + str(alpha) + "_t1_" + str(t1) + "_t2_" + str(t2) + "_t3_" + str(t3) + "/"
bb_final_list = np.load(path_results_final + folder_clustering + "Graph_list_synopsys_clusters_alpha_" + str(alpha) + "_t1_" + str(t1) + "_t2_" + str(t2) + "_t3_" + str(t3)+".npy")

clusters15 = bb_final_list[:,9]
clusters20 = bb_final_list[:,10]
clusters25 = bb_final_list[:,11]
clusters30 = bb_final_list[:,12]
clusters35 = bb_final_list[:,13]
clusters40 = bb_final_list[:,14]
clusters45 = bb_final_list[:,15]
clusters50 = bb_final_list[:,16]
graph_sorted = bb_final_list[:,6]
change_length = []
for g in range(len(graph_sorted)):
    graph = graph_sorted[g]
    timesteps = graph[:,0]
    change_length_graph = np.max(timesteps) - np.min(timesteps)
    change_length.append(change_length_graph)
change_length = np.asarray(change_length)


for cluster_nb in [clusters15, clusters20, clusters25, clusters30, clusters35, clusters40, clusters45, clusters50]:
    for cl in np.sort(np.unique(cluster_nb)):
        bb_cl_ind = np.where(cluster_nb==cl)[0]
        change_length_cl = change_length[bb_cl_ind]
        print("Cluster ", cl, " min ", np.min(change_length_cl), " max ", np.max(change_length_cl), " mean ", np.mean(change_length_cl), " meadian ", np.median(change_length_cl), " nbr_clusters ", len(bb_cl_ind))
