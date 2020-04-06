import numpy as np
import os
from codes.image_processing import open_tiff
from codes.stats_scripts import *

nb_final_clusters = 10
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


alpha = 0.5
t1 = 0.4
t2 = 0
t3 = 0.3

print("alpha_"+str(alpha)+"_t1_"+str(t1)+"_t2_"+str(t2) + "_t3_" + str(t3))

#This is the main path to segmentation folders
path_results_seg_series = os.path.expanduser('~/Desktop/Results/Segmentation_outliers_upd_filled/Montpellier_SPOT5_graph_cut_series_2D/')
#This is the folder with the segmentation with chosen parameters
seg_folder_series = "series_sigma_0.5_k_3_min_10_bands_3/"
# Results folder. This folder contains final segmentation results as well as list of BBs with chosen alpha.
# Segmentation should be a 2D raster image where each segment has its individual id.
# In this images, only change areas are segmented, change areas have 0 id value.
path_results = path_results_seg_series+seg_folder_series + "Graph_coverage_filtered/"
# Folder with chosen graph construction parameters
path_results_final = path_results+"alpha_"+str(alpha)+"_t1_"+str(t1)+"_t2_"+str(t2)+"_t3_"+str(t3)+"/"
# We open a list with BBs and corresponding graphs
bb_final_list = np.load(path_results_final + "Graph_list_alpha_"+str(alpha)+"_t1_"+str(t1)+"_t2_"+str(t2)+"_t3_"+str(t3)+".npy")


stats_file = path_results_final + "stats_alpha_" + str(alpha) + "_t1_" + str(t1) + "_t2_" + str(t2) + "_t3_" + str(t3) + ".txt"


# We open rasters with BBs and graphs (see final_bb_constraint)
covered_grids_flatten_to_be_filled_list = np.sort(list(filter(lambda f: (f.endswith(".TIF") and f.startswith("Filled_grid_2")), os.listdir(path_results_final))))
covered_grids_flatten_to_be_filled_by_bb_list = np.sort(list(filter(lambda f: (f.endswith(".TIF") and f.startswith("Filled_grid_by_bb_")), os.listdir(path_results_final))))
nbr_img = len(covered_grids_flatten_to_be_filled_list)
covered_grids_flatten_to_be_filled_by_bb = []
for i in range(nbr_img):
    grid_filled_name = covered_grids_flatten_to_be_filled_list[i]
    grid_filled_by_bb_name = covered_grids_flatten_to_be_filled_by_bb_list[i]

    grid_filled, H, W, geo, proj, bands_nb = open_tiff(path_results_final, os.path.splitext(grid_filled_name)[0])
    grid_filled_by_bb, H, W, geo, proj, bands_nb = open_tiff(path_results_final, os.path.splitext(grid_filled_by_bb_name)[0])
    covered_grids_flatten_to_be_filled_by_bb.append(np.transpose(np.reshape(grid_filled_by_bb,(bands_nb, H*W))))
    unique, count = np.unique(grid_filled, return_counts=True)
    if 0 in unique:
        perc =int(count[1]/(count[1]+count[2])*100)
    else:
        perc=0

    print_stats(stats_file, ("Image " + str(i) + " has " + str(perc) + "% uncovered pixels"), print_to_console=True)

covered_grids_flatten_to_be_filled_by_bb = np.asarray(covered_grids_flatten_to_be_filled_by_bb)


stats_about_bb = []
graph_overlapping = np.zeros((H*W))


# We compute overlapping and graph compactness (see article)
for all_bb in bb_final_list[:,:2]:
    img, ind = all_bb
    bb_footprint_ind = np.intersect1d(np.where(covered_grids_flatten_to_be_filled_by_bb[img][:,0].flatten() == img)[0], np.where(covered_grids_flatten_to_be_filled_by_bb[img][:,1].flatten() == ind)[0])
    bb_footprint = len(bb_footprint_ind)
    graph_footprint_ind = None
    for i in range(nbr_img):
        graph_footprint_ind_i = np.intersect1d(np.where(covered_grids_flatten_to_be_filled_by_bb[i][:,0] == img)[0],
                                          np.where(covered_grids_flatten_to_be_filled_by_bb[i][:,1] == ind)[0])
        if graph_footprint_ind is None:
            graph_footprint_ind = graph_footprint_ind_i
        else:
            graph_footprint_ind = np.concatenate((graph_footprint_ind, graph_footprint_ind_i))

    graph_footprint_ind = np.unique(graph_footprint_ind).astype(int)
    graph_footprint = len(graph_footprint_ind)
    graph_compactness = (bb_footprint/graph_footprint)*100
    # print(graph_compactness)
    stats_about_bb.append([img, ind, graph_compactness])
    graph_overlapping[graph_footprint_ind]+=1
stats_about_bb = np.asarray(stats_about_bb)
overall_overlapping = (len(np.where(graph_overlapping>1)[0])/len(np.where(graph_overlapping>0)[0]))*100
print_stats(stats_file, ("The whole set has " + str(overall_overlapping) + "% graph overlapping"), print_to_console=True)
graph_compactness_all = stats_about_bb[:,2]
compactness_min, compactness_max, compactness_mean = np.min(graph_compactness_all), np.max(graph_compactness_all), np.mean(graph_compactness_all)
print_stats(stats_file, ("Compactness: min " + str(compactness_min) + " max " + str(compactness_max) + " mean " + str(compactness_mean)), print_to_console=True)
graph_compactness_all_sorted = np.sort(np.copy(graph_compactness_all))
quantile = int(len(graph_compactness_all_sorted)/4)
print_stats(stats_file, "Compactness less that 0.5 = "+str(len(graph_compactness_all_sorted[graph_compactness_all_sorted<50])/len(graph_compactness_all_sorted)*100)+"%", print_to_console=True)
print_stats(stats_file, "Compactness less that 0.75 = "+str(len(graph_compactness_all_sorted[graph_compactness_all_sorted<75])/len(graph_compactness_all_sorted)*100)+"%", print_to_console=True)

q1, q2, q3, q4 = graph_compactness_all_sorted[:quantile], graph_compactness_all_sorted[quantile:quantile*2], graph_compactness_all_sorted[quantile*2:quantile*3], graph_compactness_all_sorted[quantile*3:len(graph_compactness_all_sorted)]
for q in [q1, q2, q3, q4]:
    print_stats(stats_file, "quartile min max mean "+str(np.min(q))+" "+str(np.max(q)) + " "+ str(np.mean(q)), print_to_console=True)
