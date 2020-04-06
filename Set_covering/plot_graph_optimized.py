import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.pyplot import text
from matplotlib.lines import Line2D
import os, time, re
from codes.image_processing import *
csfont = {'fontname':'Times New Roman'}



def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# The parameters from the article
alpha = 0.5
t1 = 0.4    # tau1
t2 = 0
t3 = 0.2    # tau3
bb_id = [8, 121]     # BB's id [img_id, seg_id]
image, ind = bb_id

# Montpellier
path_results_seg_series = os.path.expanduser(
    '~/Desktop/Results/Segmentation_outliers_upd_filled/Montpellier_SPOT5_graph_cut_series_2D/')
seg_folder_series = "series_sigma_0.1_k_7_min_10_bands_3_threshold_int_0.4/"


# # Rostov
# path_results_seg_series = os.path.expanduser(
#     '~/Desktop/Results/Segmentation_outliers_upd_filled/Rostov_S2_graph_cut_series_2D/')
# seg_folder_series = "series_sigma_0.1_k_7_min_10_bands_3_threshold_0995_threshold_int_0.5_17img/"


path_results = path_results_seg_series + seg_folder_series + "Graph_coverage_filtered/"
path_results_final = path_results + "alpha_" + str(alpha) + "_t1_" + str(t1) + "_t2_" + str(t2) + "_t3_" + str(t3) + "/"
bb_final_list = np.load(path_results_final + "Graph_list_synopsys_alpha_" + str(alpha) + "_t1_" + str(t1) + "_t2_" + str(t2) + "_t3_" + str(t3)+"_"+folder_encoded+".npy")


# We open segmented images.
segm_array_list = []
date_list = []
image_name_segm_list = np.sort(list(filter(lambda f: (f.endswith(".TIF") and f.startswith("Segments_1D_20")), os.listdir(path_results))))
nbr_images = len(image_name_segm_list)
print(nbr_images)
for i in range(nbr_images):
    image_name_segm = image_name_segm_list[i]
    date = (re.search("_([0-9]*).TIF", image_name_segm)).group(1)
    print(date)
    if i!= 0:
        image_name_segm_prev = image_name_segm_list[i - 1]
        date_prev = (re.search("_([0-9]*).", image_name_segm_prev)).group(1)
    if i != len(image_name_segm_list)-1:
        image_name_segm2 = image_name_segm_list[i + 1]
        date2 = (re.search("_([0-9]*).", image_name_segm2)).group(1)
    date_list.append(date)
    image_array_seg, H, W, geo, proj, bands_nb = open_tiff(path_results,
                                                           os.path.splitext(image_name_segm)[0])

    segm_array_list.append(image_array_seg)

# We find the graph by BB's id
bb_index = np.intersect1d(np.where(bb_final_list[:, 0] == image)[0],
                                  np.where(bb_final_list[:, 1] == ind)[0])[0]
graph_info = bb_final_list[bb_index]
# the graph, the segments at the same timestamp are not in the good order "geographically".
# it means that the nodes are not organized by geographical prowimity to each other, and when we connect the nodes, the edges will look disorganised.
sorted_graph_unorg = graph_info[6]
timestamps = np.unique(sorted_graph_unorg[:,0])
nb_timestamps = len(timestamps)


sorted_graph = None
for t in timestamps:
    segments_at_timestamp = sorted_graph_unorg[sorted_graph_unorg[:, 0] == t]
    image_array_seg = segm_array_list[t]
    ind_list=[]
    dist_list = []
    for seg in segments_at_timestamp:
        ind = np.transpose(np.where(np.transpose(image_array_seg) == seg[1]))   # we extract coordinates of each segment (x,y) in image pixels
        mean_ind = np.mean(ind, axis=0, dtype=int)  # we find the approximate center of the segments
        dist = np.sqrt(mean_ind[0]**2 + mean_ind[1]**2)     # we measure distance from top left corner (0,0) to the center
        ind_list.append(mean_ind)
        dist_list.append(dist)
    sort_ind = np.argsort(dist_list)
    new_sorted = np.asarray(segments_at_timestamp)[sort_ind]    # we sort by the distance from corner
    # we append sorted objects to a sorted graph
    if sorted_graph is None:
        sorted_graph = new_sorted
    else:
        sorted_graph = np.concatenate((sorted_graph, new_sorted), axis=0)
sorted_graph = np.reshape(np.asarray(sorted_graph), (-1, 2))    # all good now




graph_sceleton = [] # list with nbr of segments at each timestamp
connections = []
for t in timestamps:
    segments_at_timestamp = sorted_graph[sorted_graph[:, 0] == t]
    nb_seg = segments_at_timestamp.shape[0]   # nb of segments at this timestamp
    graph_sceleton.append(nb_seg)
    image_array_seg = segm_array_list[t].flatten()
    if t < max(timestamps):
        image_array_seg_next = segm_array_list[t+1].flatten()
        # We look to the segments at next timestamp that are connected to this timestamp
        for seg in segments_at_timestamp:
            ind = np.where(image_array_seg == seg[1])[0]
            connected = np.intersect1d(np.setdiff1d(np.unique(image_array_seg_next[ind]), [0]), sorted_graph[:, 1][sorted_graph[:, 0] == t+1])
            connections.append([seg, connected])    # edge
max_graphs = np.max(graph_sceleton) #the widest part of the graph
connections = np.asarray(connections, dtype=object)
# print(connections)


# Parameters to draw the ellipses (nodes) that correspond to segments
# Better not to touch
ell_width = 0.45
ell_height = 0.35
space_width = 0.01
space_height = 0.75
fig_width = max_graphs * ell_width + space_width * (max_graphs - 1)
fig_height = nb_timestamps * ell_height + space_height * (nb_timestamps - 1)



# print(fig_width, fig_height)
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

ell_width = ell_width/fig_width
ell_height = ell_height/fig_height
space_width = space_width/fig_width
space_height = space_height/fig_height

# We draw ellipces and we put segments numbers inside
y_start = 1 - ell_height/2
coordinates = []
for n in range(nb_timestamps):
    t = timestamps[n]
    segments_at_timestamp = sorted_graph[sorted_graph[:, 0] == t]
    nb_seg = ((sorted_graph[:, 0])[sorted_graph[:, 0] == t]).shape[0]
    if nb_seg == max_graphs:
        x_start = ell_width/2
    else:
        diff_w_max = max_graphs - nb_seg
        x_start = (diff_w_max / 2 * (ell_width + space_width)) + (ell_width/2)
    for s in range(nb_seg):
        seg = segments_at_timestamp[s]
        x = x_start + s * (ell_width + space_width)
        xy = [x, y_start]
        e = Ellipse(xy, ell_width, ell_height, angle=0)
        e.set_facecolor("white")
        e.set_edgecolor("black")
        ax.add_artist(e)
        coordinates.append([seg, xy])
        t = text(x, y_start, str(seg[0])+"-"+str(seg[1]), horizontalalignment='center', verticalalignment = 'center', transform = ax.transAxes,
                 fontname = 'Times New Roman')
        t.set_fontsize(7)
    y_start -= (ell_height + space_height)
coordinates = np.asarray(coordinates, dtype=object)



# We draw the edges
for n in range(len(connections)):
    c = connections[n]
    coord1 = coordinates[n][1]
    print(coord1)
    coord1 = coord1[0], coord1[1] - ell_height/2
    seg = c[0]
    connected_to = c[1]
    for seg_con in connected_to:
        coord2 = coordinates[np.intersect1d(np.where(coordinates[:, 0, 0] == seg[0]+1)[0], np.where(coordinates[:, 0, 1] == seg_con)[0])][0][1]
        coord2 = coord2[0], coord2[1] + ell_height / 2
        print([coord1[0], coord2[0]], [coord1[1], coord2[1]])
        l = Line2D([coord1[0], coord2[0]], [coord1[1], coord2[1]], lw=0.5, color="black")
        ax.add_line(l)
    coord1 = coord1[0], coord1[1] - ell_height/2


plt.savefig("/home/user/Dropbox/article_Montpellier/figures/graph_plot.svg", format="svg")
plt.show()
