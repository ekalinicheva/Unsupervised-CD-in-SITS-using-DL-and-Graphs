from osgeo import gdal, ogr
import numpy as np
import os, re
from codes.image_processing import create_tiff, vectorize_tiff, open_tiff


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# The parameters from the article, the same as in previous codes
alpha = 0.5
t1 = 0.4    # tau1
t2 = 0
t3 = 0.2    # tau3


print("alpha_"+str(alpha)+"_t1_"+str(t1)+"_t2_"+str(t2)+"_t3_"+str(t3))


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

# We add 3 more empty columns
for i in range(3):
    bb_final_list = np.c_[bb_final_list, np.full(len(bb_final_list), None)]


# Path and folder with encoded images
main_path_encoded = os.path.expanduser('~/Desktop/Results/Encode_Montpellier_TS_outliers/')
folder_encoded = "patch_9_feat_5.2019-09-03_1619_noise1"
path_encoded = main_path_encoded + "All_images_" + folder_encoded + "/"


# We open encoded and segmented images
path_list = []
list_image_encoded = []
list_image_date = []
segm_array_list = []
encoded_image_name_list = np.sort(list(filter(lambda f: (f.endswith(".TIF") and f.startswith("Encoded_")), os.listdir(path_encoded))))
for img in range(len(encoded_image_name_list)):
    image_name = encoded_image_name_list[img]
    image_name = os.path.splitext(image_name)[0]
    date = re.search("Encoded_([0-9]*)", str(image_name)).group(1)
    print(date)
    image_array, H, W, geo, proj, feat_nb = open_tiff(path_encoded, image_name)
    try:
        image_array_seg, H, W, geo, proj, _ = open_tiff(path_results,
                                                        "Segments_1D_"+str(date))
        segm_array_list.append(image_array_seg)
        list_image_encoded.append(image_array)
        list_image_date.append(date)
    except:
        continue
sort_ind = np.argsort(list_image_date)
list_image_date = np.asarray(list_image_date)[sort_ind]
list_image_encoded = np.asarray(list_image_encoded, dtype=float)[sort_ind]
nbr_images = len(list_image_date)
print(nbr_images)
segm_array_list = np.asarray(segm_array_list)[sort_ind]


#We normalize images
#Please check that for your dataset nodata is not taken into account, while computing min, max, meand and std
all_images_band = list_image_encoded[:, 0, :, :].flatten()
ind_data = np.where(all_images_band >= -1)[0]
list_norm = []
for band in range(len(list_image_encoded[0])):
    all_images_band = list_image_encoded[:, band, :, :].flatten()
    all_images_band=all_images_band[ind_data]
    min = np.min(all_images_band)
    max = np.max(all_images_band)
    mean = np.mean(all_images_band)
    std = np.std(all_images_band)
    list_norm.append([min, max, mean, std])
    print([min, max, mean, std])

for i in range(len(list_image_encoded)):
    for band in range(len(list_image_encoded[0])):
        list_image_encoded[i][band] = (list_image_encoded[i][band] - list_norm[band][2]) / list_norm[band][3]

list_norm = []
for band in range(len(list_image_encoded[0])):
    all_images_band = list_image_encoded[:, band, :, :].flatten()
    all_images_band = all_images_band[ind_data]
    min = np.min(all_images_band)
    max = np.max(all_images_band)
    list_norm.append([min, max])
    print([min, max])


for i in range(nbr_images):
    for band in range(len(list_image_encoded[0])):
        list_image_encoded[i][band] = (list_image_encoded[i][band]-list_norm[band][0])/(list_norm[band][1]-list_norm[band][0])


image_array_encoded_tr = np.transpose(
    np.reshape(list_image_encoded, (nbr_images, feat_nb, H * W)), (0,2,1))


# We iterate through BBs list
for c in range(len(bb_final_list)):
    candidate = bb_final_list[c]
    image, ind, size, weight, neigh, _, _, _, _ = candidate
    print(image, ind, size, weight)
    graph = np.concatenate((neigh, [[image, ind]]), axis=0)     # we concatenate BB with its attached objects
    sorted_ind = np.argsort(graph[:,0])     #we sort the graph by timestamps
    graph_sorted = graph[sorted_ind]
    bb_final_list[c][6] = graph_sorted      # we write it down to BBs list
    graph_timestamps = np.unique(graph_sorted[:, 0])    # we extract timestamps presented in graphs
    all_values_for_synopsys_median = np.zeros((len(graph_sorted), feat_nb))     # we create synopsys with meadian values
    all_values_for_synopsys_mean = np.zeros((len(graph_sorted), feat_nb))   # we create synopsys with mean values
    all_segment_sizes = np.zeros((len(graph_sorted)))
    # we get encoded values for each segment
    for g in range(len(graph_sorted)):
        seg = graph_sorted[g]
        im_seg, ind_seg = seg
        coverage_ind = np.where(segm_array_list[im_seg].flatten() == ind_seg)[0]
        encoded_values_seg = image_array_encoded_tr[im_seg][coverage_ind]
        median_values = np.median(encoded_values_seg, axis=0)
        mean_values = np.mean(encoded_values_seg, axis=0)
        all_values_for_synopsys_median[g] = median_values
        all_values_for_synopsys_mean[g] = mean_values
        all_segment_sizes[g] = len(coverage_ind)
    # we compute synopsys
    synopsys = np.zeros((len(graph_timestamps), feat_nb))
    synopsys_mean = np.zeros((len(graph_timestamps), feat_nb))
    for t in range(len(graph_timestamps)):
        timestamp = graph_timestamps[t]
        ind = np.where(graph_sorted[:,0]==timestamp)[0]
        synopsys[t]= np.sum(all_values_for_synopsys_median[ind] * np.transpose([all_segment_sizes[ind]]), axis=0) / np.sum(all_segment_sizes[ind])
        synopsys_mean[t] = np.sum(all_values_for_synopsys_mean[ind] * np.transpose([all_segment_sizes[ind]]), axis=0) / np.sum(
            all_segment_sizes[ind])


    bb_final_list[c][7] = synopsys
    bb_final_list[c][8] = synopsys_mean

np.save(path_results_final + "Graph_list_synopsys_alpha_"+str(alpha)+"_t1_"+str(t1)+"_t2_"+str(t2)+"_t3_"+str(t3) +"_"+folder_encoded+"_mean_std", bb_final_list)





