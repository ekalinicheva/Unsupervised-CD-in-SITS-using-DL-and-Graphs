from osgeo import gdal, ogr
import numpy as np
import os, time, re, datetime
from codes.image_processing import create_tiff, vectorize_tiff, open_tiff


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

#We choose alpha parameter from the article for BB construction.
alpha = 0.4

#This is the main path to segmentation folders
path_results_seg_series = os.path.expanduser('~/Desktop/Results/Segmentation_outliers_upd_filled/Montpellier_SPOT5_graph_cut_series_2D/')
#This is the folder with the segmentation with chosen parameters
seg_folder_series = "series_sigma_0.5_k_3_min_10_bands_3/"
# Results folder. This folder contains final segmentation results obtained in previous step.
# Segmentation should be a 2D raster image where each segment has its individual id.
# In this images, only change areas are segmented, change areas have 0 id value.
path_results = path_results_seg_series+seg_folder_series + "Graph_coverage_filtered/"

# Start time
start_time = time.clock()
run_name = "." + str(time.strftime("%Y-%m-%d_%H%M"))
print(run_name)

# We create a list where we will write candidate BB
bb_candidates_list = None

# We open segmented images and construct candidates bb list. When all the segmented images are iterated we will start to calculate weights.
# The list columns are Image_Id, Segm_Id, Segm_Size, Segm_Weight
segm_array_list = []    # future stack of segmented images
date_list = []  # image dates
outliers_total_list = []    # list with the corresponding outliers masks (changes/ no changes)
outliers_total = None
# We filter out only corresponding raster segmentation images. The list is sorted by date.
# An image name should be like "Segments_20021005_sigma_0.5_k_3_min_10_bands_3", where 20021005 is date
image_name_segm_list = np.sort(list(filter(lambda f: (f.endswith(".TIF") and f.startswith("Segments_1D_20")), os.listdir(path_results))))
nbr_images = len(image_name_segm_list)
print(nbr_images)
# We iterate through segmentation images to make a stack of them
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

    # For each segmentation image we create a mask with no change areas
    # 0 - no changes, 1 - changes (corresponds to the segmented area)
    outliers_array = np.zeros((H, W))
    outliers_array[image_array_seg != 0] = 1

    # outliers_total correspond to global change/no change mask
    if outliers_total is None:
        outliers_total = np.zeros((H, W))
    outliers_total += outliers_array

    outliers_total_list.append(outliers_array)

    # We write all the segments to candidate BB list ([image_id, seg_id, seg_size, seg_weight=0])
    unique, count = np.unique(image_array_seg.flatten()[np.where(image_array_seg.flatten() != 0)[0]], return_counts=True)
    candidates_bb_image = np.transpose([np.full(len(unique), i), unique, count, np.full(len(unique), None)])
    if bb_candidates_list is None:
        bb_candidates_list = candidates_bb_image
    else:
        bb_candidates_list = np.concatenate((bb_candidates_list, candidates_bb_image), axis=0)
bb_candidates_list = np.asarray(bb_candidates_list, dtype=object)



# We get indicies to sort the candidate BB in descending order
order = np.flip(bb_candidates_list[:, 2].argsort(), axis=0)
bb_candidates_list = bb_candidates_list[order]


outliers_total_list = np.asarray(outliers_total_list)
outliers_total[outliers_total != 0] = 1
grid_size = np.count_nonzero(outliers_array)    #size of the maximum coverage area of outliers

#we create a grid that we will fill with bb. -1 - no change area, 0 - not covered grid, >=1 - covered grid (if >1, there is overlapping)
#at the initialization stage, we have only -1 and 0 values
covered_grid = np.copy(outliers_total) - 1
covered_grid_flatten = covered_grid.flatten()

#we create a grid that we will fill with bb. -1 - no change area, 0 - not covered grid, >=1 - covered grid (if >1, there is overlapping)
#at the initialization stage, we have only -1 and 0 values
covered_grids = np.copy(outliers_total_list) - 1
covered_grids_flatten = np.reshape(covered_grids, (nbr_images, H*W))

# We iterate through the list of candidate BB and we compute their novelty and weight
for c in range(len(bb_candidates_list)):
    candidate = bb_candidates_list[c]
    image, ind, size = candidate[:3]
    if size > 4:    #we optinally set the minimum size for a BB
        print(image, ind, size)
        coverage_ind = np.where(segm_array_list[image].flatten()==ind)[0]   # we get pixel indicies of the segment
        novelty_pixels = covered_grid_flatten[coverage_ind] # we get values from covered grid to find out whether this footprint is already covered
        novelty_size = len(np.where(novelty_pixels == 0)[0])    # pixels that are not covered by any BB yet
        novelty = novelty_size/size     # novelty value
        # To better inderstand, read the article
        if novelty == 1:
            bb_candidates_list[c, 3] = size
            covered_grid_flatten[coverage_ind] += 1  # we fill the grid with covered pixels
        elif alpha <= novelty < 1:
            bb_candidates_list[c, 3] = novelty
            # we recompute the weight of other candidates as there is an intersection
            not_novelty_coverage_ind = np.intersect1d(coverage_ind, np.where(covered_grid_flatten >= 1)[0], assume_unique=True)
            covered_grid_flatten[coverage_ind] += 1  # we fill the grid with covered pixels
            for im in range(nbr_images):
                if im != image:
                    image_array_seg_flatten = segm_array_list[im].flatten()
                    segments_to_modify = np.unique(image_array_seg_flatten[not_novelty_coverage_ind])
                    for s in segments_to_modify:
                        if s!= 0:
                            #   we check only BBs that have been already visited and we modify their weight
                            index_bb_to_modify = np.intersect1d(np.where(bb_candidates_list[:c, 0]==im)[0], np.where(bb_candidates_list[:c, 1]==s)[0])
                            if len(index_bb_to_modify) > 0:
                                index_bb_to_modify = index_bb_to_modify[0]
                                index_of_to_be_modified, size_of_to_be_modified = bb_candidates_list[index_bb_to_modify, 1:3]
                                coverage_ind_of_to_be_modified = np.where(image_array_seg_flatten == index_of_to_be_modified)[0]
                                if bb_candidates_list[index_bb_to_modify, 3] != 0:
                                    novelty_to_be_modified = len(np.where(covered_grid_flatten[coverage_ind_of_to_be_modified]==1)[0])/size_of_to_be_modified
                                    if novelty_to_be_modified >= alpha:
                                        bb_candidates_list[index_bb_to_modify, 3] = novelty_to_be_modified
                                    else:
                                        bb_candidates_list[index_bb_to_modify, 3] = 0
                                        covered_grid_flatten[coverage_ind_of_to_be_modified] -= 1  # as bb is not novel anymore, we take it off from the coverage grid
        else:
            bb_candidates_list[c, 3] = 0
    else:
        bb_candidates_list[c, 3] = 0

# We create a raster with coverage pixels
ds = create_tiff(1, path_results + "BB_coverage_alpha_"+str(alpha)+".TIF", W, H,
                 gdal.GDT_Int16, np.reshape(covered_grid_flatten, (H, W)),  geo,
                 proj)


# We sort candidates BBs by descending order
bb_candidates_list_by_weight = np.copy(bb_candidates_list[np.flip(bb_candidates_list[:, 3].argsort(), axis=0)])
# We create the final list with final BBs that have weight greater than alpha (in our case all weights < aplha correspond to 0)
bb_final_list = bb_candidates_list_by_weight[bb_candidates_list_by_weight[:, 3] > 0]

# We save these lists
np.save(path_results + "Candidates_BB_list_alpha_"+str(alpha), bb_final_list)
np.save(path_results + "All_BB_list_alpha_"+str(alpha), bb_candidates_list_by_weight)

# We create rasters for each image where we mark
bb_summary = np.zeros((nbr_images, H*W))
for bb in bb_final_list:
    image, ind, size, weight = bb
    bb_summary[image][segm_array_list[image].flatten() == ind] = 1

bb_summary = np.reshape(bb_summary, (nbr_images, H, W))
ds = create_tiff(nbr_images, path_results + "BB_summary_alpha_"+str(alpha)+".TIF", W, H,
                 gdal.GDT_Byte, bb_summary,  geo,
                 proj)

end_time = time.clock()
total_time_pretraining = end_time - start_time
total_time_pretraining = str(datetime.timedelta(seconds=total_time_pretraining))

print("Total time BB construction =" + str(total_time_pretraining) + "\n")

