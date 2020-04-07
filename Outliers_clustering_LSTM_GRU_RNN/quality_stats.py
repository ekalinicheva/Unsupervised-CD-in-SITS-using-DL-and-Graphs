import numpy as np
import os, re
from codes.image_processing import open_tiff
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

# Optional script to compute quality statistics (NMI and ARI)


nmi_list = []
nmi_new_list = []
nmi_bi_list = []

iter = list(range(5, 51, 5))


for cl in iter:
    print("Clusters="+str(cl))
    path_results = os.path.expanduser('~/Desktop/Results/Segmentation_outliers_upd_filled/Montpellier_SPOT5_graph_cut_series_2D/series_sigma_0.3_k_6_min_10_bands_3_threshold_int_0.4/Graph_coverage_filtered/alpha_0.4_t1_0.4_t2_0_t3_0.2/')
    loss_folder = "LSTM_linear_mean/Synopsys_padding_feat_150_lr_0.001.2019-10-04_1601/"

    image_name_loss = "Kmeans_initial_clusters_"+str(cl)
    image_name_loss = "Hierarchical_initial_clusters_"+str(cl)


    image_array_cl, H, W, geo, proj, bands_nb = open_tiff(path_results + loss_folder, image_name_loss)


    path_cm = '/home/user/Desktop/Results/Segmentation_outliers_upd_filled/'
    cm_truth_name = "GT_Classes_Montpellier1"
    cm_predicted = image_array_cl.flatten()
    ind = np.where(cm_predicted > 0)[0]


    # print(cm_predicted.shape)
    cm_truth, _, _, _, _, _ = open_tiff(path_cm, cm_truth_name)
    cm_truth = cm_truth.flatten()

    # cm_truth[cm_truth == 1] = 0

    ind = np.intersect1d(np.where(cm_predicted>0)[0], np.where(cm_truth>0)[0])

    cm_truth = cm_truth[ind]

    cm_truth_new = np.zeros_like(cm_truth)
    cm_truth_bi = np.zeros_like(cm_truth)

    # Reorganized GT for Montpellier dataset
    cm_truth_new[cm_truth==2]=1
    cm_truth_new[cm_truth==9]=1
    cm_truth_new[cm_truth==4]=1
    cm_truth_new[cm_truth==3]=2
    cm_truth_new[cm_truth==11]=2
    cm_truth_new[cm_truth==5]=3
    cm_truth_new[cm_truth==10]=3
    cm_truth_new[cm_truth==6]=4
    cm_truth_new[cm_truth==8]=5
    cm_truth_new[cm_truth==7]=7
    cm_truth_new[cm_truth==13]=7
    cm_truth_new[cm_truth==15]=15
    cm_truth_new[cm_truth==12]=8
    cm_truth_new[cm_truth==14]=9
    cm_truth_new[cm_truth==1]=10


    cm_truth_bi[cm_truth==1]=1
    cm_truth_bi[cm_truth==2]=1
    cm_truth_bi[cm_truth==3]=1
    cm_truth_bi[cm_truth==4]=1
    cm_truth_bi[cm_truth==5]=1
    cm_truth_bi[cm_truth==6]=2
    cm_truth_bi[cm_truth==7]=2
    cm_truth_bi[cm_truth==10]=2
    cm_truth_bi[cm_truth==11]=2
    cm_truth_bi[cm_truth==8]=3
    cm_truth_bi[cm_truth==9]=3


    cm_predicted = cm_predicted[ind]




    print("All clusters")
    print(normalized_mutual_info_score(cm_truth, cm_predicted))
    print(adjusted_rand_score(cm_truth, cm_predicted))


    print("Bi clusters")
    print(normalized_mutual_info_score(cm_truth_bi, cm_predicted))
    print(adjusted_rand_score(cm_truth_bi, cm_predicted))



    nmi_list.append(np.round(normalized_mutual_info_score(cm_truth, cm_predicted),2))
    nmi_new_list.append(np.round(normalized_mutual_info_score(cm_truth_new, cm_predicted),2))
    nmi_bi_list.append(np.round(normalized_mutual_info_score(cm_truth_bi, cm_predicted),2))


print(nmi_list)
print(nmi_new_list)
for i in nmi_list:
    print("& "+str(i))

print(nmi_bi_list)
for i in nmi_bi_list:
    print("& "+str(i))