import os, time, re, datetime
import argparse
from osgeo import gdal
from models.model_gru_linear_padding import Encoder, Decoder
from codes.imgtotensor_patches_samples_list_lstm import ImageDataset
from codes.image_processing import open_tiff, create_tiff, vectorize_tiff
from codes.loader import dsloader
from training_functions import *
from codes.stats_scripts import *


# Create directory if does not exist
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# The parameters from the article
alpha = 0.5
t1 = 0.4    # tau1
t2 = 0
t3 = 0.2    # tau3

model = "GRU_linear"

types = ["mean", "median"]  # type of synopsys generalization - mean or median values of segments
type = types[0]


# models = ["GRU", "LSTM", "LSTM_bi", "GRU_linear", "LSTM_linear"]
# model = models[3]
# if model=="GRU":
#     from models.model_gru import Encoder, Decoder
# if model=="LSTM":
#     from models.model_lstm_padding import Encoder, Decoder
# if model=="LSTM_bi":
#     from models.model_lstm_bi import Encoder, Decoder
# if model=="GRU_linear":
#     from models.model_gru_linear_padding import Encoder, Decoder
# if model == "LSTM_linear":
#     from models.model_lstm_linear_padding import Encoder, Decoder

print(model, type)


def main():
    gpu = on_gpu()
    print("ON GPU is "+str(gpu))


    #Parameters
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--patch_size', default=1, type=int)
    parser.add_argument('--nb_features', default=150, type=int, help="Number of hidden features in GRU.")
    parser.add_argument('--nb_features_final', default=10, type=int, help="Number of final features of the encoder.")
    parser.add_argument('--nb_clusters', default=15, type=int, help="Number of desired clusters. In case if we do not compute for a range of clusters.")
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--epoch_nb', default=150, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    args = parser.parse_args()

    #Start time
    start_time = time.time()
    run_name = "." + str(time.strftime("%Y-%m-%d_%H%M"))
    print(run_name)

    #Montpellier
    path_results_seg_series = os.path.expanduser(
        '~/Desktop/Results/Segmentation_outliers_upd_filled/Montpellier_SPOT5_graph_cut_series_2D/')
    seg_folder_series = "series_sigma_0.3_k_6_min_10_bands_3_threshold_int_0.4/"
    folder_encoded = "patch_9_feat_5.2019-09-03_1619_noise1_mean_std"
    path_results = path_results_seg_series + seg_folder_series + "Graph_coverage_filtered/"
    path_results_final = path_results + "alpha_" + str(alpha) + "_t1_" + str(t1) + "_t2_" + str(t2) + "_t3_" + str(t3) + "/"

    # We open BB file that contains synopses
    bb_final_list = np.load(path_results_final + "Graph_list_synopsys_alpha_" + str(alpha) + "_t1_" + str(t1) + "_t2_" + str(t2)+ "_t3_" + str(t3) + "_"+folder_encoded+".npy")
    for z in range(8):
        bb_final_list = np.c_[bb_final_list, np.full(len(bb_final_list), None)]

    folder_results = "Synopsys_padding_feat_" + str(args.nb_features) + "_lr_" + str(args.learning_rate) + run_name

    # Folder with the results
    path_results_NN = path_results_final + model + "_" + type + "/" + folder_results + "/"
    create_dir(path_results_NN)
    stats_file = path_results_NN + 'stats.txt'
    path_model = path_results_NN + 'model' + run_name + "/"
    create_dir(path_model)

    # We add new arguments to the parser
    print_stats(stats_file, folder_encoded, print_to_console=False)
    print_stats(stats_file, str(args), print_to_console=False)
    parser.add_argument('--stats_file', default=stats_file)
    parser.add_argument('--path_results', default=path_results_NN)
    parser.add_argument('--path_model', default=path_model)
    parser.add_argument('--run_name', default=run_name)
    args = parser.parse_args()


    # We open segmentation rasters
    segm_array_list = []
    date_list = []
    image_name_segm_list = np.sort(
        list(filter(lambda f: (f.endswith(".TIF") and f.startswith("Segments_1D_20")), os.listdir(path_results))))
    nbr_images = len(image_name_segm_list)
    print(image_name_segm_list)
    for i in range(nbr_images):
        image_name_segm = image_name_segm_list[i]
        date = (re.search("_([0-9]*).TIF", image_name_segm)).group(1)
        print(date)
        date_list.append(date)
        image_array_seg, H, W, geo, proj, bands_nb = open_tiff(path_results,
                                                               os.path.splitext(image_name_segm)[0])
        segm_array_list.append(image_array_seg)
    nbr_images = np.max(bb_final_list[:, 0])+1

    # we get synopses
    if type=="mean":
        segments = bb_final_list[:, 8]
    else:
        segments = bb_final_list[:, 7]
    feat_nb = len(segments[0][0])


    # We zero-pad all the sequences, so they have the same length over the dataset (equal to dataset length). See the article
    segments_padding = np.zeros((len(bb_final_list), nbr_images, feat_nb))
    for s in range(len(segments)):
        segments_padding[s][:len(segments[s])] = segments[s]
    print(segments_padding.shape)

    # We prepare the training dataset
    image = ImageDataset(segments_padding, args.patch_size, 0, np.arange(len(segments)), feat_nb)  # we create a dataset with tensor patches
    loader_pretrain = dsloader(image, gpu, args.batch_size, shuffle=True)
    loader_enc = dsloader(image, gpu, batch_size=1000, shuffle=False)

    # We initialize the model
    encoder = Encoder(feat_nb, args.nb_features, args.nb_features_final) # On CPU
    decoder = Decoder(feat_nb, args.nb_features, args.nb_features_final) # On CPU
    if gpu:
        encoder = encoder.cuda()  # On GPU
        decoder = decoder.cuda()  # On GPU


    print_stats(stats_file, str(encoder), print_to_console=False)

    # We pretrain the model
    pretrain_lstm(args.epoch_nb, encoder, decoder, loader_pretrain, args)
    # pretrain_lstm(0, encoder, decoder, loader_pretrain, args)

    end_time = time.clock()
    total_time_pretraining = end_time - start_time
    total_time_pretraining = str(datetime.timedelta(seconds=total_time_pretraining))
    print_stats(args.stats_file, "Total time pretraining =" + str(total_time_pretraining) + "\n")

    # We start encoding and clustering
    start_time = time.time()


    bb_final_list_flipped = np.flip(np.copy(bb_final_list), axis=0)
    print_stats(stats_file, 'Initializing clusters...')
    cl_nb = list(range(5,51,5))

    labels_list, labels_h_list, hidden_array = encode_lstm(encoder, W, loader_enc, cl_nb)
    for c in range(len(cl_nb)):
        feat_cl = cl_nb[c]
        print(feat_cl)
        labels, labels_h = labels_list[c], labels_h_list[c]
        labels, labels_h = np.flip(labels, axis=0), np.flip(labels_h, axis=0)
        new_labels = np.zeros((H * W))
        new_labels_h = np.zeros((H * W))

        # We optionally write clustering results to the BB list
        for l in range(len(labels)):
            if feat_cl == 15:
                bb_final_list_flipped[l, 9] = labels_h[l] + 1
            if feat_cl == 20:
                bb_final_list_flipped[l, 10] = labels_h[l] + 1
            if feat_cl == 25:
                bb_final_list_flipped[l, 11] = labels_h[l] + 1
            if feat_cl == 30:
                bb_final_list_flipped[l, 12] = labels_h[l] + 1
            if feat_cl == 35:
                bb_final_list_flipped[l, 13] = labels_h[l] + 1
            if feat_cl == 40:
                bb_final_list_flipped[l, 14] = labels_h[l] + 1
            if feat_cl == 45:
                bb_final_list_flipped[l, 15] = labels_h[l] + 1
            if feat_cl == 50:
                bb_final_list_flipped[l, 16] = labels_h[l] + 1
            img, ind = bb_final_list_flipped[l, 0:2]
            coverage_ind = np.where(segm_array_list[img].flatten() == ind)[0]
            new_labels[coverage_ind] = labels[l] + 1
            new_labels_h[coverage_ind] = labels_h[l] + 1

        ds = create_tiff(1, args.path_results + "Kmeans_initial_clusters_"+str(feat_cl)+".TIF", W, H, gdal.GDT_Int16, np.reshape(new_labels, (H, W)), geo,
                         proj)
        ds.GetRasterBand(1).SetNoDataValue(0)
        vectorize_tiff(path_results, "Kmeans_initial_clusters_"+str(feat_cl), ds)
        ds = None
        ds = create_tiff(1, args.path_results + "Hierarchical_initial_clusters_"+str(feat_cl)+".TIF", W, H, gdal.GDT_Int16, np.reshape(new_labels_h, (H, W)), geo,
                         proj)
        ds.GetRasterBand(1).SetNoDataValue(0)
        vectorize_tiff(path_results, "Hierarchical_initial_clusters_"+str(feat_cl), ds)
        ds = None


    np.save(
        args.path_results + "Graph_list_synopsys_clusters_alpha_" + str(alpha) + "_t1_" + str(t1) + "_t2_" + str(t2) + "_t3_" + str(t3),
        np.flip(np.copy(bb_final_list_flipped), axis=0))

    end_time = time.time()
    total_time_pretraining = end_time - start_time
    total_time_pretraining = str(datetime.timedelta(seconds=total_time_pretraining))
    print_stats(stats_file, "Total time encoding =" + str(total_time_pretraining) + "\n")

if __name__ == '__main__':
    main()