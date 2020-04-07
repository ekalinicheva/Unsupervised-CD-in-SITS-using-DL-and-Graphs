import os, time, re, datetime
import numpy as np
from random import sample
import argparse

import torch
from torch import nn
from torch.autograd import Variable

from models.autoencoder_patch9_6conv_3fc_l2_pooling_sigmoid_denoising import Encoder, Decoder
from codes.imgtotensor_patches_samples_list import ImageDataset
from codes.image_processing import open_tiff, extend, encode_image
from codes.loader import dsloader
from codes.stats_scripts import on_gpu, plotting, print_stats
from codes.plot_loss import plotting
from codes.pytorchtools import EarlyStopping

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def main():
    gpu = on_gpu()
    print ("ON GPU is "+str(gpu))


    start_time = time.time()
    run_name = "."+str(time.strftime("%Y-%m-%d_%H%M"))
    print (run_name)


    #Parameters
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--patch_size', default=9, type=int)
    parser.add_argument('--nb_features', default=5, type=int)
    parser.add_argument('--batch_size', default=150, type=int)
    parser.add_argument('--bands_to_keep', default=4, type=int)
    parser.add_argument('--epoch_nb', default=4, type=int)
    parser.add_argument('--satellite', default="SPOT5", type=str)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    args = parser.parse_args()

    # path with images to encode
    path_datasets = os.path.expanduser('~/Desktop/Datasets/Montpellier_SPOT5_Clipped_relatively_normalized_03_02_mask_vegetation_water_mode_parts_2004_no_DOS1_/')
    # folder and path to results
    folder_results = "All_images_ep_" + str(args.epoch_nb) + "_patch_" + str(args.patch_size) + "_batch_" + str(
        args.batch_size) + "_feat_" + str(args.nb_features) + "_lr_" + str(args.learning_rate) + run_name + "_noise1"
    path_results = os.path.expanduser('~/Desktop/Results/Encode_TS_noise/') + folder_results + "/"
    create_dir(path_results)
    # folder with AE models
    path_model = path_results + 'model'+run_name+"/"
    create_dir(path_model)
    # file with corresponding statistics
    stats_file = path_results+'stats.txt'

    print_stats(stats_file, str(args), print_to_console=False)
    parser.add_argument('--stats_file', default=stats_file)
    parser.add_argument('--path_results', default=path_results)
    parser.add_argument('--path_model', default=path_model)
    parser.add_argument('--run_name', default=run_name)
    args = parser.parse_args()


    #We open images and "extend" them (we mirror border rows and columns for correct patch extraction)
    images_list = os.listdir(path_datasets)
    path_list = []
    list_image_extended = []
    list_image_date = []
    for image_name_with_extention in images_list:
        if image_name_with_extention.endswith(".TIF") and not image_name_with_extention.endswith("band.TIF"):
            img_path = path_datasets + image_name_with_extention
            path_list.append(img_path)
            image_date = (re.search("_([0-9]*)_", image_name_with_extention)).group(1)
            # we open images
            image_array, H, W, geo, proj, bands_nb = open_tiff(path_datasets, os.path.splitext(image_name_with_extention)[0])
            # we delete swir bands for spot-5 or blue for Sentinel-2 if needed
            if args.bands_to_keep == 3:
                if args.satellite == "SPOT5":
                    image_array = np.delete(image_array, 3, axis=0)
                else:
                    image_array = np.delete(image_array, 0, axis=0)
            bands_nb = args.bands_to_keep
            # we extend image
            image_extended = extend(image_array, args.patch_size)
            list_image_extended.append(image_extended)
            list_image_date.append(image_date)
    sort_ind = np.argsort(list_image_date)  # we arrange images by date of acquisition
    list_image_extended = np.asarray(list_image_extended, dtype=float)[sort_ind]
    list_image_date = np.asarray(list_image_date)[sort_ind]

    # We normalize all the images with dataset mean and std
    list_norm = []
    for band in range(len(list_image_extended[0])):
        all_images_band = list_image_extended[:, band, :, :].flatten()
        min = np.min(all_images_band)
        max = np.max(all_images_band)
        mean = np.mean(all_images_band)
        std = np.std(all_images_band)
        list_norm.append([min, max, mean, std])

    for i in range(len(list_image_extended)):
        for band in range(len(list_image_extended[0])):
            list_image_extended[i][band] = (list_image_extended[i][band] - list_norm[band][2]) / list_norm[band][3]

    # We rescale from 0 to 1
    list_norm = []
    for band in range(len(list_image_extended[0])):
        all_images_band = list_image_extended[:, band, :, :].flatten()
        min = np.min(all_images_band)
        max = np.max(all_images_band)
        mean = np.mean(all_images_band)
        std = np.std(all_images_band)
        list_norm.append([min, max, mean, std])


    for i in range(len(list_image_extended)):
        for band in range(len(list_image_extended[0])):
            list_image_extended[i][band] = (list_image_extended[i][band]-list_norm[band][0])/(list_norm[band][1]-list_norm[band][0])

    # We recompute mean and std to use them for creation of Gaussian noise later
    list_norm = []
    for band in range(len(list_image_extended[0])):
        all_images_band = list_image_extended[:, band, :, :].flatten()
        mean = np.mean(all_images_band)
        std = np.std(all_images_band)
        list_norm.append([mean, std])


    # We create training and validation datasets with H*W/(SITS_length)*2 patches by concatenating datasets created for every image
    image = None
    image_valid = None
    nbr_patches_per_image = int(H*W/len(list_image_extended)*2)
    # nbr_patches_per_image = H * W
    for ii in range(len(list_image_extended)):
        samples_list = np.sort(sample(range(H * W), nbr_patches_per_image))
        samples_list_valid = np.sort(sample(range(H * W), int(nbr_patches_per_image/100)))
        if image is None:
            image = ImageDataset(list_image_extended[ii], args.patch_size, ii,
                                 samples_list)  # we create a dataset with tensor patches
            image_valid = ImageDataset(list_image_extended[ii], args.patch_size, ii,
                                 samples_list_valid)  # we create a dataset with tensor patches
        else:
            image2 = ImageDataset(list_image_extended[ii], args.patch_size, ii,
                                  samples_list)  # we create a dataset with tensor patches
            image = torch.utils.data.ConcatDataset([image, image2])
            image_valid2 = ImageDataset(list_image_extended[ii], args.patch_size, ii,
                                  samples_list_valid)  # we create a dataset with tensor patches
            image_valid = torch.utils.data.ConcatDataset([image_valid, image_valid2])

    loader = dsloader(image, gpu, args.batch_size, shuffle=True)
    loader_valid = dsloader(image_valid, gpu, H, shuffle=False)


    # we create AE model
    encoder = Encoder(bands_nb, args.patch_size, args.nb_features, np.asarray(list_norm)) # On CPU
    decoder = Decoder(bands_nb, args.patch_size, args.nb_features) # On CPU
    if gpu:
        encoder = encoder.cuda()  # On GPU
        decoder = decoder.cuda()  # On GPU

    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)

    criterion = nn.MSELoss()

    with open(path_results+"stats.txt", 'a') as f:
        f.write(str(encoder) + "\n")
    f.close()

    # Here we deploy early stopping algorithm taken from https://github.com/Bjarten/early-stopping-pytorch
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    early_stopping = EarlyStopping(patience=1, verbose=True)

    # we train the model
    def train(epoch):
        encoder.train()
        decoder.train()
        train_loss_total = 0
        for batch_idx, (data, _, _) in enumerate(loader):
            if gpu:
                data = data.cuda()
            encoded, id1 = encoder(Variable(data))
            decoded = decoder(encoded, id1)
            loss = criterion(decoded, Variable(data))
            train_loss_total += loss.item()
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()
            if (batch_idx+1) % 200 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    (epoch), (batch_idx+1) * args.batch_size, len(samples_list)*len(list_image_extended),
                    100. * (batch_idx+1) / len(loader), loss.item()))
        train_loss_total = train_loss_total / len(loader)
        epoch_stats = "Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, train_loss_total)
        print(epoch_stats)
        with open(path_results + "stats.txt", 'a') as f:
            f.write(epoch_stats+"\n")
        f.close()

        # We save trained model after each epoch. Optional
        torch.save([encoder, decoder], (path_model+'ae-model_ep_'+str(epoch+1)+"_loss_"+str(round(train_loss_total, 5))+run_name+'.pkl') )
        torch.save([encoder.state_dict(), decoder.state_dict()], (path_model+'ae-dict_ep_'+str(epoch+1)+"_loss_"+str(round(train_loss_total, 5))+run_name+'.pkl'))

        #Validation part
        valid_loss_total = 0
        encoder.eval()
        decoder.eval()  # prep model for evaluation
        for batch_idx, (data, _, _) in enumerate(loader_valid):
            if gpu:
                data = data.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            encoded, id1 = encoder(Variable(data))
            decoded = decoder(encoded, id1)
            # calculate the loss
            loss = criterion(decoded, Variable(data))
            # record validation loss
            valid_loss_total += loss.item()

        valid_loss_total = valid_loss_total/len(loader_valid)

        avg_train_losses.append(train_loss_total)
        avg_valid_losses.append(valid_loss_total)

        epoch_len = len(str(args.epoch_nb))

        print_msg = (f'[{epoch:>{epoch_len}}/{args.epoch_nb:>{epoch_len}}] ' +
                     f'train_loss: {train_loss_total:.5f} ' +
                     f'valid_loss: {valid_loss_total:.5f}')
        print(print_msg)

        # We plot the loss
        if (epoch+1) % 5 == 0:
            plotting(epoch, avg_train_losses, path_results)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss_total, [encoder, decoder])



    for epoch in range(1, args.epoch_nb+1):
        train(epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break


    end_time_learning = time.clock()
    total_time_learning = end_time_learning - start_time
    total_time_learning = str(datetime.timedelta(seconds=total_time_learning))
    print_stats(args.stats_file, "Total time pretraining =" + str(total_time_learning) + "\n")


    # We get the best model (here by default it is the last one)
    best_epoch = epoch
    best_epoch_loss = avg_train_losses[best_epoch-1]
    print("best epoch " + str(best_epoch))
    print("best epoch loss " + str(best_epoch_loss))
    best_encoder = encoder
    if gpu:
        best_encoder = best_encoder.cuda()  # On GPU

    #ENCODING PART
    for ii in range(len(list_image_extended)):
        print("Encoding " + str(list_image_date[ii]))
        samples_list = np.array(range(H * W))
        image_encode = ImageDataset(list_image_extended[ii], args.patch_size, ii,
                                 samples_list)  # we create a dataset with tensor patches

        loader_encode = dsloader(image_encode, gpu, H, shuffle=False)

        name_results = list_image_date[ii]
        encode_image(best_encoder, loader_encode, H*10, args.nb_features, gpu, H, W, geo, proj, name_results, path_results)


    end_time_encoding = time.time()
    total_time_encoding = end_time_encoding - end_time_learning
    total_time_encoding = str(datetime.timedelta(seconds=total_time_encoding))
    print_stats(args.stats_file, "Total time encoding =" + str(total_time_encoding) + "\n")


if __name__ == "__main__":
    main()