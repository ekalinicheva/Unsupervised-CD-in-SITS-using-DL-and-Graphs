import os, time, re, datetime
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from osgeo import gdal, ogr
from codes.image_processing import extend, open_tiff, create_tiff

from codes.imgtotensor_patches_samples_two import ImageDataset
from codes.loader import dsloader
from codes.check_gpu import on_gpu
from random import sample
from codes.otsu_avg import otsu


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


gpu = on_gpu()  # We check if we can work on GPU, GPU RAM should be >4Gb
print ("ON GPU is "+str(gpu))


start_time = time.time()
run_name = str(time.strftime(".%Y-%m-%d_%H%M"))
print (run_name)



# Here we choose the type of the output of the encoding model
# If there is no pooling, we get one output variable (encoded features)
# If there is pooling, we get two outputs (encoded features and pooling indicies)
list_types_of_return = ["no_pooling", "pooling"]


#Parameters
type_of_return = list_types_of_return[0]
patch_size = 7
bands_to_keep = 3
epoch_nb = 1
batch_size = 150
learning_rate = 0.00005
sampleTrue = False
maskTrue = True     # if we apply mask to computa changes to particular areas
satellite = "S2"
city_name = "Rostov"



# Here we give the parameters of the pre-trained model, fo we will fine-tune it later
reference_model = "2019-11-12_1705"      # model run time
epoch_model = 3     # epoch we want to use
loss_model = 2.29e-05       # loss value of this epoch
path_models = os.path.expanduser('~/Desktop/Results/RESULTS_CHANGE_DETECTION/NN_Rostov_S2_all_images_model_pretrained/')    # global path for the results for the chosen dataset
folder_pretrained_results = "All_images_ep_3_patch_7.2019-11-12_1705/"  # folder with all the results for the chosen model


# Path do the dataset with images
# path_datasets = os.path.expanduser('~/Desktop/Datasets/Montpellier_SPOT5_Clipped_relatively_normalized_03_02_mask_vegetation_water_mode_parts_2004_no_DOS1_/')
path_datasets = os.path.expanduser('~/Desktop/Datasets/Rostov_S2_Concatenated_Clipped_norm_without_2016_filtered/')

if maskTrue:
    path_mask = os.path.expanduser('~/Desktop/Datasets/Rostov_S2_Defected_clipped/')

model_folder = folder_pretrained_results + "model."+reference_model


# We open extended images to calculate the min and max of the dataset to normalize the image values from 0 to 1
images_list = np.sort(os.listdir(path_datasets))
path_list = []
list_image_extended = []
list_image_date = []
new_images_list = []
if maskTrue:
    list_image_extended_temp = []
    list_image_mask = []
for image_name_with_extention in images_list:
    # if image_name_with_extention.endswith(".TIF") and image_name_with_extention.startswith(
    #         city_name) and not image_name_with_extention.endswith("band.TIF"):
    if image_name_with_extention.endswith(".TIF") and not image_name_with_extention.endswith("band.TIF"):
        new_images_list.append(image_name_with_extention)
        img_path = path_datasets + image_name_with_extention
        path_list.append(img_path)
        image_array, H, W, geo, proj, bands_nb = open_tiff(path_datasets,
                                                           os.path.splitext(image_name_with_extention)[0])
        if satellite == "SPOT5":
            image_date = (re.search("XS_([0-9]*)_", image_name_with_extention)).group(1)
        if satellite == "S2":
            image_date = (re.search("S2_([0-9]*).", image_name_with_extention)).group(1)
        print (image_date)
        if bands_to_keep == 3:
            if satellite == "SPOT5":
                image_array = np.delete(image_array, 3, axis=0)
                bands_nb = 3
            if satellite == "S2":
                image_array = np.delete(image_array, 0, axis=0)
                bands_nb = 3
        if satellite == "S2":
            for b in range(len(image_array)):
                image_array[b][image_array[b] > 4096] = np.max(image_array[b][image_array[b] <= 4096])
        if maskTrue:
            # in our case we have only one mask for the whole SITS, but in case you have separate masks for each couple of images, it can be useful
            path_mask = "/media/user/DATA/Results/Segmentation_new_outliers_filled/"
            mask_name = "Water_city_mask"
            mask_array, _, _, _, _, _ = open_tiff(path_mask, mask_name)
            mask_array[mask_array > 0] = 10
            image_array_mask = mask_array
            list_image_mask.append(mask_array)
        image_extended = extend(image_array, patch_size)
        list_image_extended.append(image_extended)
        list_image_date.append(image_date)
list_image_date = np.asarray(list_image_date, dtype=int)
sort_ind = np.argsort(list_image_date)  # we arrange images by date of acquisition
list_image_extended = np.asarray(list_image_extended, dtype=float)[sort_ind]
nbr_images = len(list_image_extended)
list_image_date = np.asarray(list_image_date, dtype=str)
new_images_list = np.asarray(new_images_list)[sort_ind]
if maskTrue:
    list_image_mask = np.asarray(list_image_mask)[sort_ind]



# We calculate min and max of dataset to perform image rescaling from 0 to 1 later
list_norm = []
for band in range(len(list_image_extended[0])):
    all_images_band = list_image_extended[:, band, :, :].flatten()
    min = np.min(all_images_band)
    max = np.max(all_images_band)
    list_norm.append([min, max])



for im in range(0, len(list_image_extended)-1):
    # We open the pre-trained AE model for every couple of images
    try:
        ae_model_name = "ae-model_ep_" + str(epoch_model) + "_loss_" + str(loss_model) + "." + reference_model
        encoder12, decoder12 = torch.load(path_models + model_folder + "/" + ae_model_name + ".pkl")
        encoder21, decoder21 = torch.load(path_models + model_folder + "/" + ae_model_name + ".pkl")
    except:
        encoder_model_name = "encoder-model_ep_" + str(epoch_model) + "_loss_" + str(loss_model) + "." + reference_model
        decoder_model_name = "decoder-model_ep_" + str(epoch_model) + "_loss_" + str(loss_model) + "." + reference_model
        encoder12 = torch.load(path_models + model_folder + "/" + encoder_model_name + ".pkl")
        decoder12 = torch.load(path_models + model_folder + "/" + decoder_model_name + ".pkl")
        encoder21 = torch.load(path_models + model_folder + "/" + encoder_model_name + ".pkl")
        decoder21 = torch.load(path_models + model_folder + "/" + decoder_model_name + ".pkl")

    if gpu:
        encoder12 = encoder12.cuda()  # On GPU
        decoder12 = decoder12.cuda()  # On GPU
        encoder21 = encoder21.cuda()  # On GPU
        decoder21 = decoder21.cuda()  # On GPU


    # Here we have to choose the images timestamps (whether it is t and t+1 or t and t+2)
    image_name1 = os.path.splitext(new_images_list[im])[0]
    image_date1 = list_image_date[im]

    image_name2 = os.path.splitext(new_images_list[im+1])[0]
    image_date2 = list_image_date[im+1]

    print(image_date1, image_date2)


    folder_results = folder_pretrained_results +"t_t1/" + "Joint_AE_"+image_date1 + "_" +image_date2 + "_ep_" + str(epoch_nb) + "_patch_" + str(patch_size) + run_name
    path_results = os.path.expanduser('~/Desktop/Results/RESULTS_CHANGE_DETECTION/NN_'+city_name+'_'+str(satellite)+'_all_images_model_pretrained/') + folder_results +"/"


    # we check if this changes are already computed
    if os.path.exists(path_results):
        continue


    create_dir(path_results)

    path_model = path_results + 'model'+run_name+"/" #we will save the encoder/decoder models here
    create_dir(path_model)


    driver_tiff = gdal.GetDriverByName("GTiff")
    driver_shp = ogr.GetDriverByName("ESRI Shapefile")


    image_array1, H, W, geo, proj, bands_nb = open_tiff(path_datasets, image_name1)
    image_array2, H, W, geo, proj, bands_nb = open_tiff(path_datasets, image_name2)
    if bands_to_keep == 3:
        if satellite == "SPOT5":
            image_array1 = np.delete(image_array1, 3, axis=0)
            image_array2 = np.delete(image_array2, 3, axis=0)
            bands_nb = 3
        if satellite == "S2":
            image_array1 = np.delete(image_array1, 0, axis=0)
            image_array2 = np.delete(image_array2, 0, axis=0)
            bands_nb = 3
        if satellite == "S2":
            for b in range(len(image_array)):
                image_array[b][image_array[b]>4096] = np.max(image_array[b][image_array[b]<=4096])



    image_extended1 = extend(image_array1, patch_size).astype(float)
    image_extended2 = extend(image_array2, patch_size).astype(float)



    for band in range(len(image_extended1)):
        image_extended1[band] = (image_extended1[band] - list_norm[band][0])/(list_norm[band][1]-list_norm[band][0])
        image_extended2[band] = (image_extended2[band] - list_norm[band][0])/(list_norm[band][1]-list_norm[band][0])

    if sampleTrue:
        nbr_patches_per_image = int(H * W / 2)
        samples_list = np.sort(sample(range(H * W), nbr_patches_per_image))
        image = ImageDataset(image_extended1, image_extended2, patch_size,
                             samples_list)  # we create a dataset with tensor patches
        loader = dsloader(image, gpu, batch_size=batch_size, shuffle=True)

    elif maskTrue:
        mask = np.where((list_image_mask[im] + list_image_mask[im+1]).flatten() == 0)[0]
        image = ImageDataset(image_extended1, image_extended2, patch_size,
                             mask)  # we create a dataset with tensor patches
        loader = dsloader(image, gpu, batch_size, shuffle=True)

    else:
        image = ImageDataset(image_extended1, image_extended2, patch_size,
                             list(range(H * W)))  # we create a dataset with tensor patches
        loader = dsloader(image, gpu, batch_size=batch_size, shuffle=True)

    image_enc = ImageDataset(image_extended1, image_extended2, patch_size,
                             list(range(H * W)))  # we create a dataset with tensor patches


    #we save everything to stats file
    with open(path_results+"stats.txt", 'a') as f:
        f.write("Relu activations for every layer except the last one. The last one is not activated" + "\n")
        f.write("patch_size=" + str(patch_size) + "\n")
        f.write("epoch_nb=" + str(epoch_nb) + "\n")
        f.write("batch_size=" + str(batch_size) + "\n")
        f.write("learning_rate=" + str(learning_rate) + "\n")
        f.write("sample=" + str(sampleTrue) + "\n")
    f.close()


    optimizer = torch.optim.Adam((list(encoder12.parameters()) + list(decoder12.parameters()) + list(encoder21.parameters()) + list(decoder21.parameters())), lr=learning_rate)
    criterion = nn.MSELoss()    #loss function

    with open(path_results+"stats.txt", 'a') as f:
        f.write(str(encoder12) + "\n")
    f.close()

    start_time = time.time()


    # function to fine-tune the model
    epoch_loss_list = []
    epoch_loss12_list = []
    epoch_loss21_list = []
    def train(epoch):
        # we have separate encoder/decoder models for both AE
        encoder12.train() #we swich to train mode (by default)
        decoder12.train()
        encoder21.train() #we swich to train mode (by default)
        decoder21.train()
        total_loss = 0
        total_loss12 = 0
        total_loss21 = 0
        for batch_idx, (data1, data2, _) in enumerate(loader):  #we load batches from model
            if gpu:
                data1 = data1.cuda()
                data2 = data2.cuda()
            # if/else allow us to manipulate different types of models with different input/output
            if type_of_return == list_types_of_return[0]:
                encoded12 = encoder12(Variable(data1))
                encoded21 = encoder21(Variable(data2))
                decoded12 = decoder12(encoded12)
                decoded21 = decoder21(encoded21)
            if type_of_return == list_types_of_return[1]:
                encoded12, id1 = encoder12(Variable(data1))
                decoded12 = decoder12(encoded12, id1)
                encoded21, id1 = encoder21(Variable(data2))
                decoded21 = decoder21(encoded21, id1)
            encoded21_copy = encoded21.clone().detach()
            encoded12_copy = encoded12.clone().detach()

            loss11 = criterion(encoded12, (encoded12_copy+encoded21_copy)/2)
            loss22 = criterion(encoded21, (encoded12_copy+encoded21_copy)/2)
            total_loss += loss11.item()                 #total loss for the epoch
            loss12 = criterion(decoded12, Variable(data2))
            total_loss12 += loss12.item()
            loss21 = criterion(decoded21, Variable(data1))
            total_loss21 += loss21.item()
            optimizer.zero_grad()               #everything to optimize the model
            loss11.backward(retain_graph=True)
            loss22.backward(retain_graph=True)
            loss12.backward(retain_graph=True)
            loss21.backward()
            optimizer.step()
            if (batch_idx+1) % 200 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.7f}\tLoss12: {:.7f}\tLoss21: {:.7f}'.format(
                    (epoch+1), (batch_idx+1) * batch_size, len(loader.dataset),
                    100. * (batch_idx+1) / len(loader), loss11.item(), loss12.item(), loss21.item()))
        epoch_loss = total_loss / len(loader)   #avg loss per epoch
        epoch_loss_list.append(epoch_loss)
        epoch_loss12 = total_loss12 / len(loader)   #avg loss per epoch
        epoch_loss12_list.append(epoch_loss12)
        epoch_loss21 = total_loss21 / len(loader)   #avg loss per epoch
        epoch_loss21_list.append(epoch_loss21)
        epoch_stats = "Epoch {} Complete: Avg. Loss: {:.7f}\tAvg. Loss12: {:.7f}\tAvg. Loss21: {:.7f}".format(epoch + 1, epoch_loss, epoch_loss12, epoch_loss21)
        print(epoch_stats)
        with open(path_results + "stats.txt", 'a') as f:
            f.write(epoch_stats+"\n")
        f.close()

        #we save all the models to choose the best afterwards
        torch.save([encoder12, decoder12], (path_model+'ae12-model_ep_'+str(epoch+1)+"_loss_"+str(round(epoch_loss12, 7))+run_name+'.pkl') )
        torch.save([encoder21, decoder21], (path_model+'ae21-model_ep_'+str(epoch+1)+"_loss_"+str(round(epoch_loss21, 7))+run_name+'.pkl') )


    for epoch in range(epoch_nb):
        if epoch==3:
            learning_rate = 0.00005
            optimizer = torch.optim.Adam((list(encoder12.parameters()) + list(decoder12.parameters()) + list(encoder21.parameters()) + list(decoder21.parameters())), lr=learning_rate)

            with open(path_results + "stats.txt", 'a') as f:
                f.write("new_learning_rate=" + str(learning_rate) + "\n")
            f.close()
        train(epoch)


    # we compute fine-tuning time
    end_time = time.time()
    total_time_learning = end_time - start_time
    total_time_learning = str(datetime.timedelta(seconds=total_time_learning))
    print("Total time fine-tuning =", total_time_learning)

    with open(path_results+"stats.txt", 'a') as f:
        f.write("Total time fine-tuning =" + str(total_time_learning) + "\n"+"\n")
    f.close()

    # we performe feature translation and compute the reconstruction loss to detect the change areas
    # we create data loader for the fine-tuning
    loader = dsloader(image_enc, gpu, batch_size=2000, shuffle=False)

    criterion = nn.MSELoss(reduce=False)

    for best_epoch in range(1, epoch_nb+1):
        # we choose the best epoch for reconstruction (in the article we fine-tune for only one epoch, so it does not change anything in this particular case)
        best_epoch_loss12 = epoch_loss12_list[best_epoch-1]
        best_epoch_loss21 = epoch_loss21_list[best_epoch-1]
        # we load this model
        best_encoder12, best_decoder12 = torch.load(path_model + 'ae12-model_ep_' + str(best_epoch) + "_loss_" + str(round(best_epoch_loss12, 7)) + run_name + '.pkl')
        best_encoder21, best_decoder21 = torch.load(path_model + 'ae21-model_ep_' + str(best_epoch) + "_loss_" + str(round(best_epoch_loss21, 7)) + run_name + '.pkl')

        if gpu:
            best_encoder12 = best_encoder12.cuda()  # On GPU
            best_encoder21 = best_encoder21.cuda()  # On GPU
            best_decoder12 = best_decoder12.cuda()  # On GPU
            best_decoder21 = best_decoder21.cuda()  # On GPU

        name_results12 = "From_" + image_date1 + "_to_" + image_date2 + "_ep_" + str(best_epoch)
        name_results21 = "From_" + image_date2 + "_to_" + image_date1 + "_ep_" + str(best_epoch)


        new_coordinates_loss_mean12 = []
        new_coordinates_loss_mean21 = []

        best_encoder12.eval()
        best_decoder12.eval()
        best_encoder21.eval()
        best_decoder21.eval()
        for batch_idx, (data1, data2, _) in enumerate(loader):  # we load batches from model
            if gpu:
                data1 = data1.cuda()
                data2 = data2.cuda()
                #index = index.cuda(async=True)
            if type_of_return == list_types_of_return[0]:
                encoded12 = best_encoder12(Variable(data1))
                decoded12 = best_decoder12(encoded12)
                encoded21 = best_encoder21(Variable(data2))
                decoded21 = best_decoder21(encoded21)
            if type_of_return == list_types_of_return[1]:
                encoded12, id1 = best_encoder12(Variable(data1))
                decoded12 = best_decoder12(encoded12, id1)
                encoded21, id1 = best_encoder21(Variable(data2))
                decoded21 = best_decoder21(encoded21, id1)

            loss12 = criterion(decoded12, Variable(data2))
            loss21 = criterion(decoded21, Variable(data1))

            loss_mean12 = loss12.view(-1, bands_nb, patch_size*patch_size).mean(2).mean(1)
            loss_mean21 = loss21.view(-1, bands_nb, patch_size*patch_size).mean(2).mean(1)

            if gpu:
                new_coordinates_loss_batch_mean12 = loss_mean12.data.cpu().numpy()
                new_coordinates_batch12 = decoded12.data.cpu().numpy()
                new_coordinates_loss_batch_mean21 = loss_mean21.data.cpu().numpy()
                new_coordinates_batch21 = decoded21.data.cpu().numpy()
            else:
                new_coordinates_loss_batch_mean12 = loss_mean12.data.numpy()
                new_coordinates_batch12 = decoded12.data.numpy()
                new_coordinates_loss_batch_mean21 = loss_mean21.data.numpy()
                new_coordinates_batch21 = decoded21.data.numpy()


            new_coordinates_loss_mean12.append(list(new_coordinates_loss_batch_mean12))
            new_coordinates_loss_mean21.append(list(new_coordinates_loss_batch_mean21))

            if (batch_idx + 1) % 200 == 0:
                print('Encoding : [{}/{} ({:.0f}%)]'.format(
                    (batch_idx + 1) * batch_size, len(loader.dataset),
                    100. * (batch_idx + 1) / len(loader)))

        print(len(new_coordinates_loss_mean12))
        new_coordinates_loss_mean12 = np.asarray(new_coordinates_loss_mean12).flatten()
        new_coordinates_loss_mean21 = np.asarray(new_coordinates_loss_mean21).flatten()

        if maskTrue:
            defected_mask = np.setdiff1d(np.arange(H*W), mask)
            new_coordinates_loss_mean12[defected_mask] = 0
            new_coordinates_loss_mean21[defected_mask] = 0
        else:
            defected_mask = None
            mask = None

        # We create a loss image in new coordinate system
        image_array_tr_mean = np.reshape(new_coordinates_loss_mean12, (H, W))
        loss_image_name_mean = name_results12
        loss_image_mean = path_results + "Loss_mean_" + loss_image_name_mean + ".TIF"
        dst_ds = create_tiff(1, loss_image_mean, W, H, gdal.GDT_Float32, image_array_tr_mean, geo, proj)
        dst_ds = None
        image_array_loss1 = image_array_tr_mean


        # We create a loss image in new coordinate system
        image_array_tr_mean = np.reshape(new_coordinates_loss_mean21, (H, W))
        loss_image_name_mean = name_results21
        loss_image_mean = path_results + "Loss_mean_" + loss_image_name_mean + ".TIF"
        dst_ds = create_tiff(1, loss_image_mean, W, H, gdal.GDT_Float32, image_array_tr_mean, geo, proj)
        dst_ds = None
        image_array_loss2 = image_array_tr_mean

        # we compute otsu thresholding for 2 different threshold paratemers 0.095 and 0.098
        # the parameter "changes" is used only when we have a GT change map for this couple of images and we want to compute accuracy statistics
        otsu(image_array_loss1, image_array_loss2, H, W, geo, proj, path_results, image_date1 + "_to_" + image_date2, threshold=0.995, changes=None, mask=mask)
        otsu(image_array_loss1, image_array_loss2, H, W, geo, proj, path_results, image_date1 + "_to_" + image_date2, threshold=0.998, changes=None, mask=mask)