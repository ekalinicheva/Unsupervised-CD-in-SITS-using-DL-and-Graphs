import os, time, re, datetime
import numpy as np
from random import sample
import torch
from torch import nn
from torch.autograd import Variable


from models.ae_fully_convolutional_4conv_l2 import Encoder, Decoder     #we import the selected ae model
from codes.imgtotensor_patches_samples_list import ImageDataset
from codes.image_processing import extend, open_tiff
from codes.loader import dsloader
from codes.check_gpu import on_gpu
from codes.plot_loss import plotting



def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


gpu = on_gpu()
print("ON GPU is "+str(gpu))


# Here we choose the type of the output of the encoding model
# If there is no pooling, we get one output variable (encoded features)
# If there is pooling, we get two outputs (encoded features and pooling indicies)
list_types_of_return = ["no_pooling", "pooling"]


#Parameters
type_of_return = list_types_of_return[0]
patch_size = 7
bands_to_keep = 3
epoch_nb = 3
batch_size = 150
learning_rate = 0.0005
satellite = "SPOT5"
city = "Montpellier"
maskTrue = False


start_time = time.clock()
run_name = "."+str(time.strftime("%Y-%m-%d_%H%M"))
print (run_name)

# path to the folder with SITS images
path_datasets = os.path.expanduser('~/Desktop/Datasets/Montpellier_SPOT5_Clipped_relatively_normalized_03_02_mask_vegetation_water_mode_parts_2004_no_DOS1_/')
if maskTrue:
    path_mask = os.path.expanduser('~/Desktop/Datasets/Rostov_S2_Defected_clipped/')


# Path to the results
folder_results = "All_images_ep_" + str(epoch_nb) + "_patch_" + str(patch_size) + run_name
path_results = os.path.expanduser('~/Desktop/Results/RESULTS_CHANGE_DETECTION/NN_'+city+'_'+str(satellite)+'_all_images_model_pretrained/') + folder_results + "/"
create_dir(path_results)
path_model = path_results + 'model'+run_name+"/"
create_dir(path_model)


if maskTrue:    # in case we apply mask to extract patches only from particular area
    list_image_extended_temp = []
    list_image_mask = []

#We open the images and perform their extention (we mirror the border rows and columns) for the correct patch extraction
images_list = os.listdir(path_datasets)
path_list = []
list_image_extended = []
list_image_date = []
for image_name_with_extention in np.sort(images_list):
    if image_name_with_extention.endswith(".TIF") and not image_name_with_extention.endswith("band.TIF"):
        img_path = path_datasets + image_name_with_extention
        path_list.append(img_path)
        image_array, H, W, geo, proj, bands_nb = open_tiff(path_datasets, os.path.splitext(image_name_with_extention)[0])
        if satellite == "SPOT5":
            image_date = (re.search("XS_([0-9]*)_", image_name_with_extention)).group(1)
        if satellite == "S2":
            image_date = (re.search("S2_([0-9]*).", image_name_with_extention)).group(1)
        print(image_date)
        if bands_to_keep == 3:
            if satellite == "SPOT5":
                image_array = np.delete(image_array, 3, axis=0)
                bands_nb = 3
            if satellite == "S2":
                image_array = np.delete(image_array, 0, axis=0)
                bands_nb = 3
        if satellite == "S2":   # we deal with saturated pixels for s2
            for b in range(len(image_array)):
                image_array[b][image_array[b] > 4096] = np.max(image_array[b][image_array[b]<=4096])
        if maskTrue:    # if there are masks, we open them
            mask_name = "Defected_Rostov_S2_" + str(image_date)
            image_array_mask, _, _, _, _, _ = open_tiff(path_mask, mask_name)
            image_array_mask[image_array_mask < 3] = 0
            image_array_mask[image_array_mask >= 3] = 1
        image_extended = extend(image_array, patch_size)
        list_image_extended.append(image_extended)
        list_image_date.append(image_date)
list_image_date = np.asarray(list_image_date, dtype=int)
sort_ind = np.argsort(list_image_date)  # we arrange images by date of acquisition
list_image_extended = np.asarray(list_image_extended, dtype=float)[sort_ind]
list_image_date = np.asarray(list_image_date)[sort_ind]
if maskTrue:
    list_image_mask = np.asarray(list_image_mask)[sort_ind]


# we re-scale images between 0 and 1
list_norm = []
for band in range(len(list_image_extended[0])):
    all_images_band = list_image_extended[:, band, :, :].flatten()
    min = np.min(all_images_band)
    max = np.max(all_images_band)
    list_norm.append([min, max])

for i in range(len(list_image_extended)):
    for band in range(len(list_image_extended[0])):
        list_image_extended[i][band] = (list_image_extended[i][band]-list_norm[band][0])/(list_norm[band][1]-list_norm[band][0])



# we construct training dataset with selected patches
image = None
nbr_patches_per_image = int(H*W/len(list_image_extended))
for ii in range(len(list_image_extended)):
    if maskTrue:
        mask = np.where(list_image_mask[ii].flatten()==1)[0]
        samples_list = np.sort(sample(list(mask), nbr_patches_per_image))
    else:
        samples_list = np.sort(sample(range(H * W), nbr_patches_per_image))
    if image is None:
        image = ImageDataset(list_image_extended[ii], patch_size, ii,
                             samples_list)  # we create a dataset with tensor patches
    else:
        image2 = ImageDataset(list_image_extended[ii], patch_size, ii,
                              samples_list)  # we create a dataset with tensor patches
        image = torch.utils.data.ConcatDataset([image, image2])

loader = dsloader(image, gpu, batch_size, shuffle=True)
image = None
list_image_extended = None



with open(path_results+"stats.txt", 'a') as f:
    f.write("patch_size=" + str(patch_size) + "\n")
    f.write("epoch_nb=" + str(epoch_nb) + "\n")
    f.write("batch_size=" + str(batch_size) + "\n")
    f.write("learning_rate=" + str(learning_rate) + "\n")
    f.write("bands_to_keep= " + str(bands_to_keep) + "\n")
    f.write("Nbr patches per image " + str(nbr_patches_per_image) + "\n")
f.close()

# we create models
encoder = Encoder(bands_nb, patch_size) # On CPU
decoder = Decoder(bands_nb, patch_size) # On CPU
if gpu:
    encoder = encoder.to('cuda:0')  # On GPU
    decoder = decoder.to('cuda:0')  # On GPU


optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

with open(path_results+"stats.txt", 'a') as f:
    f.write(str(encoder) + "\n")
f.close()


# Start time to compute overall training time at the end
start_time = time.time()


# We pretrain the model
epoch_loss_list = []
def train(epoch):
    encoder.train()
    decoder.train()
    total_loss = 0
    for batch_idx, (data, _, _) in enumerate(loader):
        if gpu:
            data = data.cuda()
        encoded = encoder(Variable(data))
        decoded = decoder(encoded)
        loss = criterion(decoded, Variable(data))
        total_loss += loss.item()
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()
        if (batch_idx+1) % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.7f}'.format(
                (epoch+1), (batch_idx+1) * batch_size, len(loader.dataset),
                100. * (batch_idx+1) / len(loader), loss.item()))
    epoch_loss = total_loss / len(loader)
    epoch_loss_list.append(epoch_loss)
    epoch_stats = "Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch + 1, epoch_loss)
    print(epoch_stats)
    with open(path_results + "stats.txt", 'a') as f:
        f.write(epoch_stats+"\n")
    f.close()

    # we save the model
    torch.save([encoder, decoder], (path_model+'ae-model_ep_'+str(epoch+1)+"_loss_"+str(round(epoch_loss, 7))+run_name+'.pkl') )

    if (epoch+1) % 5 == 0:
        plotting(epoch+1, epoch_loss_list, path_results)


for epoch in range(epoch_nb):
    train(epoch)


best_epoch = np.argmin(np.asarray(epoch_loss_list))+1
best_epoch_loss = epoch_loss_list[best_epoch-1]


print("best epoch " + str(best_epoch))
print("best epoch loss " + str(best_epoch_loss))


# We compute training time
end_time = time.time()
total_time_learning = end_time - start_time
total_time_learning = str(datetime.timedelta(seconds=total_time_learning))
print("Total time learning =", total_time_learning)


with open(path_results+"stats.txt", 'a') as f:
    f.write("best epoch " + str(best_epoch) + "\n")
    f.write("best epoch loss " + str(best_epoch_loss) + "\n")
    f.write("Total time learning=" + str(total_time_learning) + "\n"+"\n")
f.close()