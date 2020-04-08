'''This code creates a dataset with patches corresponding to two images to perfrom image translation (change detection step'''
import numpy as np
from torch.utils.data.dataset import Dataset
np.set_printoptions(threshold=np.nan)
import pandas as pd
import torch


def img_to_patches_to_tensor(index, list_id, image, patch_size):
    id = list_id[index]
    i = int(id/(len(image[0][0])-(patch_size-1)))
    j = id % (len(image[0][0])-(patch_size-1))
    patch = image[:, i:(i+patch_size), j:(j+patch_size)]
    return patch



def toTensor(pic):
    if isinstance(pic, np.ndarray):
        pic = pic.astype(float)
        img = torch.from_numpy(pic).float()
        return img


class ImageDataset(Dataset):

    def __init__(self, image1,  image2, patch_size, samples_list):
        self.patch_size = patch_size
        # I create pandas dataframe to be able to iterate through indices when loading patches
        self.sample_len = len(samples_list)
        self.tmp_df = pd.DataFrame(
            {'patch_idx': list(range(self.sample_len)), 'patch_id': (list(samples_list))})
        self.image1 = image1
        self.image2 = image2
        self.X = self.tmp_df['patch_idx']
        self.id = self.tmp_df['patch_id']

    def X(self):
        return self.X

    def __getitem__(self, index):
        img1 = img_to_patches_to_tensor((self.X[index]), self.id, self.image1, self.patch_size)
        img_tensor1 = toTensor(img1)
        img2 = img_to_patches_to_tensor((self.X[index]), self.id, self.image2, self.patch_size)
        img_tensor2 = toTensor(img2)
        return img_tensor1, img_tensor2, self.X[index]

    def __len__(self):
        return len(self.X.index)
