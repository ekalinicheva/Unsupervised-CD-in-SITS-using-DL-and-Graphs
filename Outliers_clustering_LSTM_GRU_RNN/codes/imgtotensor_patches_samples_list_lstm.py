import numpy as np
from torch.utils.data.dataset import Dataset
np.set_printoptions(threshold=np.nan)
import pandas as pd
import torch


def img_to_patches_to_tensor(index, image, patch_size, feat_nb):
    id = index
    if patch_size == 1:
        if feat_nb is None:
            patch = image[id].flatten()
        else:
            patch = image[id]
    return patch



def toTensor(pic):
    if isinstance(pic, np.ndarray):
        pic = pic.astype(float)
        img = torch.from_numpy(pic).float()
        return img


class ImageDataset(Dataset):

    def __init__(self, image, patch_size, image_id, samples_list, feat_nb=None):
        self.patch_size = patch_size
        # I create pandas dataframe to be able to iterate through indices when loading sequences
        self.tmp_df = pd.DataFrame(
            {'patch_idx': range(len(samples_list)), 'patch_id': (list(samples_list))})
        self.image = image
        self.image_id = image_id
        self.X = self.tmp_df['patch_id']
        self.id = self.tmp_df['patch_idx']
        self.feat_nb = feat_nb


    def X(self):
        return self.X


    def __getitem__(self, index):
        img = img_to_patches_to_tensor(self.X[index], self.image, self.patch_size, self.feat_nb)
        img_tensor = toTensor(img)
        return img_tensor, self.image_id, self.id[index]

    def __len__(self):
        return len(self.X.index)
