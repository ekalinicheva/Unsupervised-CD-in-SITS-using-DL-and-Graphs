from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def dsloader(image, gpu, batch_size, shuffle):
    if gpu:
        loader = DataLoader(image,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=1,  # 1 for CUDA
                            pin_memory=True,  # CUDA only
                            drop_last=False
                            )
    else:
        loader = DataLoader(image,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=0,  # 1 for CUDA
                            drop_last=False
                            )
    return loader


def random_dsloader(image, gpu, batch_size, indicies):
    if gpu:
        loader = DataLoader(image,
                            batch_size=batch_size,
                            sampler=SubsetRandomSampler(indicies),
                            num_workers=1,  # 1 for CUDA
                            pin_memory=True,  # CUDA only
                            drop_last=True
                            )
    else:
        loader = DataLoader(image,
                            batch_size=batch_size,
                            sampler=SubsetRandomSampler(indicies),
                            num_workers=0,  # 1 for CUDA
                            drop_last=True
                            )
    return loader