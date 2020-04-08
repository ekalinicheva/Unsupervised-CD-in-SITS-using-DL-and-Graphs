from torch.utils.data import DataLoader


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