from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, bands_nb, patch_size):
        super(Encoder, self).__init__()

        self.patch_size = patch_size

        # Stage 1
        # Feature extraction
        self.conv11 = nn.Conv2d(bands_nb, 32, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(32)
        self.conv12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(32)
        self.conv13 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(64)
        self.conv14 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm2d(64)


        # Stage 2
        # Compression
        self.linear21 = nn.Linear(64*self.patch_size*self.patch_size, 12*self.patch_size*self.patch_size)
        self.linear22 = nn.Linear(12*self.patch_size*self.patch_size, 2*self.patch_size*self.patch_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        # Stage 1
        size1 = x.size()
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x13 = F.relu(self.bn13(self.conv13(x12)))
        x14 = F.relu(self.bn14(self.conv14(x13)))


        # Stage 1a
        size1a = x14.size()
        x1a = x14.view(-1, size1a[1]*self.patch_size*self.patch_size)

        # Stage 2
        size2 = x1a.size()
        x21 = F.relu(self.linear21(x1a))
        x22 = self.linear22(x21)

        encoder = F.normalize(x22, p=2, dim=1)

        return encoder


class Decoder(nn.Module):
    def __init__(self, bands_nb, patch_size):

        super(Decoder, self).__init__()

        self.patch_size = patch_size

        # Stage 2d
        self.linear22d = nn.Linear(2*self.patch_size*self.patch_size, 12*self.patch_size*self.patch_size)
        self.linear21d = nn.Linear(12*self.patch_size*self.patch_size, 64*self.patch_size*self.patch_size)

        # Stage 1d
        self.conv14d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn14d = nn.BatchNorm2d(64)
        self.conv13d = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn13d = nn.BatchNorm2d(32)
        self.conv12d = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(32)
        self.conv11d = nn.Conv2d(32, bands_nb, kernel_size=3, padding=1)

    def forward(self, x):

        # Stage 2d
        x22d = F.relu(self.linear22d(x))
        x21d = self.linear21d(x22d)

        # Stage 1ad
        size21d = x21d.size()
        x1ad = x21d.view(size21d[0], 64, self.patch_size, self.patch_size)


        # Stage 1d
        x14d = F.relu(self.bn14d(self.conv14d(x1ad)))
        x13d = F.relu(self.bn13d(self.conv13d(x14d)))
        x12d = F.relu(self.bn12d(self.conv12d(x13d)))
        x11d = self.conv11d(x12d)

        decoder = x11d

        return decoder


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        e = self.ecoder(x)
        d = self.decoder(e)
        return e, d
