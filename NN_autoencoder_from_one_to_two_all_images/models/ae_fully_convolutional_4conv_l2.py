from torch import nn
import torch.nn.functional as F




class Encoder(nn.Module):
    def __init__(self, bands_nb, patch_size):

        super(Encoder, self).__init__()

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
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()


    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x13 = F.relu(self.bn13(self.conv13(x12)))
        x14 = self.conv14(x13)
        size14 = x14.size()
        x14_ = x14.view(size14[0], size14[1], size14[2]*size14[3])
        x14_ = F.normalize(x14_, p=2, dim=2)
        encoded = x14_.view(size14[0], size14[1], size14[2], size14[3])


        return encoded


class Decoder(nn.Module):
    def __init__(self, bands_nb, patch_size):

        super(Decoder, self).__init__()

        # Stage 1d
        # Feature decoder
        self.conv14d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn14d = nn.BatchNorm2d(64)
        self.conv13d = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn13d = nn.BatchNorm2d(32)
        self.conv12d = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(32)
        self.conv11d = nn.Conv2d(32, bands_nb, kernel_size=3, padding=1)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()


    def forward(self, x):
        # Stage 1d
        x14d = F.relu(self.bn14d(self.conv14d(x)))
        x13d = F.relu(self.bn13d(self.conv13d(x14d)))
        x12d = F.relu(self.bn12d(self.conv12d(x13d)))
        x11d = self.conv11d(x12d)

        decoded = x11d

        return decoded


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        e, id1 = self.ecoder(x)
        d = self.decoder(e, id1)
        return e, d