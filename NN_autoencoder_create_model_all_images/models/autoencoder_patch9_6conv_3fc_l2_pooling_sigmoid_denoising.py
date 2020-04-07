from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


def gaussian(ins, mean, stddev):
    noise_bands = None
    factor = 1
    for band in range(ins.size(1)):
        noise = factor * Variable(ins.data.new(torch.Size([ins.size(0), 1, ins.size(2), ins.size(3)])).normal_(mean[band], stddev[band]))
        if noise_bands is None:
            noise_bands = noise
        else:
            noise_bands = torch.cat((noise_bands, noise), 1)
    return ins + noise_bands


class Encoder(nn.Module):
    def __init__(self, bands_nb, patch_size, nb_features, norm):
        super(Encoder, self).__init__()

        self.patch_size = patch_size
        self.bands_nb = bands_nb
        self.nb_features = nb_features
        self.mean = norm[:, 0].flatten()
        self.std = norm[:, 1].flatten()


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
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, return_indices=True, stride=3)
        self.conv15 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn15 = nn.BatchNorm2d(128)
        self.conv16 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn16 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, return_indices=True, stride=1)

        # Stage 2
        # Classifier
        self.linear21 = nn.Linear(128, 64)
        self.linear22 = nn.Linear(64, 32)
        self.linear23 = nn.Linear(32, nb_features)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()

        self.apply(weights_init)



    def forward(self, x):
        # Stage 1
        if self.training:
            x = gaussian(x, self.mean, self.std)
        size1 = x.size()
        x11 = self.activation1(self.bn11(self.conv11(x)))
        x12 = self.activation1(self.bn12(self.conv12(x11)))
        x13 = self.activation1(self.bn13(self.conv13(x12)))
        x14 = self.activation1(self.bn14(self.conv14(x13)))
        x1p, id1 = self.maxpool1(x14)
        x15 = self.activation1(self.bn15(self.conv15(x1p)))
        x16 = self.activation1(self.bn16(self.conv16(x15)))
        x2p, id2 = self.maxpool2(x16)

        # Stage 1a
        x1a = x2p.view(-1, 128)
        # print(x1a.size())

        # Stage 2
        x21 = self.activation1(self.linear21(x1a))
        x22 = self.activation1(self.linear22(x21))
        x23 = self.linear23(x22)

        if self.nb_features>1:
            encoder = F.normalize(x23, p=2, dim=1)
        else:
            encoder = x23
        #encoder = x22
        return encoder, [id1, id2]


class Decoder(nn.Module):
    def __init__(self, bands_nb, patch_size, nb_features):
        super(Decoder, self).__init__()

        self.patch_size = patch_size
        self.bands_nb = bands_nb

        # Stage 3d
        # Classifier decoder
        self.linear23d = nn.Linear(nb_features, 32)
        self.linear22d = nn.Linear(32, 64)
        self.linear21d = nn.Linear(64, 128)

        # Stage 1d
        # Feature extraction decoder
        self.unpool2 = nn.MaxUnpool2d(kernel_size=3, stride=1)
        self.conv16d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn16d = nn.BatchNorm2d(128)
        self.conv15d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn15d = nn.BatchNorm2d(64)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=3, stride=3)
        self.conv14d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn14d = nn.BatchNorm2d(64)
        self.conv13d = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn13d = nn.BatchNorm2d(32)
        self.conv12d = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(32)
        self.conv11d = nn.Conv2d(32, bands_nb, kernel_size=3, padding=1)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()

        self.apply(weights_init)


    def forward(self, x, id):
        id1, id2 = id[0], id[1]
        # Stage 3d
        x23d = self.activation1(self.linear23d(x))
        x22d = self.activation1(self.linear22d(x23d))
        x21d = self.activation1(self.linear21d(x22d))


        # Stage 1ad
        size = x21d.size()
        x1ad = x21d.view(size[0], 128, 1, 1)

        # print(x1ad.size())
        # Stage 1d
        x2pd = self.unpool2(x1ad, id2, output_size=(size[0], 128, 3, 3))
        # print(x2pd.size())
        x16d = self.activation1(self.bn16d(self.conv16d(x2pd)))
        # print(x16d.size())
        x15d = self.activation1(self.bn15d(self.conv15d(x16d)))
        x1pd = self.unpool1(x15d, id1, output_size=(size[0], 64, self.patch_size, self.patch_size))
        x14d = self.activation1(self.bn14d(self.conv14d(x1pd)))
        x13d = self.activation1(self.bn13d(self.conv13d(x14d)))
        x12d = self.activation1(self.bn12d(self.conv12d(x13d)))
        x11d = self.activation2(self.conv11d(x12d))

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