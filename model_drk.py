# + ---------------- +
# | IMPORT LIBRARIES |
# + ---------------- +
import os
import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
from torch.autograd import Variable
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# + ------- +
# | SET GPU |
# + ------- +
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# + ---------------------- +
# | DEFINE CLASS FOR MODEL |
# + ---------------------- +
## Generator receives random noise z and create 1x28x28 image
## we can name each layer using OrderedDict
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc

        """
        ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, 
            groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

        Input size for Generator: torch.Size([batch_size, nz, 1, 1]) -> torch.Size([32, 100, 1, 1])
        """
        self.layer1 = nn.Sequential(
            nn.Linear(self.nz, 7*7*self.ngf*8),
            nn.ReLU()
            )   

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.ngf*8, out_channels=self.ngf*4, 
                kernel_size=3, stride=2, padding=1, output_padding=1),  ## torch.Size([32, 256, 4, 4])
            nn.BatchNorm2d(self.ngf*4),
            nn.LeakyReLU(),
            # nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=self.ngf*4, out_channels=self.ngf*2, 
                kernel_size=3, stride=1, padding=1),  ## torch.Size([32, 128, 7, 7])
            nn.BatchNorm2d(self.ngf*2),
            nn.LeakyReLU(),  
            # nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=self.ngf*2, out_channels=self.ngf, 
                kernel_size=3, stride=1, padding=1),  ## torch.Size([32, 64, 14, 14])
            nn.BatchNorm2d(self.ngf),
            nn.LeakyReLU(),
            # nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=self.ngf, out_channels=self.nc, 
                kernel_size=3, stride=2, padding=1, output_padding=1),  ## torch.Size([32, 1, 28, 28])
            nn.Tanh()
            )


    def forward(self, z):
        out = self.layer1(z)
        out = out.view(out.size()[0], self.ngf*8, 7, 7)
        out = self.layer2(out)

        return out

## Discriminator receives 1x28x28 image and returns a float number 0~1
## we can name each layer using OrderedDict
class Discriminator(nn.Module):
    def __init__(self, ndf=8, nc=3):
        super(Discriminator,self).__init__()
        self.ndf = ndf
        self.nc = nc
        """
        torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, 
            bias=True, padding_mode='zeros', device=None, dtype=None)
        
        Input size for discriminator: torch.Size([batch_size, num_channel, h, w]) -> torch.size([32, 1, 28, 28])
        """
        self.feature_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.nc, out_channels=self.ndf, 
                kernel_size=3, padding=1),  ## batch_size, ndf, h, w -> torch.size([32, 8, 28, 28])
            # nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(),
            # nn.ReLU(),

            nn.Conv2d(
                in_channels=self.ndf, out_channels=self.ndf*2, 
                kernel_size=3, stride=2, padding=1),  ## batch_size, ndf*2, h/2, w/2 -> torch.size([32, 16, 14, 14])
            # nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(),
            # nn.ReLU(),

            nn.Conv2d(
                in_channels=self.ndf*2, out_channels=self.ndf*4, 
                kernel_size=3, stride=2,padding=1),  ## batch_size, ndf*4, h/4, w/4 -> torch.size([32, 32, 7, 7])
            # nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(),
            # nn.ReLU(),

            nn.Conv2d(
                in_channels=self.ndf*4, out_channels=self.ndf*8, 
                kernel_size=3, padding=1),  ## torch.size([32, 64, 7, 7])
            # nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(),
            # nn.ReLU()
            )

        self.dis_layer = nn.Sequential(
            nn.Linear(in_features=64*7*7, out_features=1),
            nn.Sigmoid()
            )

    def forward_features(self, x):
        features = self.feature_layer(x)

        return features 

    def forward(self, x):
        features = self.forward_features(x)
        features = features.view(features.size()[0], -1)
        features = features

        discrimination = self.dis_layer(features)  ## discrimination.shape = torch.Size([32, 1])

        return discrimination, features