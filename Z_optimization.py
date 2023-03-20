import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
import torchvision

from model import Generator, Discriminator

import time

# Lambda = 0.1 according to paper
# x is new data, G_z is closely regenerated data
def Anomaly_score(x, G_z, Lambda=0.1):
    _, x_feature = discriminator(x)
    _, G_z_feature = discriminator(G_z)

    residual_loss = torch.sum(torch.abs(x - G_z))
    discrimination_loss = torch.sum(torch.abs(x_feature - G_z_feature))

    total_loss = (1 - Lambda) * residual_loss + Lambda * discrimination_loss
    return total_loss

#trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trans = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])

#train_data = torchvision.datasets.ImageFolder(root='C:/Users/Administrator/PycharmProjects/AnoGAN_pytorch/dataset/train', transform=trans)
test_data = torchvision.datasets.ImageFolder(root='C:/Users/posco/PycharmProjects/LSD_AnoGAN_pytorch/dataset/test', transform=trans)

# Set Data Loader(input pipeline)
#train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
#test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = len(test_data), num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = 1, num_workers=0)

# Put class objects on Multiple GPUs using
# torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
# device_ids: default all devices / output_device: default device 0
# along with .cuda()

generator = nn.DataParallel(Generator()).cuda()
discriminator = nn.DataParallel(Discriminator()).cuda()

# model restore if any
try:
    generator.load_state_dict(torch.load('./saved_model/generator.pkl'))
    discriminator.load_state_dict(torch.load('./saved_model/discriminator.pkl'))
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass

for i, (image, label) in enumerate(test_loader):
    start = time.time()
    test_image = Variable(image).cuda()
    test_size = image.size()[0]

    z = Variable(init.normal(torch.zeros(test_size, 100), mean=0, std=0.1), requires_grad=True)
    z_optimizer = torch.optim.Adam([z], lr=1e-4)

    gen_fake = generator(z.cuda())
    loss = Anomaly_score(Variable(test_image).cuda(), gen_fake)
    #print(loss)

    for j in range(5): #original 5000
        gen_fake = generator(z.cuda())
        loss = Anomaly_score(Variable(test_image).cuda(), gen_fake, Lambda=0.01)
        loss.backward()
        z_optimizer.step()

    #print(loss.cpu().data)
    print(str(loss.item()))
    print("time :", time.time()-start)