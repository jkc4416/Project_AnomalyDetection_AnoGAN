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
from model import Generator
from model import Discriminator
import torchvision
from utils import EarlyStopping
import argparse

# + ------- +
# | SET GPU |
# + ------- +
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

epoch = 20
batch_size = 64
learning_rate = 0.0002
num_gpus = 8

# Download Data
trans = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
#trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = torchvision.datasets.ImageFolder(
    root='D:/gitbucket/Project_LSD_AnoGAN_torch/LSD_JKC/Python_LSD/LSD_AnoGAN_pytorch/dataset/train', 
    transform=trans
    )  ## Size of train images: 1 x 28 x 28
test_data = torchvision.datasets.ImageFolder(
    root='D:/gitbucket/Project_LSD_AnoGAN_torch/LSD_JKC/Python_LSD/LSD_AnoGAN_pytorch/dataset/test', 
    transform=trans
    )  ## Size of test images: 1 x 28 x 28

# Set Data Loader(input pipeline)
train_loader = torch.utils.data.DataLoader(
    dataset=train_data, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=0
    )
#test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = len(test_data))

# Put class objects on Multiple GPUs using
# torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
# device_ids: default all devices / output_device: default device 0
# along with .cuda()

generator = nn.DataParallel(Generator()).cuda()
discriminator = nn.DataParallel(Discriminator()).cuda()

#generator = Generator()
#discriminator = Discriminator()

# loss function, optimizers, and labels for training

loss_func = nn.MSELoss()
gen_optim = torch.optim.Adam(
    generator.parameters(), 
    lr=5*learning_rate,
    betas=(0.5,0.999)
    )
dis_optim = torch.optim.Adam(
    discriminator.parameters(), 
    lr=learning_rate,
    betas=(0.5,0.999)
    )

# ones_label = Variable(torch.ones(batch_size, 1)).cuda()
# zeros_label = Variable(torch.zeros(batch_size, 1)).cuda()

def image_check(gen_fake):
    img = gen_fake.data.numpy()
    for i in range(2):
        plt.imshow(img[i][0],cmap='gray')
        plt.show()

# model restore if any
try:
    generator.load_state_dict(torch.load('./saved_model/generator.pkl'))
    discriminator.load_state_dict(torch.load('./saved_model/discriminator.pkl'))
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass

# train
for i in range(epoch):
    for j, (image, label) in enumerate(train_loader):
        image = Variable(image).cuda()
        batch_size = image.size()[0]
        #print(batch_size)

        ones_label = Variable(torch.ones(batch_size, 1)).cuda()
        zeros_label = Variable(torch.zeros(batch_size, 1)).cuda()

        # generator
        gen_optim.zero_grad()

        z = Variable(init.normal(torch.Tensor(batch_size, 100), mean=0, std=0.1)).cuda()
        gen_fake = generator.forward(z)
        dis_fake, _ = discriminator.forward(gen_fake)

        gen_loss = torch.sum(loss_func(dis_fake, ones_label))  # fake classified as real
        gen_loss.backward(retain_graph=True)
        gen_optim.step()

        # discriminator
        dis_optim.zero_grad()

        z = Variable(init.normal(torch.Tensor(batch_size, 100), mean=0, std=0.1)).cuda()
        gen_fake = generator.forward(z)
        dis_fake, _ = discriminator.forward(gen_fake)

        dis_real, _ = discriminator.forward(image)
        dis_loss = torch.sum(loss_func(dis_fake, zeros_label)) + torch.sum(loss_func(dis_real, ones_label))
        dis_loss.backward()
        dis_optim.step()

        # model save
        if j % 50 == 0:
            # print(gen_loss,dis_loss)
            torch.save(generator.state_dict(), './saved_model/generator.pkl')
            torch.save(discriminator.state_dict(), './saved_model/discriminator.pkl')

            print("{}th iteration gen_loss: {} dis_loss: {}".format(i, gen_loss.data, dis_loss.data))
            v_utils.save_image(gen_fake.data[0:25], "./result/gen_{}_{}.png".format(i, j), nrow=5)

    #image_check(gen_fake.cpu())
