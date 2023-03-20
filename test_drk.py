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
import torchvision
from PIL import Image
from model_drk import Generator, Discriminator
import time
import glob
import argparse

# + ------- +
# | SET GPU |
# + ------- +
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# + ------------- +
# | SET ARGUMENTS |
# + ------------- +
parser = argparse.ArgumentParser(description='Arguments for AnoGAN model training')
parser.add_argument('--random_seed', type = int, default = 42, help = 'Random seed')
parser.add_argument('--epochs', type = int, default = 10, help = 'The number of epochs')
parser.add_argument('--train_val_ratio', type = float, default = 0.2, help = 'Ratio between train and val dataset')
parser.add_argument('--batch_size', type = int, default = 32, help = 'The size of batch')
parser.add_argument('--time_step', type = int, default = 30, help = 'Timestep for temporalization of time-series data')
parser.add_argument('--lr', type = int, default = 0.0002, help = 'Learning rate')
parser.add_argument('--num_gpus', type = int, default = 1, help = 'The number of available GPUs')
parser.add_argument('--num_workers', type = int, default = 0, help = 'The number of available workers')
parser.add_argument('--nz', type = int, default = 100, help = 'Dimension of noise vector')
parser.add_argument('--ngf', type = int, default = 64, help = 'Number of conv filters for generator')
parser.add_argument('--ndf', type = int, default = 8, help = 'Number of conv filters for discriminator')
parser.add_argument('--nc', type = int, default = 1, help = 'Number of channels for input image')  ## color: 3 / gray: 1
parser.add_argument('--beta1', type = float, default = 0.5, help = 'Parameters of Adam optimizer')

### ----- args 에 위 내용 저장
# args = parser.parse_args()  ### Pycharm 에서 사용하는 경우의 Code
args = parser.parse_args(args=[])  ### Jupyter notebook 에서 사용하는 경우의 Code

### ----- 입력받은 인자값 출력
print("##### Arguments #####\n")
print("random_seed: ", args.random_seed)
print("Epochs: ", args.epochs)
print("train_val_ratio: ", args.train_val_ratio)
print("batch_size: ", args.batch_size)
print("Learning rate: ", args.lr)
print("num_gpus: ", args.num_gpus)
print("num_workers: ", args.num_workers)
print("nz: ", args.nz)
print("ngf: ", args.ngf)
print("ndf: ", args.ndf)
print("nc: ", args.nc)
print("beta1: ", args.beta1)

# + ------------------ +
# | SET HYPERPARAMETER |
# + ------------------ +
params ={
    "epoch": args.epochs,
    "batch_size": args.batch_size,
    "learning_rate": args.lr,
    "num_gpus": args.num_gpus,
    "num_workeres": args.num_workers,
    "nz": args.nz,
    "ngf": args.ngf, 
    "ndf": args.ndf, 
    "nc": args.nc, 
    "beta1": args.beta1
}

# + ----------------------------- +
# | DEFINE ANOMALY SCORE FUNCTION |
# + ----------------------------- +
### ----- Lambda = 0.1 according to paper
### ----- x is new data, G_z is closely regenerated data
def Anomaly_score(x, G_z, Lambda=0.1):
    _, x_feature = discriminator(x)  ## out, feature = discriminator(x)
    _, G_z_feature = discriminator(G_z)  ## out, feature = discriminator(G_z)

    residual_loss = torch.sum(torch.abs(x - G_z))  ## Real image와 생성된 Fake image 간 둘의 차이로 Residual loss 계산
    discrimination_loss = torch.sum(torch.abs(x_feature - G_z_feature))  ## Discriminator를 통한 Real image의 Feature와 Fake image의 Feature 간 차이 계산 (Feature 단위의 MAE Loss)

    total_loss = (1 - Lambda) * residual_loss + Lambda * discrimination_loss  ## 위에서 구한 두 가지 Loss의 합으로 각각 (1-lambda), lambda의 가중치를 곱하여 합산

    return total_loss  ## 이 Total loss는 GAN 모델 학습이 아니라 입력 이미지와 가장 유사한 이미지를 생성하기 위해 Latent space의 z 벡터를 Backpropagation을 통해 업데이트 하기 위한 Loss

# + ------------ +
# | DEFINE MODEL |
# + ------------ +
# Put class objects on Multiple GPUs using
# torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
# device_ids: default all devices / output_device: default device 0
# along with .cuda()
generator = nn.DataParallel(Generator(params["nz"], params["ngf"], params["nc"])).cuda()
discriminator = nn.DataParallel(Discriminator(params["ndf"], params["nc"])).cuda()

# + ------------------ +
# | LOAD TRAINED MODEL |
# + ------------------ +
try:
    generator.load_state_dict(torch.load('./test/saved_model/generator.pkl'))
    discriminator.load_state_dict(torch.load('./test/saved_model/discriminator.pkl'))
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass

### ----- Set evaluation mode
generator.eval()
discriminator.eval()

"""
start = time.time()
image = glob.glob("D:/gitbucket/Project_LSD_AnoGAN_torch/LSD_JKC/Python_LSD/LSD_AnoGAN_pytorch/dataset/test/test/*jpg")[0]
img = Image.open(image)
trans1 = transforms.ToTensor()
trans2 = transforms.Normalize((0.5,), (0.5,))

image = trans2(trans1(img))
image = image.unsqueeze(0)

test_image = Variable(image).cuda()
test_size = image.size()[0]

z = Variable(init.normal(torch.zeros(test_size, 100), mean=0, std=0.1), requires_grad=True)
z_optimizer = torch.optim.Adam([z], lr=1e-4)

gen_fake = generator(z.cuda())
loss = Anomaly_score(Variable(test_image).cuda(), gen_fake)

for j in range(5): #original 5000
    gen_fake = generator(z.cuda())
    loss = Anomaly_score(Variable(test_image).cuda(), gen_fake, Lambda=0.01)
    loss.backward()
    z_optimizer.step()

#print(loss.cpu().data)
print(loss.item())
print("time :", time.time()-start)
"""

# + ----------------------------------- +
# | COMPUTE STATISTIC FOR TRAINING DATA |
# + ----------------------------------- +
### ----- 학습 데이터의 평균, 표준편차 계산
trans = transforms.Compose([
    transforms.ToTensor(), ## torch.Tensor로 변환
    transforms.ToPILImage(), ## Convert a tensor or an ndarray to PIL image
    transforms.Grayscale(num_output_channels=1), ## Grayscale로 데이터 변환
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
    ])

val_data = torchvision.datasets.ImageFolder(
    # root=r'D:\gitbucket\Project_LSD_AnoGAN_torch\LSD_JKC\Python_LSD\LSD_AnoGAN_pytorch\dataset_raw_convert\car1_mt_30t_figure_1sec\stft\total',
    root=r'D:\gitbucket\Project_LSD_AnoGAN_torch\LSD_JKC\Python_LSD\LSD_AnoGAN_pytorch\dataset_raw_convert\car1_lt_30t_figure_1sec\stft\total',
    transform=trans
    )
val_loader = torch.utils.data.DataLoader(
    dataset=val_data, batch_size=1, shuffle=False, num_workers=0
    )

# + --------- +
# | INFERENCE |
# + --------- +
val_anomaly = []
total = []

for i, (image, label) in enumerate(val_loader):

    # test_image = Variable(image).cuda()  ## https://medium.com/@poperson1205/%EC%B4%88%EA%B0%84%EB%8B%A8-pytorch%EC%97%90%EC%84%9C-tensor%EC%99%80-variable%EC%9D%98-%EC%B0%A8%EC%9D%B4-a846dfb72119
    test_image = torch.Tensor(image).cuda()
    test_size = image.size()[0]

    # z = Variable(init.normal(torch.zeros(test_size, 100), mean=0, std=0.1), requires_grad=True)   ## https://medium.com/@poperson1205/%EC%B4%88%EA%B0%84%EB%8B%A8-pytorch%EC%97%90%EC%84%9C-tensor%EC%99%80-variable%EC%9D%98-%EC%B0%A8%EC%9D%B4-a846dfb72119
    z = torch.Tensor(init.normal_(torch.zeros(test_size, 100), mean=0, std=0.1)).requires_grad_(requires_grad=True)
    z_optimizer = torch.optim.Adam([z], lr=1e-4)

    gen_fake = generator(z.cuda())  ## Generate fake image
    loss = Anomaly_score(torch.Tensor(test_image).cuda(), gen_fake)
    # print(loss)

    for j in range(5):  ## Original 5000
        gen_fake = generator(z.cuda())
        # loss = Anomaly_score(Variable(test_image).cuda(), gen_fake, Lambda=0.01)
        loss = Anomaly_score(torch.Tensor(test_image).cuda(), gen_fake, Lambda=0.01)
        loss.backward()
        z_optimizer.step() ## Parameter update

    print(loss.item())
    start = time.time()
    total.append([loss.item(), start])
    val_anomaly.append(loss.item())

val_mean = np.mean(val_anomaly)
val_std = np.std(val_anomaly)
# print(val_anomaly)
print('mean: ', val_mean)
print('std: ', val_std)

import csv
with open('test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(total)
