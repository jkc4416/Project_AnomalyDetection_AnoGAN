# + ---------------- +
# | IMPORT LIBRARIES |
# + ---------------- +
import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torchvision.utils as v_utils
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from model_drk import Generator
from model_drk import Discriminator
import torchvision
from utils import EarlyStopping, data_loader, weights_init
import argparse
import random



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
parser.add_argument('--epochs', type = int, default = 30, help = 'The number of epochs')
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
    "num_workers": args.num_workers,
    "nz": args.nz,
    "ngf": args.ngf, 
    "ndf": args.ndf, 
    "nc": args.nc, 
    "beta1": args.beta1
}

### ----- Secure reproducibility
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.random_seed)



# + ---------------- +
# | TARGET LADLE CAR |
# + ---------------- +
cars = ['car1', 'car2']


for idx, target_car in enumerate(cars):
    # + ----------------------------- +
    # | LOAD DATASET & SET DATALOADER |
    # + ----------------------------- +
    train_loader, test_loader = data_loader(target_car, params['batch_size'], params['num_workers'])



    # + ------------ +
    # | DEFINE MODEL |
    # + ------------ +
    ### ----- Put class objects on Multiple GPUs using
    ### ----- torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
    ### ----- device_ids: default all devices / output_device: default device 0
    ### ----- along with .cuda()
    generator = nn.DataParallel(Generator(params["nz"], params["ngf"], params["nc"])).cuda()
    discriminator = nn.DataParallel(Discriminator(params["ndf"], params["nc"])).cuda()
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    # generator = Generator()
    # discriminator = Discriminator()



    # + ----------------------------------------------------- +
    # | DEFINE LOSS FUNCTIONN, OPTIMIZER, LABLES FOR TRAINING |
    # + ----------------------------------------------------- +
    loss_func = nn.MSELoss()
    # loss_func = nn.BCELoss()
    gen_optim = torch.optim.Adam(
        generator.parameters(), 
        lr=5*params["learning_rate"],
        betas=(params["beta1"], 0.999)
        )  ## Opimizer for generator
    dis_optim = torch.optim.Adam(
        discriminator.parameters(), 
        lr=params["learning_rate"],
        betas=(params["beta1"], 0.999)
        )  ## Opimizer for discriminator
    # ones_label = Variable(torch.ones(batch_size, 1)).cuda()  ## deprecated
    # zeros_label = Variable(torch.zeros(batch_size, 1)).cuda()  ## deprecated
    # ones_label = torch.Tensor(torch.ones(batch_size, 1)).cuda()
    # zeros_label = torch.Tensor(torch.zeros(batch_size, 1)).cuda()



    # + -------------- +
    # | MODEL TRAINING |
    # + -------------- +
    ### ----- Load checkpoints
    # try:
    #     generator.load_state_dict(torch.load('./saved_model/generator.pkl'))
    #     discriminator.load_state_dict(torch.load('./saved_model/discriminator.pkl'))
    #     print("\n--------model restored--------\n")
    # except:
    #     print("\n--------model not restored--------\n")
    #     pass
    gen_losses = []
    dis_losses = []

    min_gen_loss = 1000
    min_dis_loss = 1000

    ### ----- Train
    for i in range(params["epoch"]):
        for j, (image, label) in enumerate(train_loader):
            # image = Variable(image).cuda()  ## deprecated
            image = torch.Tensor(image).cuda()
            # print(image.size())
            batch_size = image.size()[0]
            # print(batch_size)

            # ones_label = Variable(torch.ones(batch_size, 1)).cuda()  ## deprecated
            # zeros_label = Variable(torch.zeros(batch_size, 1)).cuda()  ## deprecated
            ones_label = torch.Tensor(torch.ones(batch_size, 1)).cuda()  ## torch.Size([batch_size, 1])
            zeros_label = torch.Tensor(torch.zeros(batch_size, 1)).cuda()  ## torch.Size([batch_size, 1])

            ### ----- Generator
            gen_optim.zero_grad()

            z = Variable(init.normal(torch.Tensor(batch_size, 100), mean=0, std=0.1)).cuda()  ## deprecated
            # z = torch.Tensor(init.normal_(torch.zeros(batch_size, 100), mean=0, std=0.1)).requires_grad_(requires_grad=True)
            gen_fake = generator.forward(z)  ## Generate fake image
            dis_fake, _ = discriminator.forward(gen_fake)  ## Discriminate generated fake image

            gen_loss = torch.sum(loss_func(dis_fake, ones_label))  ## fake classified as real
            gen_losses.append(gen_loss.detach().cpu().item())

            gen_loss.backward(retain_graph=True)
            gen_optim.step()

            

            ### ----- Discriminator
            dis_optim.zero_grad()

            z = Variable(init.normal(torch.Tensor(batch_size, 100), mean=0, std=0.1)).cuda()  ## deprecated
            # z = torch.Tensor(init.normal_(torch.zeros(batch_size, 100), mean=0, std=0.1)).requires_grad_(requires_grad=True)
            gen_fake = generator.forward(z)
            dis_fake, _ = discriminator.forward(gen_fake)

            dis_real, _ = discriminator.forward(image)  ## image.shape = torch.Size([32, 1, 28, 28])
            dis_loss = torch.sum(loss_func(dis_fake, zeros_label)) + torch.sum(loss_func(dis_real, ones_label))
            dis_losses.append(dis_loss.detach().cpu().item())

            dis_loss.backward()
            dis_optim.step()

            

            ### ----- Model save
            if j % 50 == 0:
                print("{}th iteration gen_loss: {} dis_loss: {}".format(i, gen_loss.data, dis_loss.data))
                
                if min_dis_loss > dis_loss:
                    min_dis_loss = dis_loss
                    torch.save(generator.state_dict(), './model/' + target_car + '/generator.pkl')  ## 경로에 '_' 포함 시 계속 경로 문제로 OSError 발생 (OSError: [Errno 22] Invalid argument)
                    torch.save(discriminator.state_dict(), './model/' + target_car + '/discriminator.pkl')

                    print("Model save!")

                # v_utils.save_image(gen_fake.data[0:25], "./result/gen_{}_{}.png".format(i, j), nrow=5)

        # image_check(gen_fake.cpu())
