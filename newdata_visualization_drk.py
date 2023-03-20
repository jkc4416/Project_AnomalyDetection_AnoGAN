# + ---------------- +
# | IMPORT LIBRARIES |
# + ---------------- +
import os
import glob
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
import pandas as pd
import logging
from utils import save_file_list, extractFolders
import datetime
import csv
from sklearn.preprocessing import MinMaxScaler



# + ---------------- +
# | SET PLOT OPTIONS |
# + ---------------- +
plt.rc('font', size=20)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('legend', fontsize=15)
plt.rc('figure', titlesize=20)



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
parser.add_argument('--target_filename', type = str, default = 'SB25880_20210723-234831_1', help = 'Name of the target file')

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
print("target_file: ", args.target_filename)



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



# + ----------------------------- +
# | DEFINE ANOMALY SCORE FUNCTION |
# + ----------------------------- +
    


# + ----------- +
# | SENSOR TYPE |
# + ----------- +
sensor_type = ['aco', 'vib']



# + -------------------------- +
# | LOAD DATA FROM TARGET FILE |
# + -------------------------- +
### ----- List of csv files
for idx, sensor in enumerate(sensor_type):

    ### ----- Directory for training image files
    rootpath_rawdata = './newdataset_raw'
    rootpath_imgs = './newdataset_stft'
    
    raw_file_ls = save_file_list(rootpath_rawdata)
    subFoldersSensor = extractFolders(rootpath_imgs)
    subFoldersSensor = [folder for folder in subFoldersSensor if 'total' not in folder]

    for idx, (filePath, folderPath) in enumerate(zip(raw_file_ls, subFoldersSensor)):
        target_file_name = folderPath.split('\\')[1]

        path_train_imgs = folderPath + '/stft/*'

        target_train_imgs = glob.glob(path_train_imgs)
        target_train_imgs = [img for img in target_train_imgs if img.endswith(".jpg")]
        
        print("The number of train imgs: {}".format(len(target_train_imgs)))



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
            generator.load_state_dict(torch.load('./newdata_model/' + sensor + '_generator.pkl'))
            discriminator.load_state_dict(torch.load('./newdata_model/' + sensor + '_discriminator.pkl'))
            print("\n--------model restored--------\n")
        except:
            print("\n--------model not restored--------\n")
            pass

        ### ----- Set evaluation mode
        generator.eval()
        discriminator.eval()



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
        data = torchvision.datasets.ImageFolder(
            root = folderPath,
            transform=trans
            )
        data_loader = torch.utils.data.DataLoader(
            dataset=data, batch_size=1, shuffle=False, num_workers=0
            )



        # + --------- +
        # | INFERENCE |
        # + --------- +
        score_anomaly = []
        total = []
            
        for idx, (image, label) in enumerate(data_loader):

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
            score_anomaly.append(loss.item())



        val_mean = np.mean(score_anomaly)
        val_std = np.std(score_anomaly)
        # print(val_anomaly)
        print('mean: ', val_mean)
        print('std: ', val_std)

        import csv
        with open('test.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(total)


        # path___ = r'D:\gitbucket\Project_LSD_AnoGAN_torch\LSD_JKC\Python_LSD\LSD_AnoGAN_pytorch\dataset_raw_convert\car1\SB25880_20210723-234831_1.csv'
        df_ = pd.read_csv(filePath, encoding='ISO-8859-1', sep="/t", header=24)
        df_.columns = ['Sensor']

        date_ls_sensor = []
        date_ls_img = []

        strDateTime_sensor = '2021-01-01 00:00:00.000000'
        dateDateTime_sensor = datetime.datetime.strptime(strDateTime_sensor, '%Y-%m-%d %H:%M:%S.%f')

        strDateTime_img = '2021-01-01 00:00:00.000000'
        dateDateTime_img = datetime.datetime.strptime(strDateTime_sensor, '%Y-%m-%d %H:%M:%S.%f')

        for idx in range(len(df_)):
            date_ls_sensor.append(dateDateTime_sensor)
            dateDateTime_sensor = dateDateTime_sensor + datetime.timedelta(seconds=0.0005)
        
        for idx in range(len(data_loader)):
            date_ls_img.append(dateDateTime_img)
            dateDateTime_img = dateDateTime_img + datetime.timedelta(seconds=0.5)


        def plot_graphs(df, datetime_sensor, datetime_img, score_anomaly):
            plt.ioff()

            fig, ax = plt.subplots(figsize=(16, 10))        
            fig.subplots_adjust(right=0.75)
            
            twin1 = ax.twinx()

            p1, = ax.plot(datetime_sensor, df["Sensor"], "k-", label="Sensor")
            p2, = twin1.plot(datetime_img, score_anomaly, "g--", label="Score", alpha=0.5)
            # p3, = twin2.plot(datetime_series, score_anomaly_normalized, "g--", label="Score", alpha=0.5)
            # plt.text(df["Date"][0], 0.85, 'Average anomaly score: ' + str(round(avg_score_anomaly_normalized, 3)))      
            # plt.ioff()
            ax.set_xlabel("Records")        
            ax.set_ylabel("Sensor")
            twin1.set_ylabel("Score")
            

            ax.yaxis.label.set_color(p1.get_color())
            twin1.yaxis.label.set_color(p2.get_color())
            tkw = dict(size=4, width=1.5)
            ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
            twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
            # ax.tick_params(axis='x', **tkw)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=6)
            ax.tick_params(axis='x', rotation=90)

            ax.legend(handles=[p1, p2])

            plt.savefig(folderPath.split('\\')[0] + '/' + folderPath.split('\\')[1]  + 'plot.jpg')
            plt.close()

        plot_graphs(df_, date_ls_sensor, date_ls_img, score_anomaly)