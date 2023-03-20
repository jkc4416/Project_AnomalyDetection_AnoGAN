# + ---------------- +
# | IMPORT LIBRARIES |
# + ---------------- +
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision
import logging
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.utils as v_utils
from torch.utils.data import DataLoader

# + -------------- +
# | SAVE FILE LIST |
# + -------------- +
def save_file_list(path):
    """
    1. Save file list for the target path
    """
    file_ls = []  ### train data file path 목록을 저장하기 위한 빈 리스트 생성
    max_depth = 0

    for path, dir, files in os.walk(path):
        for file in files:
            current = os.path.join(path, file)

            if os.path.getsize(current) > 1000000000:  ### 일정 크기 이상인 파일들은 삭제
                os.remove(current)
            else:                
                file_ls.append(current)

                if len(current.split('/')) > max_depth: 
                    max_depth = len(current.split('/'))

    # file_ls = [filename for filename in file_ls if '.csv' in filename]  ### .csv 확장자로 끝나는 파일만 file_ls 변수에 저장

    print(len(file_ls)) ### train data file 개수 확인

    return file_ls


# + ------------ +
# | LOAD DATASET |
# + ------------ +
def  data_loader(target_car, batch_size, num_workers):
    trans = transforms.Compose([
        transforms.ToTensor(), 
        transforms.ToPILImage(), 
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, ), (0.5, ))
        ])
    # trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    train_data = torchvision.datasets.ImageFolder(
        # root='D:/gitbucket/Project_LSD_AnoGAN_torch/LSD_JKC/Python_LSD/LSD_AnoGAN_pytorch/dataset/train', 
        root='./dataset_raw_convert/' + target_car + '_mt_30t_figure_1sec_ldwt/stft/total', 
        transform=trans
        )  ## Size of train images: 1 x 28 x 28

    test_data = torchvision.datasets.ImageFolder(
        # root='D:/gitbucket/Project_LSD_AnoGAN_torch/LSD_JKC/Python_LSD/LSD_AnoGAN_pytorch/dataset/test', 
        root='./dataset_raw_convert/' + target_car + '_lt_30t_figure_1sec_ldwt/stft/total', 
        transform=trans
        )  ## Size of test images: 1 x 28 x 28

    # + -------------- +
    # | SET DATALOADER |
    # + -------------- +
    ### ----- Set Data Loader(input pipeline)
    train_loader = DataLoader(
        dataset=train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
        )
    test_loader = DataLoader(
        dataset = test_data, 
        batch_size = len(test_data),
        drop_last=True
        )
    
    return train_loader, test_loader



    # file_ls = [filename for filename in file_ls if '.csv' in filename]  ### .csv 확장자로 끝나는 파일만 file_ls 변수에 저장

    print(len(file_ls)) ### train data file 개수 확인

    return file_ls



# + ------------------------- +
# | DEFINE WEIGHT INITIALIZER |
# + ------------------------- +
## https://comlini8-8.tistory.com/7
def weights_init(m):  
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
  
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data,0)



# + --------------------- +
# | CHECK GENERATED IMAGE |
# + --------------------- +
def image_check(gen_fake):
    img = gen_fake.data.numpy()

    for i in range(2):
        plt.imshow(img[i][0],cmap='gray')
        plt.show()



# + -------------------------- +
# | DEFINE TEMPORALIZE FUCTION |
# + -------------------------- +
def temporalize(X, time_range, step_size):
    onesec_X = []

    for i in range(len(X)-time_range-step_size):
        print(i)
        seq = X[i*step_size:i*step_size+time_range]

        if len(seq) == time_range:
            onesec_X.append(seq)

        else:
            break

    return onesec_X



# + --------------------------------------- +
# | DEFINE FUNCTION TO EXTRACT FOLDER NAMES |
# + --------------------------------------- +
def extractFolders(rootpath):
    ls_folders = list()
    for item in os.listdir(rootpath):  ## 해당 폴더 내 모든 파일 및 폴더 추출
        sub_folder = os.path.join(rootpath, item)

        if os.path.isdir(sub_folder): ## 폴더 여부 확인
            ls_folders.append(sub_folder)
    
    return ls_folders
    


# + -------------- +
# | EARLY STOPPING |
# + -------------- +
class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss