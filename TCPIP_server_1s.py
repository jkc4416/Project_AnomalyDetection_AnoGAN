import socket
import time
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import settings

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

import model
from model import Generator
from model import Discriminator
import torchvision
import glob
import csv

print("Ready!")

# 접속할 서버 주소입니다. 여기에서는 루프백(loopback) 인터페이스 주소 즉 localhost를 사용합니다.
HOST = '127.0.0.1'

# 클라이언트 접속을 대기하는 포트 번호입니다.
PORT = 9999

# 소켓 객체를 생성합니다. 주소 체계(address family)로 IPv4, 소켓 타입으로 TCP 사용합니다.
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 포트 사용중이라 연결할 수 없다는 WinError 10048 에러 해결를 위해 필요합니다.
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# bind 함수는 소켓을 특정 네트워크 인터페이스와 포트 번호에 연결하는데 사용됩니다.
server_socket.bind((HOST, PORT))

# 서버가 클라이언트의 접속을 허용하도록 합니다.
server_socket.listen()

# accept 함수에서 대기하다가 클라이언트가 접속하면 새로운 소켓을 리턴합니다.
client_socket, addr = server_socket.accept()

# 접속한 클라이언트의 주소입니다.
print('Connected by', addr)

def Anomaly_score(x, G_z, Lambda=0.1):
    _, x_feature = discriminator(x)
    _, G_z_feature = discriminator(G_z)

    residual_loss = torch.sum(torch.abs(x - G_z))
    discrimination_loss = torch.sum(torch.abs(x_feature - G_z_feature))

    total_loss = (1 - Lambda) * residual_loss + Lambda * discrimination_loss
    return total_loss

def scale_minmax(X, X_min=-1, X_max=0, min=0.0, max=1.0):
    if X_min == -1:
        X_std = (X - X.min()) / (X.max() - X.min())
    else:
        X_std = (X - X_min) / (X_max - X_min)
    X_scaled = X_std * (max - min) + min
    return X_scaled, X.min(), X.max()

def spectrogram_image(y, sr, out, hop_length, n_mels, is_train=True, x_min=0, x_max=0):
    # use log-melspectrogram
    #mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*2, hop_length=int(hop_length/10))
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*2, hop_length=hop_length)
    mels = np.log(mels + 1e-9) # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    if is_train:
        img, x_min, x_max = scale_minmax(mels, -1, 0, 0, 255)
        img = img.astype(np.uint8)
    else:
        img, x_min, x_max = scale_minmax(mels, x_min, x_max, 0, 255)
        img = img.astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy
    print(img.shape)

    m, n = img.shape

    # save as PNG
    #img = img[200:300, :]
    img = Image.fromarray(img, 'L')
    #img.save('./dataset/test/test/total.jpg')
    img = img.resize((n, 28))
    img = np.array(img)

    if is_train:
        # for i in range(8, n-36):
        for i in range(0, n - 28):
            blank = []
            if i<150: #350
            #if i <= 350:  # 350
                temp = img[:, i:i + 28]
                for k in range(14):
                    blank.append(temp[:,-1])
                for k in range(14):
                    blank.append(temp[:,-2])
                blank = np.array(blank)
                temp = Image.fromarray(blank)
                #temp = Image.fromarray(temp)
                temp.save('./dataset/train/train/' + str(i) + '.jpg')
            else:
                temp = img[:, i:i + 28]
                for k in range(14):
                    blank.append(temp[:,-1])
                for k in range(14):
                    blank.append(temp[:, -2])
                blank = np.array(blank)
                temp = Image.fromarray(blank)
                #temp = Image.fromarray(temp)
                temp.save('./dataset/valid/valid/' + str(i) + '.jpg')
    else:
        blank = [] #210108
        img_total = img
        # temp = img[:, -37:-9]
        temp = img #210108
        for k in range(14): #210108
            blank.append(temp[:, -1]) #210108
        for k in range(14): #210108
            blank.append(temp[:, -2]) #210108
        blank = np.array(blank) #210108
        temp = Image.fromarray(blank) #210108
        #temp = Image.fromarray(img)
        temp.save('./dataset/test/test/0.jpg')
        # img_total = Image.fromarray(img_total)
        # img_total.save('./dataset/test/test/total.jpg')

    return x_min, x_max

# 무한루프를 돌면서
temp = []
save_cnt=0 #201121
while True:

    # 클라이언트가 보낸 메시지를 수신하기 위해 대기합니다.
    data = client_socket.recv(250000)
    print('Received from', addr, data.decode())

    rev = repr(data.decode()).split(',')
    flag = int(rev[0][1])

    # 학습데이터 수집 모드
    if flag == 0:
        temp.extend(list(map(float, rev[901:1000])))  # original 2000
        temp.extend([float(rev[1000][0:8])])  # original 2000
        #print(len(rev))

    # 학습 모드
    elif flag == 1:
        # 세팅 : 기존에 학습된 모델, 학습 데이터 reset
        settings.settings()
        temp_ = np.array(temp)

        hop_length = 500  # number of samples per time-step in spectrogram (512) #original 1000
        n_mels = 350  # number of bins in spectrogram. Height of image
        time_steps = int(len(temp) / 500)  # number of time-steps. Width of image (384) #original 1000
        #time_steps = len(temp) / 500

        sr = 1000 #original 2000
        out = 'out.png'

        start_sample = 0  # starting at beginning
        length_samples = time_steps * hop_length
        window = temp_[start_sample:start_sample + length_samples]

        # 신호를 melspectrogram으로 변환 후 저장
        x_min, x_max = spectrogram_image(window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels, x_min=0, x_max=0)

        # 학습 시작
        start = time.time()
        # 파라미터 설정
        epoch = 30 #original 20
        batch_size = 64
        learning_rate = 0.0002
        num_gpus = 8

        # 모델 로딩
        generator = nn.DataParallel(Generator()).cuda()
        discriminator = nn.DataParallel(Discriminator()).cuda()

        # Loss 및 Optimizer 세팅
        loss_func = nn.MSELoss()
        gen_optim = torch.optim.Adam(generator.parameters(), lr=5 * learning_rate, betas=(0.5, 0.999))
        dis_optim = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        # Data Loader 설정
        trans = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_data = torchvision.datasets.ImageFolder(root='C:/Users/posco/PycharmProjects/LSD_AnoGAN_pytorch/dataset/train', transform=trans)
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)

        for i in range(epoch):
            for j, (image, label) in enumerate(train_loader):
                image = Variable(image).cuda()
                batch_size = image.size()[0]
                # print(batch_size)

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
        print("time :", time.time() - start)

        # 학습 완료 후 완료 여부 전송
        score_ = "et," + str(20)
        client_socket.send(score_.encode()) #201119 수정
        temp = []
        cnt = 0

        total = []
        save_cnt = save_cnt + 1  # 201121

        test_anomaly = []

    # 테스트 모드
    elif flag == 2:
        start = time.time()
        # 14초 대기
        if cnt <= 135:
            temp.extend(list(map(float, rev[901:1000])))  # original 2000
            temp.extend([float(rev[1000][0:8])])  # original 2000
            cnt = cnt+1
            continue
        else:
            temp.extend(list(map(float, rev[901:1000])))  # original 2000
            temp.extend([float(rev[1000][0:8])])  # original 2000
            temp = temp[100:]

            #client_socket.send("s".encode())
            temp_ = np.array(temp)

            # 세팅
            hop_length = 500  # number of samples per time-step in spectrogram (512) #original 1000
            n_mels = 350  # number of bins in spectrogram. Height of image
            #time_steps = int(len(temp) / 500)  # number of time-steps. Width of image (384) #original 1000
            time_steps = len(temp) / 500

            sr = 1000 #original 2000
            out = 'out.png'

            start_sample = 0  # starting at beginning
            #length_samples = time_steps * hop_length
            length_samples = int(time_steps * hop_length)
            window = temp_[start_sample:start_sample + length_samples]
            # 0.01~0.02초
            #신호를 melspectrogram으로 변환
            spectrogram_image(window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels, is_train=False, x_min=x_min, x_max=x_max)

            # 점수 계산
            image = glob.glob("C:/Users/posco/PycharmProjects/LSD_AnoGAN_pytorch/dataset/test/test/0.jpg")[0]
            img = Image.open(image)
            #trans1 = transforms.ToTensor()
            #trans2 = transforms.Normalize((0.5,), (0.5,))

            trans = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

            #image = trans2(trans1(img))
            image = trans(img)
            image = image.unsqueeze(0)

            test_image = Variable(image).cuda()
            test_size = image.size()[0]

            z = Variable(init.normal(torch.zeros(test_size, 100), mean=0, std=0.1), requires_grad=True)
            z_optimizer = torch.optim.Adam([z], lr=1e-4)

            gen_fake = generator(z.cuda())
            loss = Anomaly_score(Variable(test_image).cuda(), gen_fake)

            for j in range(5):  # original 5000
                gen_fake = generator(z.cuda())
                loss = Anomaly_score(Variable(test_image).cuda(), gen_fake, Lambda=0.01)
                loss.backward()
                z_optimizer.step()

            if cnt<=145:
                test_anomaly.append(loss.item())
                score = 0
                test_mean = np.mean(test_anomaly)
                test_std = np.std(test_anomaly)
            else:
                if (loss.item()-test_mean) / test_std>=0:
                    score = (loss.item()-test_mean) / test_std
                else:
                    score = 0
                #score = loss.item() - test_mean
            print(score)
            score = "e," + str(score)
            client_socket.send(score.encode())
            cnt = cnt+1

        print("time :", time.time() - start)
        """
        if cnt>500: #500
            # 201121
            with open(str(save_cnt) + '.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(total)
        """

    # 빈 문자열을 수신하면 루프를 중지합니다.
    if not data:
        break

    # 수신받은 문자열을 출력합니다.
    #print('Received from', addr, data.decode())

    # 받은 문자열을 다시 클라이언트로 전송해줍니다.(에코)
    #client_socket.sendall(data)


# 소켓을 닫습니다.
client_socket.close()
server_socket.close()