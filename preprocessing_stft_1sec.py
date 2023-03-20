# + ---------------- +
# | IMPORT LIBRARIES |
# + ---------------- +
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import logging
from utils import save_file_list
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import scipy
from scipy.signal import stft
import datetime
import shutil

pd.get_option("display.max_columns")
pd.get_option("display.max_rows", 120)
# plt.rcParams["figure.figsize"] = [0.28, 0.28]
plt.rcParams["figure.autolayout"] = True

root_path = './dataset_raw_convert'

logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s',
                    datefmt ='%m/%d %I:%M:%S %p',
                    level=logging.DEBUG,
                    filename = './logging.log')


# + ------------------- +
# | CHECK DATAFILE LIST |
# + ------------------- +
file_ls = save_file_list(root_path)
ls_convert = [ls for ls in file_ls if '.csv' in ls and 'lt30t' not in ls[-20:] and 'mt30t' not in ls and '.jpg' not in ls[-20:]]
ls_convert_lt30t = [ls for ls in file_ls if '.csv' in ls and 'lt30t' in ls[-20:] and 'mt30t' not in ls and '.jpg' not in ls[-20:]]
ls_convert_mt30t = [ls for ls in file_ls if '.csv' in ls and 'lt30t' not in ls[-20:] and 'mt30t' in ls and '.jpg' not in ls[-20:]]

print(len(ls_convert))

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

### ----- Function using scipy library
def plot_stft_1sec(file_ls):
    for idx, file in enumerate(file_ls):
        print(file)

        # if 'car1' in file:
        #     vmax=0.02
        # else:
        #     vmax=0.005

        path_savefolder = './dataset_raw_convert/' + file[22:].split("\\")[0] + '_figure_1sec/stft/' + file[22:].split("\\")[1].replace('.csv', '') + '/stft'

        if os.path.exists(path_savefolder):
            pass
        else:
            os.makedirs(path_savefolder)


        scaler = MinMaxScaler()

        df_tot = pd.read_csv(file, sep=',')
        datetime_ = pd.to_datetime(df_tot['Date'])
        datetime_ = datetime_[0]

        arr_sensor_scaled = scaler.fit_transform(df_tot['Sensor'].to_numpy().reshape(-1, 1))
        arr_sensor_scaled = arr_sensor_scaled.squeeze(1)
        
        sensor_data_1sec_list = temporalize(arr_sensor_scaled, 1000, 10000)

        for idx, seq in enumerate(sensor_data_1sec_list):
            f, t, Zxx = stft(
                x=seq, fs=1000, window='hann', 
                nperseg=256, noverlap=256//2, nfft=256)
            # plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=0.03, shading='gouraud')
            plt.figure(figsize=(0.59,0.59))
            plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=0.0003, shading='gouraud', cmap=cm.gray)
            plt.axis('off')
            # plt.title('STFT Magnitude')
            # plt.ylabel('Frequency [Hz]')
            # plt.xlabel('Time [sec]')


            
            plt.savefig(
                path_savefolder + '/'
                + file[22:].split("\\")[1].replace('.csv', '')
                + '_stft_00' + str(idx) + '_' + str(datetime_).replace(':', '_')
                + '.jpg', 
                bbox_inches='tight',
                pad_inches=0)
            plt.close()

            datetime_ = datetime_ + datetime.timedelta(seconds=10)
                
            # plt.show()  


plot_stft_1sec(ls_convert)
plot_stft_1sec(ls_convert_lt30t)
plot_stft_1sec(ls_convert_mt30t)


### ----- 전체 Dataset 단일 폴더에 저장
rootpath_total_imgs = './dataset_raw_convert/car1_mt_30t_figure_1sec/stft/'
ls_total_imgs = save_file_list(rootpath_total_imgs)

if os.path.exists(rootpath_total_imgs + 'total/stft/'):
    print('That folder already exists')
    pass
else:
    os.makedirs(rootpath_total_imgs + 'total/stft/')
    print('A folder has been created!')

for idx, file in enumerate(ls_total_imgs):
    shutil.copyfile(file, rootpath_total_imgs + 'total/stft/' + file[-62:])