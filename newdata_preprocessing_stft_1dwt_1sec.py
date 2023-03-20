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
from utils import save_file_list, temporalize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
from scipy.signal import stft
import datetime
import shutil
import csv
from itertools import product



# + ------------------- +
# | CHECK DATAFILE LIST |
# + ------------------- +
rootpath = './newdataset_raw'
file_ls = save_file_list(rootpath)

aco_file_ls = [file for file in file_ls if '.txt' in file and 'Aco' in file]
vib_file_ls = [file for file in file_ls if '.txt' in file and 'Vib' in file]



### ----- Function using scipy library
def plot_stft_1sec(file_ls):    
    for idx, file in enumerate(file_ls):
        print(file)

        path_savefolder = './newdataset_stft/' + file.split("\\")[1].replace('.txt','') + '/stft'

        if os.path.exists(path_savefolder):
            pass
        else:
            os.makedirs(path_savefolder)


        scaler = MinMaxScaler()
        
        df = pd.read_csv(file, encoding='ISO-8859-1', sep="/t", header=24)
        df.columns = ['Sensor']
        # datetime_ = pd.to_datetime(df_tot['Date'])

        arr_sensor_scaled = scaler.fit_transform(df['Sensor'].to_numpy().reshape(-1, 1))
        arr_sensor_scaled = arr_sensor_scaled.squeeze(1)
        
        sensor_data_1sec_list = temporalize(arr_sensor_scaled, 2000, 1000)

        for idx, seq in enumerate(sensor_data_1sec_list):
            f, t, Zxx = stft(
                x=seq, fs=2000, window='hann', 
                nperseg=256, noverlap=256//2, nfft=256)
            plt.figure(figsize=(0.37,0.37))
            plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=0.0001, shading='gouraud', cmap=cm.gray)
            plt.axis('off')
            # plt.title('STFT Magnitude')
            # plt.ylabel('Frequency [Hz]')
            # plt.xlabel('Time [sec]')
            
            idx_ = 100000 + idx

            plt.savefig(
                path_savefolder + '/'
                + file.split("\\")[1].replace('.txt', '')
                + '_stft_' + str(idx_) + '.jpg', 

                bbox_inches='tight',
                pad_inches=0)
                
            plt.close()           

plot_stft_1sec(aco_file_ls)
plot_stft_1sec(vib_file_ls)


# ### ----- 전체 Dataset 단일 폴더에 저장
# sensor_type = ['aco', 'vib']
# ub = ['_']
# cases = ['lt', 'mt']
# adjacent = [sensor_type, ub, cases]
# num_of_cases = list(product(*list(adjacent)))

# def join_tuple_string(strings_tuple) -> str:
#     return ''.join(strings_tuple)

# for idx, case in enumerate(num_of_cases):
#     case_ = join_tuple_string(case)
#     rootpath_total_imgs = './dataset_raw_convert/' + case_ + '_30t_figure_1sec_ldwt/stft/'
#     ls_total_imgs = save_file_list(rootpath_total_imgs)
#     ls_total_imgs = [file for file in ls_total_imgs if 'datetime' not in file]  ## datetime.csv 제외

#     if os.path.exists(rootpath_total_imgs + 'total/stft/'):
#         print('That folder already exists')
#         pass
    
#     else:
#         os.makedirs(rootpath_total_imgs + 'total/stft/')
#         print('A folder has been created!')

#     for idx, file in enumerate(ls_total_imgs):
#         try:
#             shutil.copyfile(file, rootpath_total_imgs + 'total/stft/' + file.split('\\')[-1])
#         except shutil.SameFileError:
#             pass