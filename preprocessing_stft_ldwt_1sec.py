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
ls_convert = [ls for ls in file_ls if '.csv' in ls and 'lt30t' not in ls[-20:] and 'mt30t' not in ls and '.jpg' not in ls[-20:] and 'datetime' not in ls]
ls_convert_lt30t = [ls for ls in file_ls if '.csv' in ls and 'lt30t' in ls[-20:] and 'mt30t' not in ls and '.jpg' not in ls[-20:] and 'datetime' not in ls]
ls_convert_mt30t = [ls for ls in file_ls if '.csv' in ls and 'lt30t' not in ls[-20:] and 'mt30t' in ls and '.jpg' not in ls[-20:] and 'datetime' not in ls]

print(len(ls_convert))



### ----- Function using scipy library
def plot_stft_1sec(file_ls):    
    for idx, file in enumerate(file_ls):
        print(file)

        path_savefolder = './dataset_raw_convert/' + file.split("\\")[1] + \
            '_figure_1sec_ldwt/stft/' + file.split("\\")[2].replace('.csv', '') + \
                '/stft'

        if os.path.exists(path_savefolder):
            pass
        else:
            os.makedirs(path_savefolder)


        scaler = MinMaxScaler()

        df_tot = pd.read_csv(file, sep=',')
        strDateTime_ = df_tot['Date']
        # datetime_ = pd.to_datetime(df_tot['Date'])

        arr_ldwt = df_tot['LD_weight'].to_numpy()
        arr_sensor_scaled = scaler.fit_transform(df_tot['Sensor'].to_numpy().reshape(-1, 1))
        arr_sensor_scaled = arr_sensor_scaled.squeeze(1)
        arr_strDateTime = strDateTime_.to_numpy()
        

        ldwt_1sec_list = temporalize(arr_ldwt, 1000, 10000)
        sensor_data_1sec_list = temporalize(arr_sensor_scaled, 1000, 10000)
        datetime_list = temporalize(arr_strDateTime, 1000, 10000)

        ls_datetime_1st_element = list()

        if file.split('\\')[1][:4] == 'car1':
            vmax_value=0.0001
        elif file.split('\\')[1][:4] == 'car2':
            vmax_value=0.0001
        else:
            pass

        for idx, (seq, ld_wt, datetime_1st) in enumerate(zip(sensor_data_1sec_list, ldwt_1sec_list, datetime_list)):
            if (ld_wt >= 30).all():  ## Laddle weight 
                vmax_=vmax_value
            else:
                vmax_=vmax_value

            datetime_ = datetime_1st[0]

            f, t, Zxx = stft(
                x=seq, fs=1000, window='hann', 
                nperseg=256, noverlap=256//2, nfft=256)
            # plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=0.03, shading='gouraud')
            plt.figure(figsize=(0.59,0.59))
            plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=vmax_, shading='gouraud', cmap=cm.gray)
            plt.axis('off')
            # plt.title('STFT Magnitude')
            # plt.ylabel('Frequency [Hz]')
            # plt.xlabel('Time [sec]')
            
            plt.savefig(
                path_savefolder + '/'
                + file.split("\\")[2].replace('.csv', '')
                + '_stft_' + str(datetime_).replace(':', '_')
                + '.jpg', 
                bbox_inches='tight',
                pad_inches=0)
                
            plt.close()           

            ls_datetime_1st_element.append(datetime_)
            # plt.show()  
        
        ### ----- Save datetime list to *.csv file
        ## Field name
        field = ['Date']
        
        ## Date row of csv file
        row = ls_datetime_1st_element
    
        with open(path_savefolder + '/' + 'datetime.csv', 'w', newline='') as f:
            ## using csv.writer method from csv package
            write = csv.writer(f)

            write.writerow(field)
            write.writerow(row)

plot_stft_1sec(ls_convert)
plot_stft_1sec(ls_convert_lt30t)
plot_stft_1sec(ls_convert_mt30t)


### ----- 전체 Dataset 단일 폴더에 저장
cars = ['car1', 'car2']
ub = ['_']
cases = ['lt', 'mt']
adjacent = [cars, ub, cases]
num_of_cases = list(product(*list(adjacent)))

def join_tuple_string(strings_tuple) -> str:
    return ''.join(strings_tuple)

for idx, case in enumerate(num_of_cases):
    case_ = join_tuple_string(case)
    rootpath_total_imgs = './dataset_raw_convert/' + case_ + '_30t_figure_1sec_ldwt/stft/'
    ls_total_imgs = save_file_list(rootpath_total_imgs)
    ls_total_imgs = [file for file in ls_total_imgs if 'datetime' not in file]  ## datetime.csv 제외

    if os.path.exists(rootpath_total_imgs + 'total/stft/'):
        print('That folder already exists')
        pass
    
    else:
        os.makedirs(rootpath_total_imgs + 'total/stft/')
        print('A folder has been created!')

    for idx, file in enumerate(ls_total_imgs):
        try:
            shutil.copyfile(file, rootpath_total_imgs + 'total/stft/' + file.split('\\')[-1])
        except shutil.SameFileError:
            pass

        