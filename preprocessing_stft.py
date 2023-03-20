# + ---------------- +
# | IMPORT LIBRARIES |
# + ---------------- +
import librosa
import librosa.display
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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

pd.get_option("display.max_columns")
pd.get_option("display.max_rows", 120)
plt.rcParams["figure.figsize"] = [5, 5]
plt.rcParams["figure.autolayout"] = True

hl = 512 # number of samples per time-step in spectrogram
hi = 128 # Height of image
wi = 384 # Width of image


root_path = './dataset_raw_convert'

logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s',
                    datefmt ='%m/%d %I:%M:%S %p',
                    level=logging.DEBUG,
                    filename = './logging.log')

# + ------------------- +
# | CHECK DATAFILE LIST |
# + ------------------- +
file_ls = save_file_list(root_path)

ls_raw = [ls for ls in file_ls if '.csv' in ls and 'lt30t' not in ls[-20:] and '.jpg' not in ls[-20:]]
ls_lt_30t = [ls for ls in file_ls if 'lt30t' in ls[-20:] and '.jpg' not in ls[-20:]]

print(len(ls_raw))
print(len(ls_lt_30t))

def plot_stft(file_ls):
    for idx, file in enumerate(file_ls):
        try:     
            print(file)
            df_tot = pd.read_csv(file, sep=',')
            arr_sensor = df_tot['Sensor'].to_numpy().reshape(1, -1)[0]
            D = np.abs(librosa.stft(arr_sensor))

            """
            sr = 1000 (Sampling rate)        
            """
            librosa.display.specshow(
            librosa.amplitude_to_db(D, ref=np.max), 
            sr=1000, 
            y_axis='linear', 
            x_axis='time')
            plt.title('STFT results for sensor data')
            plt.ylim(0, 500)          
            plt.savefig(
                './dataset_raw_convert/'
                + file[22:].split("\\")[0]
                + '_fig/stft/' 
                + file[22:].split("\\")[1]
                + '_stft.jpg')
            # plt.show()  
        except librosa.util.exceptions.ParameterError as e:
            pass
    
# plot_stft(ls_raw)
plot_stft(ls_lt_30t)