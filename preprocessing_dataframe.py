# + ---------------- +
# | IMPORT LIBRARIES |
# + ---------------- +
import pandas as pd
import numpy as np
import os
import glob
import logging
from utils import save_file_list
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

pd.get_option("display.max_columns")
pd.get_option("display.max_rows", 120)


root_path = './dataset_raw'

logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s',
                    datefmt ='%m/%d %I:%M:%S %p',
                    level=logging.DEBUG,
                    filename = './logging.log')

# + ------------------- +
# | CHECK DATAFILE LIST |
# + ------------------- +
csv_ls = save_file_list(root_path)
csv_ls_tot_raw = [ls for ls in csv_ls if 'ACS' not in ls and 'DL' not in ls and 'MD' not in ls]
csv_ls_raw_car1 = [ls for ls in csv_ls_tot_raw if '_1.csv' in ls]  ## Raw signal file for #1 tundish car
csv_ls_raw_car2 = [ls for ls in csv_ls_tot_raw if '_2.csv' in ls]  ## Raw signal file for #2 tundish car

# + -------------------------- +
# | FUNCTION FOR VISUALIZATION |
# + -------------------------- +
def plot_graphs(df, folder_name='car1_fig'):
    plt.ioff()
    fig, ax = plt.subplots()        
    fig.subplots_adjust(right=0.75)

    twin1 = ax.twinx()

    p1, = ax.plot(df["Date"], df["LD_weight"], "k-", label="LD_weight")
    p2, = twin1.plot(df["Date"], df["Sensor"], "r-", label="Sensor", alpha=0.5)
    # plt.ioff()
    ax.set_xlabel("Time")        
    ax.set_ylabel("LD_weight")
    twin1.set_ylabel("Sensor")

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

    plt.savefig(f'./dataset_raw_convert/{folder_name}/' + file[14:-4] + '.jpg')
    plt.close()

# + ------------------------------------------------------------- +
# | CONVERT RAW DATA FILE TO NEW DATA FILE FOR DATA PREPROCESSING |
# + ------------------------------------------------------------- +
for idx, file in enumerate(csv_ls_tot_raw):
    cols = ["Date", "SteelCode", "Heat_num", "LD_num", "LD_count", "LD_weight", "TD_num", "TD_temp", "TD_weight", "Car_num", 
    "Sensor1", "Sensor2", "Sensor3", "Sensor4", "Sensor5", "Sensor6", "Sensor7", "Sensor8", "Sensor9", "Sensor10", 
    "Sensor11", "Sensor12", "Sensor13", "Sensor14", "Sensor15", "Sensor16", "Sensor17", "Sensor18", "Sensor19", "Sensor20", 
    "Sensor21", "Sensor22", "Sensor23", "Sensor24", "Sensor25", "Sensor26", "Sensor27", "Sensor28", "Sensor29", "Sensor30", 
    "Sensor31", "Sensor32", "Sensor33", "Sensor34", "Sensor35", "Sensor36", "Sensor37", "Sensor38", "Sensor39", "Sensor40", 
    "Sensor41", "Sensor42", "Sensor43", "Sensor44", "Sensor45", "Sensor46", "Sensor47", "Sensor48", "Sensor49", "Sensor50", 
    "Sensor51", "Sensor52", "Sensor53", "Sensor54", "Sensor55", "Sensor56", "Sensor57", "Sensor58", "Sensor59", "Sensor60", 
    "Sensor61", "Sensor62", "Sensor63", "Sensor64", "Sensor65", "Sensor66", "Sensor67", "Sensor68", "Sensor69", "Sensor70", 
    "Sensor71", "Sensor72", "Sensor73", "Sensor74", "Sensor75", "Sensor76", "Sensor77", "Sensor78", "Sensor79", "Sensor80", 
    "Sensor81", "Sensor82", "Sensor83", "Sensor84", "Sensor85", "Sensor86", "Sensor87", "Sensor88", "Sensor89", "Sensor90", 
    "Sensor91", "Sensor92", "Sensor93", "Sensor94", "Sensor95", "Sensor96", "Sensor97", "Sensor98", "Sensor99", "Sensor100"]

    df = pd.read_csv(file, encoding='euc-kr', header=1)  ## Remove the first row to load the data set into a data frame
    df.columns = cols

    df_cond = df.copy().iloc[:, :10]
    arr_cond_exp = np.repeat(np.array(df_cond), 100, 0)


    df_sensor = df.copy().iloc[:,10:]
    arr_sensor = np.array(df_sensor).reshape(-1, 1)  ## arr_sensor.shape = (len_data, 1)

    arr_tot = np.concatenate((arr_cond_exp, arr_sensor), axis=1)

    df_tot = pd.DataFrame(
        arr_tot, columns = ["Date", "SteelCode", "Heat_num", "LD_num", 
        "LD_count", "LD_weight", "TD_num", "TD_temp", "TD_weight", 
        "Car_num", "Sensor"])
    df_tot['Date'] = pd.to_datetime(df_tot['Date'], format="%Y-%m-%d %H:%M:%S")
    
    cut_idxs = df_tot[df_tot['LD_weight']<30].index  ## Index for data cutting

    if len(cut_idxs) > 0:
        cut_idx = cut_idxs[0]
        sr = 1000  ## Sampling rate (1kHz)
        time_bf_fill = sr * 60  ## 1min -> 1kHz * 60 sec = 60,000 records
        df_tot_mt_30t = df_tot[:cut_idx].reset_index().drop('index', axis=1)
        df_tot_lt_30t = df_tot[cut_idx:cut_idx + time_bf_fill].reset_index().drop('index', axis=1)

    else:
        continue

    if "_1.csv" in file:
        pass
        plot_graphs(df_tot, folder_name='car1_fig')
        plot_graphs(df_tot_lt_30t, folder_name='car1_lt_30t_fig')
        plot_graphs(df_tot_mt_30t, folder_name='car1_mt_30t_fig')

        df_tot.to_csv('./dataset_raw_convert/car1/' + file[14:], sep=',', index=None)
        df_tot_lt_30t.to_csv('./dataset_raw_convert/car1_lt_30t/' + file[14:-4] + '_lt30t.csv', sep=',', index=None)
        df_tot_mt_30t.to_csv('./dataset_raw_convert/car1_mt_30t/' + file[14:-4] + '_mt30t.csv', sep=',', index=None)

    else:
        plot_graphs(df_tot, folder_name='car2_fig')
        plot_graphs(df_tot_lt_30t, folder_name='car2_lt_30t_fig')
        plot_graphs(df_tot_mt_30t, folder_name='car2_mt_30t_fig')

        df_tot.to_csv('./dataset_raw_convert/car2/' + file[14:], sep=',', index=None)
        df_tot_lt_30t.to_csv('./dataset_raw_convert/car2_lt_30t/' + file[14:-4] + '_lt30t.csv', sep=',', index=None)
        df_tot_mt_30t.to_csv('./dataset_raw_convert/car2_mt_30t/' + file[14:-4] + '_mt30t.csv', sep=',', index=None)