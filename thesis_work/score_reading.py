import os
import sys
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import GPy
# print os.getcwd()
# plt.switch_backend('agg')


def load_data_frames(search_path):
    os.system('ls '+search_path+'.csv > temp')
    df_tot = pd.DataFrame()
    with open('temp') as df_tot_list: 
        for df_tot_name in df_tot_list:
            df_tot_new = pd.read_csv(df_tot_name.strip(), header=0, index_col=0)
            df_tot_new['exp'] = df_tot_name.strip('.csv\n').strip('scores/') + df_tot_new['exp'].astype('str')
            df_tot = pd.concat((df_tot,df_tot_new), axis=0)
    GPs, SGPs, DGPs = df_tot[df_tot.model == 'GP'], df_tot[df_tot.model== 'SGP'], df_tot[df_tot.model == 'DGP']
    SGP_dict, DGP_dict = {}, {}
    for m in list(set(SGPs.M)):
        SGP_dict[str(int(m))] = SGPs[SGPs.M == m]
    for m in list(set(DGPs.M)):
        DGP_dict[str(int(m))] = DGPs[DGPs.M == m]
    os.system('rm temp')
    return GPs, SGP_dict, DGP_dict

def pivot_mean_std(df): 
    df_error = pd.DataFrame()
    df = df.drop_duplicates(subset=['N_train','exp'], keep='last')
    df_error['mse_mean'] = df.pivot(index='N_train', columns='exp',values='mse').mean(axis=1)
    df_error['mse_std'] = df.pivot(index='N_train', columns='exp',values='mse').std(axis=1)
    df_error['nll_mean'] = df.pivot(index='N_train', columns='exp',values='nll').mean(axis=1)
    df_error['nll_std'] = df.pivot(index='N_train', columns='exp',values='nll').std(axis=1)
    return df_error

def plot_errors(df, metric='mse', init=False): 
    avgs = pivot_mean_std(df)
    model = list(set(df.model))[0]
    if (model == 'DGP')&(init): 
        print list(set(df.init))
        label = model +' '+ str(list(set(df.init))[0])
    else: 
        label = model
    x, y, std = avgs.index.values, avgs[metric+'_mean'], avgs[metric+'_std']
    plt.errorbar(x, y,yerr=std,label=label)
    plt.legend()
    plt.title(metric.upper()+'s')
		
