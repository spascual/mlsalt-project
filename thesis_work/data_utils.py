import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import GPy
import os
import sys
import pdb

def load_data(path, N_train, test=0.4, norm_out=False):
    X,y = np.loadtxt(path, delimiter=',', unpack=True)
    X,y = X.reshape(-1,1), y.reshape(-1,1)
    print 'Dataset size: ', X.shape[0], ' Test size: ', 0.4*X.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=42)
    N = X_train.shape[0]
    randind = np.random.permutation(N)
    print 'Training: ', N_train
    X_train, y_train = X_train[randind[0:N_train],:], y_train[randind[0:N_train],:]
    if norm_out:       
        scaler = StandardScaler().fit(y_train)
        y_train = scaler.transform(y_train)
        y_test = scaler.transform(y_test)
    return X_train, y_train, X_test, y_test

def delete_blocks(X,y, intervals = [2,4,8,9]):
	print 'Removing blocks...', intervals
	bod = (X<intervals[0]).reshape(-1,1)
	for i in range(len(intervals)-2):
	    if (i+1)%2 == 0:  
	        bod += ((X>intervals[i])&(X<intervals[i+1])).reshape(-1,1)
	bod += (X>intervals[-1]).reshape(-1,1)
	return X[bod].reshape(-1,1), y[bod].reshape(-1,1)

def start_df(model_name, N_train, M=np.NaN, hidden_size=np.NaN, optimizer=np.NaN): 
	column_names = ['exp','model', 'N_train', 'M', 'hidden_size', 'optimizer',
					'mse', 'nll','config','init',
					'outlier', 'time']
	df = pd.DataFrame(columns=column_names)
	df['model'] = model_name
	df['N_train'] = N_train
	df['M'] = M
	df['hidden_size'] = str(hidden_size)
	df['optimizer'] = optimizer
	return df

def df_name(df): 
	conf_col = ['exp','model', 'N_train', 'M', 'optimizer','config', 'init']
	file_name = '_'.join(map(str, list(df.loc[0][conf_col]))).strip('_nan') 
	return file_name

def save_df(main_folder, df, name='default'): 
	if name=='default':
		file_name = df_name(df) + '.csv' 
	else: 
		file_name = name + '.csv' 
	if os.path.exists(main_folder + file_name): 
		with open(main_folder + file_name,'a') as f:
			df.to_csv(f, header=False)
	else:
		df.to_csv(main_folder + file_name, header=True)
	print 'Saving scores at: ', main_folder + file_name

def start_df_old(model_name, N_train, M=np.NaN, hidden_size=np.NaN, optimizer=np.NaN): 
	column_names = ['exp','model', 'N_train', 'M', 'hidden_size', 'optimizer',
					'mse', 'nll',
					'outlier', 'outlier_var', 'time', 'config']
	df = pd.DataFrame(columns=column_names)
	df['model'] = model_name
	df['N_train'] = N_train
	df['M'] = M
	df['hidden_size'] = str(hidden_size)
	df['optimizer'] = optimizer
	return df
	