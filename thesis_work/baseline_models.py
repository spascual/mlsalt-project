import numpy as np
import pandas as pd
import GPy

from context import aep
from context import load_data
from context import delete_blocks
from context import start_df
from context import save_df
from context import metrics
from thesis_work.plots import GP_figures
from thesis_work.plots import SGP_figures

import time
import sys
import os
import pdb
import matplotlib.pyplot as plt



def full_GP_reg(X_train, y_train, X_test, y_test):
	t0 = time.clock()
	print "create full GP model and optimize ..."
	model_gp = GPy.models.GPRegression(X_train, y_train, GPy.kern.RBF(input_dim=1))
	model_gp.optimize('lbfgsb',messages=False)
	    # optimizers = {'fmin_tnc': opt_tnc,
	    # 'simplex': opt_simplex, 'lbfgsb': opt_lbfgsb,'scg': opt_SCG,
	    # 'adadelta':Opt_Adadelta,'rprop':RProp}
	t1 = time.clock()
	mean_gp, var_gp = model_gp.predict(X_test)
	results_gp = metrics.METRICS(y_test, mean_gp, var_gp)
	mse_gp , nll_gp = results_gp.mse(), results_gp.nll() 
	print 'Test MSE =%.3f, NLL =%.3f' % (mse_gp, nll_gp)

	df = start_df(['GP'], N_train=X_train.shape[0])
	df['mse'], df['nll'], df['time']= mse_gp, nll_gp, t1 - t0

	return model_gp, df


def sparse_GP_reg(X_train, y_train, X_test, y_test, M=30):
	t0 = time.clock()
	print "create SGP model and optimize ..."
	# Z = np.random.rand(M,1)*10
	# model_sgp = GPy.models.SparseGPRegression(X_train,y_train, GPy.kern.RBF(input_dim=1),Z=Z)
	# model_sgp.Z.unconstrain()
	model_sgp = GPy.models.SparseGPRegression(X_train,y_train, kernel=GPy.kern.RBF(input_dim=1),num_inducing=M)
	if M %2 == 1:
		model_sgp.inference_method=GPy.inference.latent_function_inference.FITC()
	# model_sgp.inference_method=GPy.inference.latent_function_inference.DTC()
	model_sgp.optimize('bfgs',messages=False)
	t1 = time.clock()

	mean_sgp, var_sgp = model_sgp.predict(X_test)
	results_sgp = metrics.METRICS(y_test, mean_sgp, var_sgp)
	mse_sgp , nll_sgp = results_sgp.mse(), results_sgp.nll() 

	df = start_df(['SGP'], N_train=X_train.shape[0],M=M)
	df['mse'], df['nll'], df['time']= mse_sgp, nll_sgp, t1 - t0

	print 'Test MSE =%.3f, NLL =%.3f' % (mse_sgp, nll_sgp)
	return model_sgp, df

## TODO: 
# def sparse_aep_GP():


# path = 'thesis_work/exp_DGP_samples/data/DGP[1]_3.txt'
# print os.getcwd()
# X_train, y_train, X_test, y_test = load_data(path, N_train=500, test=0.4, norm_out=False)
# X_test, y_test = delete_blocks(X_test, y_test,
#                                intervals=[2,4,8,9])

# model, df = full_GP_reg(X_train, y_train, X_test, y_test)
# fig = GP_figures(model).plot()
# fig.savefig('../fig.png')
# save_df('../', df, name='scores')