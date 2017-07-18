import numpy as np
import pandas as pd
import GPy

from context import aep
from context import gpr
from context import load_data
from context import delete_blocks
from context import start_df
from context import save_df
from context import metrics
from thesis_work.plots import DGP_figures

import time
import sys
import os
import pdb
import matplotlib.pyplot as plt

default_conf = {'M': 40, 'hidden_size': [2], 
			'optimizer':'adam', 'max_it':1000,
            'MB': 250, 'lr': 0.01, 'fixed_hyp': [], 'init_type':'long_short'}

def optimize_aep(model, config, reinit=True):

	if config['optimizer']=='adam':
		param, costs = model.optimise(method='adam', adam_lr=config['lr'], mb_size=config['MB'], 
			maxiter=config['max_it'], disp=True, return_cost=True, reinit_hypers=reinit)
	else:		
		params, cost = model.optimise(method='L-BFGS-B', maxiter=config['max_it'], disp=False, 
			return_cost=True, reinit_hypers=reinit)
		# params, cost = model.optimise(method='L-BFGS-B', maxiter=0, disp=False, 
		# 	return_cost=True, reinit_hypers=reinit)
		# it_, costs = 0, [cost[0]]
		# step = 10
		# print 'L-BFGS-B iter ', it_, ' , obj ', cost[0]
		# while it_ < config['max_it']: 
		# 	params, cost = model.optimise(method='L-BFGS-B', reinit_hypers=False,
		# 									 maxiter=step, disp=False, return_cost=True)
		# 	it_ += step
		# 	costs.append(cost[0])
		# 	if it_ % 50 == 0:
		# 		print 'L-BFGS-B iter ', it_, ' , obj ', costs[-1]
		# costs.pop()
		costs = np.array(cost)
	return model, costs



def aep_DGP_reg(X_train, y_train, X_test, y_test, 
				conf_dict= default_conf,
				return_cost=False):

	t0 = time.clock()
	print "Create DGP model and optimize ..."
	model = aep.SDGPR(X_train, y_train, conf_dict['M'], conf_dict['hidden_size'], lik='Gaussian')

	model.set_fixed_params(conf_dict['fixed_hyp'])
	print 'Optimise with fixed: ', model.fixed_params
	model, costs = optimize_aep(model, config=conf_dict, reinit=conf_dict['init_type'])
	t1 = time.clock()

	mean, var = model.predict_y(X_test)
	results = metrics.METRICS(y_test, mean, var)
	mse , nll, sn = results.mse(), results.nll(), round(np.exp(model.get_hypers()['sn']),3)

	df = start_df(['DGP'], N_train=X_train.shape[0],M=conf_dict['M'], 
				hidden_size=conf_dict['hidden_size'], 
				optimizer=conf_dict['optimizer'] + '_Mxit' + str(conf_dict['max_it']))
	df['mse'], df['nll'], df['time'], df['config'], df['init'] = mse, nll, t1 - t0, sn, conf_dict['init_type']


	print 'Test MSE= %.3f, NLL= %.3f, SN= %.3f' % (mse, nll, sn)
	if return_cost:
	    return model, df, costs
	else:
	    return model, df

def cont_optimization(model, X_test, y_test, conf_dict=default_conf, new_max_it=250, return_cost=False): 
	it_0 = conf_dict['max_it']
	conf_dict['max_it'] = new_max_it
	model, costs = optimize_aep(model, config=conf_dict, reinit=False)

	mean, var = model.predict_y(X_test)
	results = metrics.METRICS(y_test, mean, var)
	mse , nll, sn = results.mse(), results.nll(), round(np.exp(model.get_hypers()['sn']),3) 

	conf_dict['max_it'] += it_0
	df = start_df(['DGP'], N_train=model.x_train.shape[0],M=conf_dict['M'], 
				hidden_size=conf_dict['hidden_size'], 
				optimizer=conf_dict['optimizer'] + '_Mxit' + str(conf_dict['max_it']))
	df['mse'], df['nll'], df['time'], df['config'] = mse, nll, 'cont.', sn
	print 'Test MSE= %.3f, NLL= %.3f, SN= %.3f' % (mse, nll, sn)
	if return_cost:
	    return model, df, costs
	else:
	    return model, df


# path = 'thesis_work/data/sample_1.txt'
# print os.getcwd()
# X_train, y_train, X_test, y_test = load_data(path, N_train=750, test=0.4, norm_out=False)
# # X_test, y_test = delete_blocks(X_test, y_test,
#                                # intervals=[2,4,8,9])

# model, df, costs = aep_DGP_reg(X_train, y_train, X_test, y_test, 
# 				conf_dict= default_conf,
# 				return_cost=True)

# fig1 = DGP_figures(model, default_conf).plot()
# plt.show()
# fig1.savefig('../fig_hidden.png')

# cont_optimization(model)

