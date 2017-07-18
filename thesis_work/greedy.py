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
from aep_config import aep_DGP_reg

import time
import sys
import os
import pdb
import matplotlib.pyplot as plt

default_conf = {'M': 40, 'hidden_size': [2], 
			'optimizer':'adam', 'max_it':1000,
            'MB': 250, 'lr': 0.01, 'fixed_hyp': []}

## Train SGP
path = 'thesis_work/data/sample_1.txt'
print os.getcwd()
X_train, y_train, X_test, y_test = load_data(path, N_train=750, test=0.4, norm_out=False)

model_sgp = aep.SDGPR(X_train, y_train, default_conf['M'], hidden_sizes=[2], lik='Gaussian')
# param = model_sgp.optimise(method='adam', adam_lr=default_conf['lr'], mb_size=default_conf['MB'], 
# 			maxiter=100, disp=True, return_cost=False, reinit_hypers=True)
param = model_sgp.optimise(method='adam', adam_lr=default_conf['lr'], mb_size=default_conf['MB'], 
			maxiter=1000, disp=True, return_cost=False, reinit_hypers='greedy')

mean, var = model_sgp.predict_y(X_test)
results = metrics.METRICS(y_test, mean, var)
mse , nll, sn = results.mse(), results.nll(), round(np.exp(model_sgp.get_hypers()['sn']),3)

print 'Test MSE= %.3f, NLL= %.3f, SN= %.3f' % (mse, nll, sn)