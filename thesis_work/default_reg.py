import numpy as np
import pandas as pd
import GPy
import time
import sys
import os
import pdb

from context import aep
from context import load_data
from context import delete_blocks
from context import start_df
from context import save_df
from context import metrics
from thesis_work.plots import GP_figures
from thesis_work.plots import SGP_figures
from thesis_work.plots import DGP_figures

from baseline_models import full_GP_reg
from baseline_models import sparse_GP_reg
from aep_config import cont_optimization
from aep_config import aep_DGP_reg


N_train = int(sys.argv[1])
print 'Run experiment with ', N_train

path = 'thesis_work/data/sample_' + str(sys.argv[2]) + '.txt'
main_folder = 'thesis_work/scores/'
X_train, y_train, X_test, y_test = load_data(path, N_train=N_train, test=0.4, norm_out=False)
# X_test, y_test = delete_blocks(X_test, y_test,
#                                intervals=[2,4,8,9])
M = 50

config_dict0 = {'M': M, 'hidden_size': [2], 
			'optimizer':'adam', 'max_it':1500,
            'MB': 250, 'lr': 0.01, 'fixed_hyp': [], 'init_type' : True}

model_aep, df = aep_DGP_reg(X_train, y_train, X_test, y_test, 
				conf_dict= config_dict0,
				return_cost=False)

save_df(main_folder, df, name='default'+ str(sys.argv[2]))