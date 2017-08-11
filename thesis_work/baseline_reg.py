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

# path = 'thesis_work/data/sample_' + str(sys.argv[2]) + '.txt'
path = 'thesis_work/data/dgp_samples/new/sample_' + str(sys.argv[2]) + '.txt'
# main_folder = 'thesis_work/scores/'
main_folder = 'thesis_work/scores/'
X_train, y_train, X_test, y_test = load_data(path, N_train=N_train, test=0.4, norm_out=False)
# X_test, y_test = delete_blocks(X_test, y_test,
#                                intervals=[2,4,8,9])
M = 60
## BASELINE MODELS:
model_gp, df_gp = full_GP_reg(X_train, y_train, X_test, y_test)
save_df(main_folder, df_gp, name='baseline'+ str(sys.argv[2]))

model_sgp, df_sgp = sparse_GP_reg(X_train, y_train, X_test, y_test, M=M)
save_df(main_folder, df_sgp, name='baseline'+ str(sys.argv[2]))
