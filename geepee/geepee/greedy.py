import cPickle as pickle
import numpy as np
import GPy
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import os
import sys
import aep_models as aep

def get_params_SGP_old(X_train, y_train, M): 
    default_conf = {'M': M, 'hidden_size': [], 
			'optimizer':'adam', 'max_it':1000,
            'MB': 250, 'lr': 0.01, 'fixed_hyp': []}
    print 'Perform sparse GP regression'
    model_sgp = aep.SDGPR(X_train, y_train, M,
                      hidden_sizes=default_conf['hidden_size'], lik='Gaussian')
    param = model_sgp.optimise(method='adam', adam_lr=default_conf['lr'], 
                               mb_size=default_conf['MB'], maxiter=default_conf['max_it'],
                               disp=True, return_cost=False, 
                               reinit_hypers=True)
                               # reinit_hypers='fixed_zu')
    # import pdb; pdb.set_trace()
    return param

def get_params_SGP(X_train, y_train, M): 
    default_conf = {'M': M, 'hidden_size': [], 
      'optimizer':'adam', 'max_it':1000,
            'MB': 250, 'lr': 0.01, 'fixed_hyp': []}
    print 'Perform sparse GP regression'
    model_sgp = GPy.models.SparseGPRegression(X_train,y_train, kernel=GPy.kern.RBF(input_dim=1),num_inducing=M)
    model_sgp.optimize('bfgs',messages=False)


    param = {'zu_0':model_sgp.Z.values, 'sf_0':model_sgp.kern.variance.values,
            'ls_0': model_sgp.kern.lengthscale.values, 'sn': model_sgp.likelihood.variance.values}

    # import pdb; pdb.set_trace()
    return param