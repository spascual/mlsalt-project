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
    print 'Perform sparse GP regression'
    model_sgp = GPy.models.SparseGPRegression(X_train,y_train, kernel=GPy.kern.RBF(input_dim=1),num_inducing=M)
    if M % 2 == 1:
      model_sgp.inference_method=GPy.inference.latent_function_inference.FITC()
    model_sgp.optimize('bfgs',messages=False)

    JITTER = 1e-5
    zu = model_sgp.Z.values
    mu, Su = model_sgp.predict(zu, full_cov=True)
    Kuu = model_sgp.kern.K(zu)
    Kuu += np.diag(JITTER * np.ones((M, )))
    Kuuinv = np.linalg.inv(Kuu)
    # WRITTING PARAMS IN THE RIGHT SHAPE
    Dout = 2
    eta1_R = np.zeros((Dout, M * (M + 1) / 2))
    eta2 = np.zeros((Dout, M))
    for d in range(Dout):
      Suinv = np.linalg.inv(Su+  np.diag(1e-3 * np.random.rand(M)))
      theta1 = Suinv - Kuuinv
      # val, V = np.linalg.eig(theta1)
      # D = np.diag(val)
      # D[D<0] = JITTER
      # theta1 = V.T * D * V
      if np.min(np.linalg.eigvals(theta1)) <= 0:
        print 'theta1 not positive definite'
        theta1 = Suinv 
    
      theta2 = np.dot(Suinv, mu)
      R = np.linalg.cholesky(theta1).T
      triu_ind = np.triu_indices(M)
      diag_ind = np.diag_indices(M)
      R[diag_ind] = np.log(R[diag_ind])
      eta1_d = R[triu_ind].reshape((M * (M + 1) / 2,))
      eta2_d = theta2.reshape((M,))
      eta1_R[d, :] = eta1_d
      eta2[d, :] = eta2_d

    param = {'zu_0':zu,
            'sf_0':np.log(model_sgp.kern.variance.values),
            'ls_0': np.log(model_sgp.kern.lengthscale.values),
            'eta2_0': eta2, 'eta1_R_0' : eta1_R,
            # 'theta1_0' : theta1_0, 
            'sn': np.log(model_sgp.likelihood.variance.values)}

    # import pdb; pdb.set_trace()
    return param