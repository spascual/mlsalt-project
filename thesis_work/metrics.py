import numpy as np
import math
from sklearn.metrics import mean_squared_error as mse
import scipy.stats as stats
import pdb

class METRICS(object):
	"""docstring for METRICS"""
	def __init__(self, y_ref, mean_pred, var_pred):
		super(METRICS, self).__init__()
		self.y_ref = y_ref.reshape(-1,1)
		self.mean_pred = mean_pred.reshape(-1,1)
		self.var_pred = var_pred.reshape(-1,1)

	def mse(self): 
		mean_squared_error = np.mean((self.y_ref - self.mean_pred)**2)
		return mean_squared_error

	def nll(self): 
		nll = - np.mean( -0.5 * np.log(2 * np.pi * (self.var_pred))
					 - 0.5*(self.y_ref - self.mean_pred)**2/(self.var_pred)
					 )
		return nll

def ho_nll_samples(model, X_test, y_test, N_samples=100):
    N_test = X_test.shape[0]
    samples = model.sample_f(X_test, no_samples=N_samples).reshape(-1,N_samples)
    sn = np.exp(model.get_hypers()['sn'])
    ho_lik = 0.0
    for i in range(N_test):
        mixture = stats.norm.pdf(y_test[i], loc=samples[i,:], scale=np.sqrt(sn))
        ho_lik += np.log(np.mean(mixture))
    return -ho_lik/N_test

def nll_MLE_samples(model, X_test, y_test, N_samples=100):
	N_test = X_test.shape[0]
	samples = model.sample_f(X_test, no_samples=N_samples).reshape(-1,N_samples)
	m = np.mean(samples,axis=1).reshape(-1,1)
	v = np.var(samples,axis=1).reshape(-1,1)
	sn = np.exp(model.get_hypers()['sn'])
	nll_samples = METRICS(y_test, m, v + sn * np.ones((N_test,1))).nll()
	
	return nll_samples