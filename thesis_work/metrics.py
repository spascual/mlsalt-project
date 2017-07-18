import numpy as np
import math
from sklearn.metrics import mean_squared_error as mse
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
		nll = - np.mean( -0.5 * np.log(2 * math.pi * (self.var_pred))
					 - 0.5*(self.y_ref - self.mean_pred)**2/(self.var_pred)
					 )
		return nll


