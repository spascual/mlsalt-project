import numpy as np
import GPy
import copy
import pdb
from ..models.base_model import BaseModel

class GPR(BaseModel):

	def __init__(self, input_dim, kernel=None, noise_var=None):
		if kernel is None:
			kernel = GPy.kern.RBF(input_dim=input_dim, ARD=True)
		self.kernel = kernel
		self.noise_var = noise_var
		self.input_dim = input_dim

		self.model = None
		self.start_point = None

	def train(self, X, y):
		kern = copy.deepcopy(self.kernel)
		self.model = GPy.models.GPRegression(X, y, kern)
		
		# fix noise
		if self.noise_var:
			self.model.likelihood.variance.fix(self.noise_var)
		
		# optimise
		# self.model.optimize(start=self.start_point)
		self.model.optimize()
		self.start_point = self.model.param_array
		self.X = X
		self.Y = y


	def predict(self, X, full_cov=False, grad=False):
		m, v = self.model.predict(X, full_cov=full_cov)
		if grad:
			dmdx, dvdx = self.model.predictive_gradients(X)
			return m, v, dmdx[:, :, 0], dvdx
		else:
			return m, v


	def get_noise(self):
		return self.model.likelihood.variance[0]


	def sample_functions(self, X, n_funcs=1):

		return self.model.posterior_samples_f(X, n_funcs)

