import numpy as np


class BaseTask(object):

	def __init__(self, X_lower, X_upper, do_scaling=False, X_opt=None, f_opt=None):
		self.X_lower = X_lower
		self.X_upper = X_upper
		self.n_dims = self.X_lower.shape[0]

		if X_opt is not None:
			self.X_opt = X_opt

		if f_opt is not None:
			self.f_opt = f_opt


		self.do_scaling = do_scaling
		if do_scaling:
			self.original_X_lower = self.X_lower
			self.original_X_upper = self.X_upper

			self.X_lower = np.zeros_like(self.X_lower)
			self.X_upper = np.ones_like(self.X_upper)


	def objective_function(self, x):
		pass


	def objective_function_test(self, x):
		pass


	def transform(self, x):
		top = x - self.original_X_lower
		bottom = self.original_X_upper - self.original_X_lower
		return top * 1.0 / bottom


	def retransform(self, x):
		de_scaled = x * (self.original_X_upper - self.original_X_lower)
		de_meaned = de_scaled + self.original_X_lower
		return de_meaned


	def evaluate(self, x):
		if self.do_scaling:
			x = self.retransform(x)
		return self.objective_function(x)


	def evaluate_test(self, x):
		if self.do_scaling:
			x = self.retransform(x)
		return self.objective_function_test(x)


