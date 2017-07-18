from scipy.stats import norm
import numpy as np
import pdb
from deepbo.acquisition.base_acquisition import BaseAcquisition

class Exploration(BaseAcquisition):
	def __init__(self, model):
		super(Exploration, self).__init__(model)

	def evaluate(self, x, grad=False, **kwargs):
		x = x.reshape([1, self.model.input_dim])
		if grad:
			m, v, dmdx, dvdx = self.model.predict(x, grad=True)
		else:
			m, v = self.model.predict(x, grad=False)
		m = m[0, 0]
		v = v[0, 0]

		f = v

		if grad:
			df = dvdx
			df = df.reshape(x.shape)
			return f, df, m, v
		else:
			return f, m, v
