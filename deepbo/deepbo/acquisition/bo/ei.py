from scipy.stats import norm
import numpy as np
import pdb
from deepbo.acquisition.base_acquisition import BaseAcquisition

class EI(BaseAcquisition):
	def __init__(self, model):
		super(EI, self).__init__(model)

	def evaluate(self, x, grad=False, **kwargs):
		x = x.reshape([1, self.model.input_dim])
		if grad:
			m, v, dmdx, dvdx = self.model.predict(x, grad=True)
		else:
			m, v = self.model.predict(x, grad=False)
		loc_best, obj_best = self.model.get_best()
		s = np.sqrt(v)

		z = (obj_best - m) / s
		f = s * (z * norm.cdf(z) + norm.pdf(z))

		if grad:
			dmdx = dmdx[0, :].reshape([1, self.model.input_dim])
			dvdx = dvdx[0, :].reshape([1, self.model.input_dim])
			dsdx = dvdx / (2*s)
			df = (-dmdx * norm.cdf(z) + (dsdx * norm.pdf(z)))
			df = df.reshape(x.shape)
			return f, df, m, v
		else:
			return f, m, v
