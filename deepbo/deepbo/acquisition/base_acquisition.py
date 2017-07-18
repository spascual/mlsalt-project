import numpy as np

class BaseAcquisition(object):
	def __init__(self, model):
		self.model = model

	def update(self, model):
		self.model = model

	def evaluate(self, x, grad=False):
		pass

	def __call__(self, x, **kwargs):
		return self.evaluate(x, **kwargs)