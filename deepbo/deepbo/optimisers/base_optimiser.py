import numpy as np
from ..utils.utils import rng_check


class BaseOptimiser(object):

	def __init__(self, objective_function, lower, upper, rng=None):
		self.lower = lower
		self.upper = upper
		self.objective_function = objective_function
		self.rng = rng_check(rng)

	def maximize(self):
		pass