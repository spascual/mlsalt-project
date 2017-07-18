"""
Local gradient-based solver using multiple restarts.
modified from the original pybo version
"""

import numpy as np
import scipy.optimize
import pdb

import matplotlib.pylab as plt
from .inits.strategies import init_uniform
from ..optimisers.base_optimiser import BaseOptimiser

class LBFGSSearch(BaseOptimiser):

	def __init__(self, objective_function, lower, upper, no_grid=100, no_best=10, rng=None):
		self.no_grid = no_grid
		self.no_best = no_best
		super(LBFGSSearch, self).__init__(objective_function, lower, upper)
		lower = np.array(lower, ndmin=2)
		upper = np.array(upper, ndmin=2)
		self.bounds = np.concatenate([lower, upper], axis=0).T
		self.lbfgs_bounds = [(lower[0, i], upper[0, i]) for i in range(lower.shape[1])]
		self.x_grid = init_uniform(self.bounds, self.no_grid, rng)

	def maximise(self, plot=False):
		"""
		Compute the objective function on an initial grid, pick `nbest` points, and
		maximize using LBFGS from these initial points.
		"""

		f = self.objective_function
		# compute func_grad on points xgrid
		finit = [f(self.x_grid[i, :], grad=False)[0] for i in range(self.no_grid)]
		idx_sorted = np.argsort(np.array(finit).reshape((self.no_grid, )))[::-1]

		def objective(x):
			fx, gx, _, _ = f(x, grad=True)
			return -fx, -gx.ravel()

		# TODO: the following can easily be multiprocessed
		result = []
		for i in idx_sorted[:self.no_best]:
			xi = self.x_grid[i, :]
			result.append(scipy.optimize.fmin_l_bfgs_b(objective, xi, bounds=self.lbfgs_bounds)[:2])

		# loop through the results and pick out the smallest.
		xmin, fmin = result[np.argmin(_[1] for _ in result)]

		if plot:
			fig, axs = plt.subplots(2)
			axs[0].plot(self.objective_function.model.X, self.objective_function.model.Y, 'ko')
			x = np.linspace(self.lower[0], self.upper[0], 1000)
			ys = np.zeros(1000)
			ms = np.zeros(1000)
			vs = np.zeros(1000)
			for i in range(1000):
				ys[i], ms[i], vs[i] = self.objective_function(np.array(x[i]))
			axs[0].plot(x, ms, '-b')
			top = ms + 2*np.sqrt(vs)
			bottom = ms - 2*np.sqrt(vs)
			axs[0].fill_between(x, top, bottom, color='b', alpha=0.3)
			axs[0].set_xlim([self.lower[0], self.upper[0]])
			axs[1].plot(x, ys, '-b')
			axs[1].plot(xmin, -fmin, 'ro')
			axs[1].set_xlim([self.lower[0], self.upper[0]])
			plt.show()

		# return the values (negate if we're finding a max)
		xmin = xmin.reshape([1, self.objective_function.model.input_dim])
		return xmin, -fmin