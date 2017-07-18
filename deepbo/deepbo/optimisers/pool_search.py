import numpy as np
from ..optimisers.base_optimiser import BaseOptimiser
import pdb

import matplotlib.pylab as plt

class PoolSearch(BaseOptimiser):
	def __init__(self, objective_function):
		super(PoolSearch, self).__init__(objective_function, None, None)

	def maximise(self, X_pool):
		X_pool = np.array(X_pool, ndmin=2)
		no_pool = X_pool.shape[0]
		f_pool = np.zeros(no_pool)
		for i in range(no_pool):
			f_pool[i] = self.objective_function(X_pool[i, :], grad=False)[0]
		
		max_arg = f_pool.argmax()
		winner = np.argwhere(f_pool == f_pool[max_arg]).ravel()
		best = np.random.choice(np.array(winner))
		x_star = X_pool[best, :]


		fig, axs = plt.subplots(2, sharex=True)
		axs[0].plot(self.objective_function.model.X, self.objective_function.model.Y, 'ko')
		x = np.linspace(0, 6, 1000)
		ys = np.zeros(1000)
		ms = np.zeros(1000)
		vs = np.zeros(1000)
		for i in range(1000):
			ys[i], ms[i], vs[i] = self.objective_function(x[i])
		axs[0].plot(x, ms, '-b')
		top = ms + 2*np.sqrt(vs)
		bottom = ms - 2*np.sqrt(vs)
		axs[0].fill_between(x, top, bottom, color='b', alpha=0.3)
		axs[1].plot(X_pool, f_pool, 'ob')
		axs[1].plot(x_star, f_pool[best], 'ro')
		plt.show()

		return best, x_star, f_pool[best]

