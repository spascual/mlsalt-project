import numpy as np
import pdb
import matplotlib.pylab as plt
from ..optimisers.base_optimiser import BaseOptimiser

class GridSearch(BaseOptimiser):
	
	def __init__(self, objective_function, lower, upper, resolution=1000):
		self.resolution = resolution
		super(GridSearch, self).__init__(objective_function, lower, upper)


	def maximise(self):
		x = np.linspace(self.lower[0], self.upper[0], self.resolution)
		ys = np.zeros(self.resolution)
		ms = np.zeros(self.resolution)
		vs = np.zeros(self.resolution)
		for i in range(self.resolution):
			ys[i], ms[i], vs[i] = self.objective_function(x[i])

		y = np.array(ys)
		max_arg = y.argmax()
		winner = np.argwhere(y == y[max_arg])[:, 0]
		best = np.random.choice(np.array(winner))
		x_star = x[best]

		fig, axs = plt.subplots(2)
		axs[0].plot(self.objective_function.model.X, self.objective_function.model.Y, 'ko')
		axs[0].plot(x, ms, '-b')
		top = ms + 2*np.sqrt(vs)
		bottom = ms - 2*np.sqrt(vs)
		axs[0].fill_between(x, top, bottom, color='b', alpha=0.3)
		axs[0].set_xlim([self.lower[0], self.upper[0]])
		axs[1].plot(x, ys, '-b')
		axs[1].plot(x_star, y[best], 'ro')
		axs[1].set_xlim([self.lower[0], self.upper[0]])
		plt.show()

		x_star = x_star.reshape([1, self.objective_function.model.input_dim])
		return x_star, y[best]

