import numpy as np

from ..base_task import BaseTask

class Branin(BaseTask):

	def __init__(self):
		X_lower = np.array([-5, 0]).reshape([2])
		X_upper = np.array([10, 15]).reshape([2])
		X_opt = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
		f_opt = 0.397887
		super(Branin, self).__init__(X_lower, X_upper, X_opt=X_opt, f_opt=f_opt)

	def objective_function(self, x):
		y = (x[:, 1] - (5.1 / (4 * np.pi ** 2)) * x[:, 0] ** 2 + 5 * x[:, 0] / np.pi - 6) ** 2
		y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[:, 0]) + 10
		y = np.reshape(y, [1, 1])
		return y