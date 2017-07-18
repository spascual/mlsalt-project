import numpy as np

from ..base_task import BaseTask


class GoldsteinPrice(BaseTask):

    def __init__(self):
        X_lower = np.array([-2, -2]).reshape([2])
        X_upper = np.array([2, 2]).reshape([2])
        opt = np.array([[0, -1]])
        fopt = 3
        super(GoldsteinPrice, self).__init__(X_lower, X_upper, X_opt=opt, f_opt=fopt)

    def objective_function(self, x):
        a = x[:, 0]
        b = x[:, 1]
        y = ((1 + (a + b + 1)**2 * (19 - 14 * a + 3 * a**2 - 14 * b + 6 * a * b + 3 * b**2)) 
            * (30 + (2 * a - 3 * b)**2 * (18 - 32 * a + 12 * a**2 + 48 * b - 36 * a * b + 27 * b**2)))
        y = np.reshape(y, [1, 1])
        return y