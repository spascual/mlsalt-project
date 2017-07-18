import numpy as np

from ..base_task import BaseTask

class Rosenbrock(BaseTask):

    def __init__(self, d=3):
        self.d = d
        X_lower = np.array([-1.5, -1.5, -1.5]).reshape([3])
        X_upper = np.array([2.5, 2.5, 2.5]).reshape([3])
        opt = np.ones([1, d])
        fopt = 0.0
        super(Rosenbrock, self).__init__(X_lower, X_upper, X_opt=opt, f_opt=fopt)

    def objective_function(self, x):
        f = np.zeros([x.shape[0], 1])
        for j in range(x.shape[0]):
            for i in range(self.d - 1):
                f[j] += 100 * (x[j, i+1] - x[j, i] ** 2) ** 2
                f[j] += (x[j, i] - 1) ** 2
        return f
