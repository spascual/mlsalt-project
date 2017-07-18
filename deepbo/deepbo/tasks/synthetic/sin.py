import numpy as np

from ..base_task import BaseTask


class SinOne(BaseTask):

    def __init__(self):
        """
        One dimensional sin function introduced in the paper:
            K. Kawaguchi, L. P. Kaelbling, and T. Lozano-Perez.
            Bayesian Optimization with Exponential Convergence.
            In Advances in Neural Information Processing (NIPS), 2015
            
        """
        X_lower = np.array([0])
        X_upper = np.array([1])
        opt = np.array([[0.6330131633013163]])
        fopt = np.array([[0.042926342433644127]])
        super(SinOne, self).__init__(X_lower, X_upper, X_opt=opt, f_opt=fopt)

    def objective_function(self, x):
        y = 0.5 * np.sin(13 * x) * np.sin(27 * x) + 0.5
        y = np.reshape(y, [1, 1])
        return y



class SinTwo(BaseTask):

    def __init__(self):
        """
        Two dimensional sin function introduced in the paper:
            
            K. Kawaguchi, L. P. Kaelbling, and T. Lozano-Perez.
            Bayesian Optimization with Exponential Convergence.
            In Advances in Neural Information Processing (NIPS), 2015
            
        """
        X_lower = np.array([0, 0])
        X_upper = np.array([1, 1])

        self.sin_1 = SinOne()
        opt = np.array([[self.sin_1.X_opt[0, 0], self.sin_1.X_opt[0, 0]]])
        fopt = self.sin_1.f_opt * self.sin_1.f_opt
        super(SinTwo, self).__init__(X_lower, X_upper, X_opt=opt, f_opt=fopt)

    def objective_function(self, x):
        y = self.sin_1.objective_function(x[:, np.newaxis, 0]) * \
                 self.sin_1.objective_function(x[:, np.newaxis, 1])
        y = np.reshape(y, [1, 1])
        return y
