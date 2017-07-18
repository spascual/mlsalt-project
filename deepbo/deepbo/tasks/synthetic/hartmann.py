import numpy as np

from ..base_task import BaseTask

class Hartmann6(BaseTask):

    def __init__(self):
        X_lower = np.array([0, 0, 0, 0, 0, 0]).reshape([6])
        X_upper = np.array([1, 1, 1, 1, 1, 1]).reshape([6])
        opt = np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]])
        fopt = -3.32237

        super(Hartmann6, self).__init__(X_lower, X_upper, X_opt=opt, f_opt=fopt)

        self.alpha = [1.00, 1.20, 3.00, 3.20]
        self.A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                           [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                           [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                           [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
        self.P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                    [2329, 4135, 8307, 3736, 1004, 9991],
                                    [2348, 1451, 3522, 2883, 3047, 6650],
                                    [4047, 8828, 8732, 5743, 1091, 381]])

    def objective_function(self, x):
        """6d Hartmann test function
            input bounds:  0 <= xi <= 1, i = 1..6
            global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
            min function value = -3.32237
        """

        external_sum = 0
        for i in range(4):
            internal_sum = 0
            for j in range(6):
                internal_sum = internal_sum + self.A[i, j] * (x[:, j] - self.P[i, j]) ** 2
            external_sum = external_sum + self.alpha[i] * np.exp(-internal_sum)

        return -external_sum[:, np.newaxis]


class Hartmann3(BaseTask):

    def __init__(self):

        X_lower = np.array([0, 0, 0]).reshape([3])
        X_upper = np.array([1, 1, 1]).reshape([3])
        opt = np.array([[0.114614, 0.555649, 0.852547]])
        fopt = -3.86278

        super(Hartmann3, self).__init__(X_lower, X_upper, X_opt=opt, f_opt=fopt)

        self.alpha = [1.0, 1.2, 3.0, 3.2]
        self.A = np.array([[3.0, 10.0, 30.0],
                           [0.1, 10.0, 35.0],
                           [3.0, 10.0, 30.0],
                           [0.1, 10.0, 35.0]])
        self.P = 0.0001 * np.array([[3689, 1170, 2673],
                                    [4699, 4387, 7470],
                                    [1090, 8732, 5547],
                                    [381, 5743, 8828]])

    def objective_function(self, x):

        external_sum = 0
        for i in range(4):
            internal_sum = 0
            for j in range(3):
                internal_sum = internal_sum \
                            + self.A[i, j] * (x[:, j] \
                            - self.P[i, j]) ** 2
            external_sum = external_sum + self.alpha[i] * np.exp(-internal_sum)

        return -external_sum[:, np.newaxis]