import numpy as np


class BaseModel(object):

    def __init__(self, *args, **kwargs):
        self.X = None
        self.Y = None


    def train(self, X, Y):
        self.X = X
        self.Y = Y


    def update(self, X, Y):
        X = np.append(self.X, X, axis=0)
        Y = np.append(self.Y, Y, axis=0)
        self.train(X, Y)


    def predict(self, X, grad=False):
        raise NotImplementedError()

    def get_best(self):
        best_idx = np.argmin(self.Y)
        return self.X[best_idx, :], self.Y[best_idx]
