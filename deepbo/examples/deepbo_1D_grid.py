import matplotlib.pyplot as plt
import numpy as np

from .context import gpr
from .context import dgpr
from .context import ei
from .context import grid_search
from .context import base_task

np.random.seed(42)

# The optimization function that we want to optimize. It gets a numpy array with shape (N,D) where N >= 1 are the number of datapoints and D are the number of features

class ExampleTask(base_task.BaseTask):
    def __init__(self):
        X_lower = np.array([0]).reshape([1, 1])
        X_upper = np.array([6]).reshape([1, 1])
        super(ExampleTask, self).__init__(X_lower, X_upper)

    def objective_function(self, x):
        y = np.sin(3 * x) * 4 * (x - 1) * (x + 2)
        y = -np.sin(8 * (x-3)) / (8 * (x-3) + 1e-5)
        return y

task = ExampleTask()

# Defining the method to model the objective function
# model = gpr.GPR(input_dim=task.n_dims, noise_var=1e-4)
model = dgpr.DGPR(input_dim=task.n_dims)

# The acquisition function that we optimize in order to pick a new x
acquisition_func = ei.EI(model)

# Set the method that we will use to optimize the acquisition function
maximiser = grid_search.GridSearch(acquisition_func, 
    task.X_lower, task.X_upper, resolution=100)

# Draw one random point and evaluate it to initialize BO
X = np.array([np.random.uniform(task.X_lower, task.X_upper, task.n_dims)])
Y = task.evaluate(X)

# This is the main Bayesian optimization loop
for i in xrange(100):

    # Fit the model on the data we observed so far
    model.train(X, Y)

    # Update the acquisition function model with the retrained model
    acquisition_func.update(model)

    # Optimize the acquisition function to obtain a new point
    new_x, obj_val = maximiser.maximise()

    # Evaluate the point and add the new observation to our set of previous seen points
    new_y = task.objective_function(np.array(new_x))
    X = np.append(X, new_x, axis=0)
    Y = np.append(Y, new_y, axis=0)