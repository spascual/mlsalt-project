import matplotlib.pyplot as plt
import numpy as np

from .context import gpr
from .context import dgpr
from .context import ei
from .context import lbfgs_search
from .context import branin

import pdb

np.random.seed(42)

task = branin.Branin()

# Defining the method to model the objective function
# model = gpr.GPR(input_dim=task.n_dims, noise_var=1e-4)
model = dgpr.DGPR(input_dim=task.n_dims)

# The acquisition function that we optimize in order to pick a new x
acquisition_func = ei.EI(model)

# Set the method that we will use to optimize the acquisition function
maximiser = lbfgs_search.LBFGSSearch(acquisition_func, 
    task.X_lower, task.X_upper)

# Draw one random point and evaluate it to initialize BO
X = np.array([np.random.uniform(task.X_lower, task.X_upper, task.n_dims)])
Y = task.evaluate(X)

# This is the main Bayesian optimization loop
for i in xrange(100):

    # Fit the model on the data we observed so far
    model.train(X, Y)

    y_best = model.get_best()[1][0]
    true_best = task.f_opt
    print '\ni=%d, IR=%.8f' % (i, np.abs(y_best-true_best))

    # Update the acquisition function model with the retrained model
    acquisition_func.update(model)

    # Optimize the acquisition function to obtain a new point
    new_x, obj_val = maximiser.maximise()

    # Evaluate the point and add the new observation to our set of previous seen points
    new_y = task.objective_function(np.array(new_x))
    X = np.append(X, new_x, axis=0)
    Y = np.append(Y, new_y, axis=0)