import numpy as np
import os
import argparse


from .context import gpr
from .context import dgpr
from .context import ei
from .context import lbfgs_search
from .context import branin
from .context import hartmann
from .context import sin
from .context import goldstein_price
from .context import rosenbrock

parser = argparse.ArgumentParser(description='run bayesopt experiment',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--function',
            action="store", dest="function",
            help="function name, eg. branin, hartmann", default="branin")
parser.add_argument('-p', '--save_path',
            action="store", dest="save_path",
            help="save path", default="/tmp/")
parser.add_argument('-s', '--seed', type=int,
            action="store", dest="random_seed",
            help="random seed, eg. 10", default=42)
parser.add_argument('-n', '--network',
            action="store", dest="network",
            help="type of networks,", default="gp")
parser.add_argument('-k', '--steps', type=int,
            action="store", dest="no_steps",
            help="number of steps, eg. 10", default=50)

args = parser.parse_args()

function_name = args.function.lower()
trial = args.random_seed
np.random.seed(trial)
network_type = args.network.lower()
no_steps = args.no_steps
save_path = args.save_path

if function_name == 'branin':
    task = branin.Branin()
elif function_name == 'hartmann3':
    task = hartmann.Hartmann3()
elif function_name == 'hartmann6':
    task = hartmann.Hartmann6()
elif function_name == 'sinone':
    task = sin.SinOne()
elif function_name == 'sintwo':
    task = sin.SinTwo()
elif function_name == 'goldstein_price':
    task = goldstein_price.GoldsteinPrice()
elif function_name == 'rosenbrock':
    task = rosenbrock.Rosenbrock()

# Defining the method to model the objective function
if network_type == 'gp':
    model = gpr.GPR(input_dim=task.n_dims, noise_var=1e-4)
else:
    model = dgpr.DGPR(input_dim=task.n_dims)

# The acquisition function that we optimize in order to pick a new x
acquisition_func = ei.EI(model)

# Set the method that we will use to optimize the acquisition function
maximiser = lbfgs_search.LBFGSSearch(acquisition_func, 
    task.X_lower, task.X_upper)

outname = save_path + 'bo_synthetic_' + function_name + '_' + network_type + '_' + str(trial) + '.regret'
outfile = open(outname, 'w')

np.random.seed(trial)
# Draw one random point and evaluate it to initialize BO
X = np.array([np.random.uniform(task.X_lower, task.X_upper, task.n_dims)])
Y = task.evaluate(X)
# This is the main Bayesian optimization loop
for k in xrange(no_steps):

    # Fit the model on the data we observed so far
    model.train(X, Y)

    y_best = model.get_best()[1][0]
    true_best = task.f_opt
    ir = np.abs(y_best-true_best)
    print '\ntrial %d, k=%d, IR=%.8f' % (trial, k, ir)
    outfile.write('%.8f\n' % ir)
    outfile.flush()
    os.fsync(outfile.fileno())

    # Update the acquisition function model with the retrained model
    acquisition_func.update(model)

    # Optimize the acquisition function to obtain a new point
    new_x, obj_val = maximiser.maximise()

    # Evaluate the point and add the new observation to our set of previous seen points
    new_y = task.objective_function(np.array(new_x))
    X = np.append(X, new_x, axis=0)
    Y = np.append(Y, new_y, axis=0)

outfile.close()
