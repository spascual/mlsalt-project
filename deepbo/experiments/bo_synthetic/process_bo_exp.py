import matplotlib
matplotlib.use('Agg')
import os
import time
import zlib
import numpy as np
import sys
import cPickle as pickle
import json
import sys


# functions = ['branin', 'hartmann']
functions = ['rosenbrock']
functions = ['sinone']
functions = ['branin']
save_path = '/scratch/tdb40/deepbo_synthetic_results/'
no_trials = 40
networks = ['gp', 'dgp']
no_steps = 50

res = {}
res['gp'] = np.zeros((no_trials, no_steps))
res['dgp'] = np.zeros((no_trials, no_steps))
for func in functions:
    for net in networks:
        for i in range(no_trials):
            name = save_path + 'bo_synthetic_' + func + '_' + net + '_' + str(i) + '.regret'
            res_i = np.loadtxt(name)
            res[net][i, :] = res_i

res_gp = np.log10(res['gp'])
res_dgp = np.log10(res['dgp'])

import scikits.bootstrap as boot
res_med_gp = np.zeros(no_steps)
res_low_gp = np.zeros(no_steps)
res_high_gp = np.zeros(no_steps)

res_med_dgp = np.zeros(no_steps)
res_low_dgp = np.zeros(no_steps)
res_high_dgp = np.zeros(no_steps)
for i in range(no_steps):
    print i
    res_med_gp[i] = np.median(res_gp[:, i])
    ci = boot.ci(res_gp[:, i], np.median, method='pi')
    res_low_gp[i] = ci[0]
    res_high_gp[i] = ci[1]

    res_med_dgp[i] = np.median(res_dgp[:, i])
    ci = boot.ci(res_dgp[:, i], np.median, method='pi')
    res_low_dgp[i] = ci[0]
    res_high_dgp[i] = ci[1]



import matplotlib.pylab as plt
plt.figure()
plt.errorbar(np.linspace(0, no_steps), res_med_gp, yerr=res_high_gp-res_low_gp, fmt='r-', label='GP')
plt.errorbar(np.linspace(0, no_steps), res_med_dgp, yerr=res_high_dgp-res_low_dgp, fmt='b-', label='DGP')
plt.legend()
plt.xlabel('Number of evaluations')
plt.ylabel('Log10 of median of immediate regret')
plt.savefig('/tmp/gp_dgp_bo.png')

