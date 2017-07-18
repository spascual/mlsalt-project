import os
import time
import zlib
import numpy as np
import sys
import cPickle as pickle
import json
import sys


import subprocess

processes = set()
max_processes = 10

# functions = ['branin', 'hartmann3']
# functions = ['sinone', 'sintwo', 'hartmann6']
functions = ['rosenbrock', 'goldstein_price']
# save_path = '/tmp/'
save_path = '/scratch/tdb40/deepbo_synthetic_results/'
no_trials = 100
networks = ['gp', 'dgp']
no_steps = 50

for net in networks:
	command_list = []
	for func in functions:
		for t in range(no_trials):
			 cmd = 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.bo_synthetic.bo_exp --function %s --seed %d --network %s --steps %d --save_path %s' % (func, t, net, no_steps, save_path)
			 command_list.append(cmd)
	for i, command in enumerate(command_list):
		print 'running ', command
		processes.add(subprocess.Popen(command, shell=True))
		if len(processes) >= max_processes:
			os.wait()
			processes.difference_update([
				p for p in processes if p.poll() is not None])
