import sys
import os

# cwd = os.getcwd()
# print cwd
if os.path.exists('/Users/thangbui/Desktop/geepee/geepee/'):
	sys.path.insert(0, '/Users/thangbui/Desktop/geepee/geepee/')
	import numpy as np
	import aep_models as aep
	from deepbo.models.base_model import BaseModel
else:
	if os.getcwd() == '/Users/sergiopascualdiaz/mlsalt-code/thesis_work':
		cwd = os.getcwd().strip('thesis_work')
		path = cwd + 'geepee/geepee/'
	else:
		cwd = os.getcwd()
		path = cwd + '/geepee/geepee/'
	print path	
	sys.path.insert(0, path)
	import aep_models as aep 
	path = cwd + 'deepbo/'
	# print path	
	sys.path.insert(0, path) 
	from deepbo.models.base_model import BaseModel
	import numpy as np

class DGPR(BaseModel):

	def __init__(self, input_dim, noise_var=None):
		self.input_dim = input_dim
		self.model = None
		if noise_var is not None:
			self.noise_var = noise_var
			self.fixed_noise = True
		else:
			self.fixed_noise = False
		
	def train(self, X, y, 
			M=30, hidden_size=[2], no_epochs=2000, max_minibatch_size=50, 
			lrate=0.02):

		no_train = X.shape[0]
		M0 = np.min([M, no_train])
		no_layers = len(hidden_size) + 1
		no_pseudos = [M0] + [M for _ in range(no_layers-1)]
		print no_pseudos
		if no_train <= max_minibatch_size:
			minibatch_size = no_train
		else:
			minibatch_size = max_minibatch_size
		self.model = aep.SDGPR(X, y, no_pseudos, hidden_size)
		if self.fixed_noise:
			self.model.lik_layer.sn = np.log(self.noise_var) / 2
			self.model.set_fixed_params('sn')

		# if no_train < 10:
		# 	for i in range(no_layers):
		# 		self.model.set_fixed_params('zu_%d'%i)

		# if minibatch_size < no_train:
		maxiter = int(np.round(no_epochs * no_train * 1.0 / minibatch_size))
		self.model.optimise(
			method='adam', maxiter=maxiter, 
			mb_size=minibatch_size, adam_lr=lrate) 
		# else:
		# 	self.model.optimise(method='L-BFGS-B', maxiter=no_epochs)

		self.X = X
		self.Y = y

	def predict(self, X, grad=False):
		if grad:
			m, v, dmx, dvx = self.model.predict_y_with_input_grad(X)
			return m[0], v[0], dmx, dvx
		else:
			m, v = self.model.predict_y(X)
			return m[0], v[0]