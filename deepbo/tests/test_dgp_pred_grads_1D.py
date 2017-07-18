import math
import numpy as np
from .context import AEPDGP_net
np.random.seed(42)
import pdb
import unittest
from scipy.optimize import check_grad

class DeepGPPredictionGrad(unittest.TestCase):

	def test_1D_one_layer(self):

		# def func(x):
		#     y = x.copy()
		#     y[y < 0.0] = 0.0
		#     y[y > 0.0] = 1.0
		#     return y + 0.05*np.random.randn(x.shape[0], 1)

		def func(x):
		    y = np.sin(10*x) / (10*x + 1e-5)
		    return y + 0.01*np.random.randn(x.shape[0], 1)

		# number of GP layers
		nolayers = 1
		# number of hidden dimension in intermediate hidden layers
		n_hiddens = []
		# number of inducing points per layer

		M = 20
		notrain = 5
		n_pseudos = [np.min([notrain, M])] + [M for _ in range(nolayers-1)]
		no_epochs = 10
		no_points_per_mb = 50

		X_train = np.reshape(np.linspace(-1, 1, notrain), (notrain, 1))
		y_train = func(X_train)
		y_train = np.reshape(y_train, (notrain, 1))

		# We construct the network
		net = AEPDGP_net.AEPDGP_net(X_train, y_train, n_hiddens, n_pseudos)
		# train
		test_nll, test_rms, energy = net.train(no_epochs=no_epochs,
		                               no_points_per_mb=no_points_per_mb,
		                               lrate=0.01, compute_logZ=True, fixed_params=['sn'])

		# We make predictions for the test set
		
		def m_func(x):
			m, v, dm, dv = net.predict_with_input_gradients(x)
			return m
		def m_grad(x):
			m, v, dm, dv = net.predict_with_input_gradients(x)
			return dm
		def v_func(x):
			m, v, dm, dv = net.predict_with_input_gradients(x)
			return v
		def v_grad(x):
			m, v, dm, dv = net.predict_with_input_gradients(x)
			return dv

		for i in range(10):
			x = np.random.randn()
			self.assertAlmostEqual([0], check_grad(m_func, m_grad, [x]), 4)


		for i in range(10):
			x = np.random.randn()
			self.assertAlmostEqual([0], check_grad(v_func, v_grad, [x]), 4)

	def test_1D_multiple_layers(self):

		# def func(x):
		#     y = x.copy()
		#     y[y < 0.0] = 0.0
		#     y[y > 0.0] = 1.0
		#     return y + 0.05*np.random.randn(x.shape[0], 1)

		def func(x):
		    y = np.sin(10*x) / (10*x + 1e-5)
		    return y + 0.01*np.random.randn(x.shape[0], 1)

		# number of GP layers
		nolayers = 3
		# number of hidden dimension in intermediate hidden layers
		n_hiddens = [3, 2]
		# number of inducing points per layer

		M = 20
		notrain = 20
		n_pseudos = [np.min([notrain, M])] + [M for _ in range(nolayers-1)]
		no_epochs = 10
		no_points_per_mb = 50

		X_train = np.reshape(np.linspace(-1, 1, notrain), (notrain, 1))
		y_train = func(X_train)
		y_train = np.reshape(y_train, (notrain, 1))

		# We construct the network
		net = AEPDGP_net.AEPDGP_net(X_train, y_train, n_hiddens, n_pseudos)
		# train
		test_nll, test_rms, energy = net.train(no_epochs=no_epochs,
		                               no_points_per_mb=no_points_per_mb,
		                               lrate=0.01, compute_logZ=True, fixed_params=['sn'])

		# We make predictions for the test set
		
		def m_func(x):
			m, v, dm, dv = net.predict_with_input_gradients(x)
			return m
		def m_grad(x):
			m, v, dm, dv = net.predict_with_input_gradients(x)
			return dm
		def v_func(x):
			m, v, dm, dv = net.predict_with_input_gradients(x)
			return v
		def v_grad(x):
			m, v, dm, dv = net.predict_with_input_gradients(x)
			return dv

		for i in range(10):
			x = np.random.randn()
			self.assertAlmostEqual([0], check_grad(m_func, m_grad, [x]), 4)


		for i in range(10):
			x = np.random.randn()
			self.assertAlmostEqual([0], check_grad(v_func, v_grad, [x]), 4)

if __name__ == '__main__':
    unittest.main()

