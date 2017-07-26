import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from context import aep
import GPy
import os
import sys
import pdb

class GP_figures(object):
	"""docstring for GP_plots"""
	def __init__(self, model):
		super(GP_figures, self).__init__()
		self.model = model

	def plot(self):
		x_min, x_max = np.min(self.model.X)-2, np.max(self.model.X)+2
		xx = np.linspace(x_min, x_max, 500)[:, None]
		mean, var = self.model.predict(xx)
		fig = plt.figure()
		plt.plot(self.model.X, self.model.Y, 'kx', mew=1)
		plt.plot(xx, mean, 'b', lw=2)
		plt.fill_between(
		    xx[:, 0],
		    mean[:, 0] - 2 * np.sqrt(var[:, 0]),
		    mean[:, 0] + 2 * np.sqrt(var[:, 0]),
		    color='blue', alpha=0.2)
		plt.xlim(x_min, x_max)
		return fig

class SGP_figures(object):
	"""docstring for GP_plots"""
	def __init__(self, model):
		super(SGP_figures, self).__init__()
		self.model = model

	def plot(self):
		x_min, x_max = np.min(self.model.X)-2, np.max(self.model.X)+2
		xx = np.linspace(x_min, x_max, 500)[:, None]
		mean, var = self.model.predict(xx)
		fig = plt.figure()
		plt.plot(self.model.X, self.model.Y, 'kx', mew=1, alpha=0.9)
		plt.plot(xx, mean, 'b', lw=2)
		plt.fill_between(
		    xx[:, 0],
		    mean[:, 0] - 2 * np.sqrt(var[:, 0]),
		    mean[:, 0] + 2 * np.sqrt(var[:, 0]),
		    color='blue', alpha=0.2)
		zu = self.model.Z.values
		mean_u, var_u = self.model.predict(zu)
		plt.errorbar(zu, mean_u, yerr=2 * np.sqrt(var_u), fmt='ro')
		y_min = np.min(mean) - 1.5
		plt.plot(zu, y_min * np.ones(zu.shape), 'rx')
		plt.xlim(x_min, x_max)
		return fig




class DGP_figures(object):
	"""docstring for GP_plots"""
	def __init__(self, model, conf_dict):
		super(DGP_figures, self).__init__()
		self.model = model
		self.conf_dict = conf_dict

	def plot_init(self): 
		model_init = aep.SDGPR(self.model.x_train, self.model.y_train, self.conf_dict['M'], 
		                       self.conf_dict['hidden_size'], lik='Gaussian')
		model_init.set_fixed_params(self.conf_dict['fixed_hyp'])
		model_init.optimise(method='adam', maxiter=1, disp=False,reinit_hypers=self.conf_dict['init_type'] )
		x_min, x_max = np.min(self.model.x_train)-2, np.max(self.model.x_train)+2
		xx = np.linspace(x_min, x_max, 500)[:, None]
		mean, var = model_init.predict_f(xx)
		zu_init = model_init.sgp_layers[0].zu
		mean_u, var_u = model_init.predict_f(zu_init)
		fig = plt.figure()
		plt.plot(self.model.x_train, self.model.y_train, 'kx', mew=1,alpha=0.6)
		plt.plot(xx, mean, 'b', lw=2)
		plt.fill_between(
		    xx[:, 0],
		    mean[:, 0] - 2 * np.sqrt(var[:, 0]),
		    mean[:, 0] + 2 * np.sqrt(var[:, 0]),
		    color='blue', alpha=0.2)
		plt.errorbar(zu_init, mean_u, yerr=2 * np.sqrt(var_u), fmt='ro')
		y_min = np.min(self.model.y_train) - 1.5
		plt.plot(zu_init, y_min * np.ones(zu_init.shape), 'kx')
		plt.xlim(x_min, x_max)
		return fig

	def plot_sample(self):
		no_samples = 10
		x_min, x_max = np.min(self.model.x_train)-2, np.max(self.model.x_train)+2
		xx = np.linspace(x_min, x_max, 1000)[:, None]
		f_samples = self.model.sample_f(xx, no_samples)
		fig = plt.figure()
		plt.plot(self.model.x_train, self.model.y_train, 'kx', mew=1,alpha=0.6)
		for i in range(no_samples):
			plt.plot(xx, f_samples[:, :, i], linewidth=0.5, alpha=0.5)
		plt.xlim(x_min, x_max)
		return fig

	def plot_cost(self, costs): 
		intervals = self.conf_dict['max_it']/(costs.shape[0]-1)
		curve = np.concatenate((np.arange(0,self.conf_dict['max_it'], intervals).reshape(-1,1)
                       ,costs.reshape(-1,1)), axis=1)
		fig = plt.figure()
		plt.plot(curve[:,0],curve[:,1] )
		plt.ylabel('Objective')
		plt.xlabel('Iterations')
		return fig

	def plot(self):
		x_min, x_max = np.min(self.model.x_train)-2, np.max(self.model.x_train)+2
		xx = np.linspace(x_min, x_max, 500)[:, None]
		mean, var = self.model.predict_f(xx)
		zu = self.model.sgp_layers[0].zu
		mean_u, var_u = self.model.predict_f(zu)
		fig = plt.figure()
		plt.plot(self.model.x_train, self.model.y_train, 'kx', mew=1,alpha=0.6)
		plt.plot(xx, mean, 'b', lw=2)
		plt.fill_between(
		    xx[:, 0],
		    mean[:, 0] - 2 * np.sqrt(var[:, 0]),
		    mean[:, 0] + 2 * np.sqrt(var[:, 0]),
		    color='blue', alpha=0.2)
		plt.errorbar(zu, mean_u, yerr=2 * np.sqrt(var_u), fmt='ro')
		model_init = aep.SDGPR(self.model.x_train, self.model.y_train, self.conf_dict['M'], 
		                       self.conf_dict['hidden_size'], lik='Gaussian')
		model_init.optimise(method='L-BFGS-B', maxiter=0, disp=False, reinit_hypers=self.conf_dict['init_type'])
		zu_init = model_init.sgp_layers[0].zu
		y_min,y_max = np.min(self.model.y_train) - 1.5, np.max(self.model.y_train)+1.5
		plt.plot(zu, y_min * np.ones(zu.shape), 'rx')
		plt.plot(zu_init, y_max * np.ones(zu_init.shape), 'kx')
		plt.xlim(x_min, x_max)
		return fig

	def in_h(self, x): 
		m, v = self.model.sgp_layers[0].forward_prop_thru_post(x)
		return m,v

	def h_out(self, m, v): 
		m_out, v_out = self.model.sgp_layers[1].forward_prop_thru_post(m,v)
		return m_out, v_out

	def h_out2(self, z): 
		m_out, v_out = self.model.sgp_layers[1].forward_prop_thru_post(z)
		return m_out, v_out

	def in_out(self, x): 
		m, v = self.model.predict_f(x)
		return m, v

	def plot_in_h(self): 
		for layer in self.model.sgp_layers:
		    layer.update_posterior()
		x_min, x_max = np.min(self.model.x_train)-2, np.max(self.model.x_train)+2
		xx = np.linspace(x_min, x_max, 500)[:, None]
		zu = self.model.sgp_layers[0].zu
		model_init = aep.SDGPR(self.model.x_train, self.model.y_train, self.conf_dict['M'], 
		                       self.conf_dict['hidden_size'], lik='Gaussian')
		model_init.optimise(method='L-BFGS-B', maxiter=0, disp=False)
		zu_init = model_init.sgp_layers[0].zu

		mean, var = self.in_h(xx) #(500,2)2D data
		mean_train, var_train = self.in_h(self.model.x_train)
		mean_u, var_u = self.in_h(zu) #2D data

		fig1 = plt.figure()
		plt.plot(xx, mean[:,0], 'b', lw=2)
		plt.plot(self.model.x_train, mean_train[:,0], 'k+', alpha=0.8, label='proj X_tr')
		plt.fill_between(
		    xx[:, 0],
		    mean[:, 0] - 2 * np.sqrt(var[:, 0]),
		    mean[:, 0] + 2 * np.sqrt(var[:, 0]),
		    color='blue', alpha=0.2)
		plt.errorbar(zu, mean_u[:, 0], yerr=2 * np.sqrt(var_u[:, 0]), fmt='ro',label='proj z0')

		h1_min,h1_max = np.min(mean[:,0]) - 1.0, np.max(mean[:,0])+1.0
		plt.plot(zu, h1_min * np.ones(zu.shape), 'rx')
		plt.plot(zu_init, h1_max * np.ones(zu_init.shape), 'kx')
		plt.legend()
		plt.xlim(x_min, x_max)
		plt.xlabel('x - Input space')
		plt.ylabel('h1 - 1st hidden unit')

		## Figure 2
		fig2 = plt.figure()
		plt.plot(xx, mean[:,1], 'b', lw=2)
		plt.plot(self.model.x_train, mean_train[:,1], 'k+', alpha=0.8, label='proj X_tr')
		plt.fill_between(
		    xx[:, 0],
		    mean[:, 1] - 2 * np.sqrt(var[:, 1]),
		    mean[:, 1] + 2 * np.sqrt(var[:, 1]),
		    color='blue', alpha=0.2)
		plt.errorbar(zu, mean_u[:, 1], yerr=2 * np.sqrt(var_u[:, 1]), fmt='ro',label='proj z0')

		h1_min,h1_max = np.min(mean[:,1]) - 1.0, np.max(mean[:,1])+1.0
		plt.plot(zu, h1_min * np.ones(zu.shape), 'rx')
		plt.plot(zu_init, h1_max * np.ones(zu_init.shape), 'kx')
		plt.legend()
		plt.xlim(x_min, x_max)
		plt.xlabel('x - Input space')
		plt.ylabel('h2 - 2nd hidden unit')

		return fig1 ,fig2

	def plot_hidden(self):
		x_min, x_max = np.min(self.model.x_train)-2, np.max(self.model.x_train)+2
		xx = np.linspace(x_min, x_max, 500)[:, None]
		zu = self.model.sgp_layers[0].zu
		z1 = self.model.sgp_layers[1].zu
		model_init = aep.SDGPR(self.model.x_train, self.model.y_train, self.conf_dict['M'], 
		                       self.conf_dict['hidden_size'], lik='Gaussian')
		model_init.optimise(method='L-BFGS-B', maxiter=0, disp=False)
		zu_init = model_init.sgp_layers[0].zu
		z1_init = model_init.sgp_layers[1].zu

		mean, var = self.in_h( xx) #(500,2)2D data
		mean_u, var_u = self.in_h( zu) #2D data

		fig3 = plt.figure()
		plt.plot( mean[:,0], mean[:,1], 'b', lw=2, label='in_h GPs')
		plt.plot(
		    mean[:, 0] - 2 * np.sqrt(var[:, 0]),
		    mean[:, 1] - 2 * np.sqrt(var[:, 1]),
		    color='blue', alpha=0.2)
		plt.plot(
		    mean[:, 0] + 2 * np.sqrt(var[:, 0]),
		    mean[:, 1] + 2 * np.sqrt(var[:, 1]),
		    color='blue', alpha=0.2)
		plt.plot(mean_u[:, 0], mean_u[:, 1], 'ro',alpha=0.5,label='proj z0')
		for i in range(zu.shape[0]):
		    plt.plot([mean_u[i, 0] - 2 * np.sqrt(var_u[i, 0]),mean_u[i, 0] + 2 * np.sqrt(var_u[i, 0])],
		             [mean_u[i, 1] - 2 * np.sqrt(var_u[i, 1]),mean_u[i, 1] + 2 * np.sqrt(var_u[i, 1])],
		            'r', alpha=0.5)

		plt.plot(z1[:, 0], z1[:, 1], 'xk', label='opt z1')
		plt.plot(z1_init[:, 0], z1_init[:, 1], 'g.', label='init z1')
		plt.ylabel('h2 - 2nd hidden unit')
		plt.xlabel('h1 - 1st hidden unit')
		plt.legend()
		return fig3

	def plot_h_out(self): 
		x_min, x_max = np.min(self.model.x_train)-2, np.max(self.model.x_train)+2
		xx = np.linspace(x_min, x_max, 500)[:, None]
		zu = self.model.sgp_layers[0].zu
		z1 = self.model.sgp_layers[1].zu
		model_init = aep.SDGPR(self.model.x_train, self.model.y_train, self.conf_dict['M'], 
		                       self.conf_dict['hidden_size'], lik='Gaussian')
		model_init.optimise(method='L-BFGS-B', maxiter=0, disp=False)
		zu_init = model_init.sgp_layers[0].zu
		z1_init = model_init.sgp_layers[1].zu

		mean, var = self.in_h(xx) #(500,2)2D data
		mean_u, var_u = self.in_h(zu) #2D data

		mean_u_out, var_u_out = self.in_out(zu) 
		mz1, vz1 = self.h_out2(z1)
		zu1_err = 2*np.sqrt(vz1)

		fig = plt.figure()
		ax1 = fig.add_subplot(111, projection='3d')
		ax = fig.gca(projection='3d')
		x = y = np.arange(-5.0, 5.0, 0.25)
		H1, H2 = np.meshgrid(x, y)
		zs = np.array([self.h_out2(np.array([x,y]))[0]for x,y in zip(np.ravel(H1), np.ravel(H2))])
		Z = zs.reshape(H1.shape)
		ax1.plot_surface(H1, H2, Z, alpha=0.4,cmap=cm.coolwarm)
		ax.plot(z1[:,0],z1[:,1],mz1[:,0],'.b', mew=0, label='opt z1')
		for i in range(mz1.shape[0]):
		    ax.plot([z1[i,0],z1[i,0]], [z1[i,1],z1[i,1]]
		        ,[mz1[i,0] - zu1_err[i,0], mz1[i,0] + zu1_err[i,0]] ,marker='_', color='r',alpha=0.5)

		ax.plot(mean_u[:,0],mean_u[:,1],mean_u_out[:,0],'kx',alpha=0.5, label='proj z0' ) 
		ax1.set_xlabel('h1 - 1st hidden unit')
		ax1.set_ylabel('h2 - 2nd hidden unit')
		ax1.set_zlabel('y - Output space')
		ax.legend()
		return fig

def save_fig(main_folder, df, fig): 
	# main_folder = 'thesis_work/exp_' + df['exp'][0] + '/indv_exps/'
	file_name = df_name(df) + '.png'
	fig.savefig(main_folder + file_name)

