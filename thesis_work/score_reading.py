import os
import sys
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import GPy
# print os.getcwd()
# plt.switch_backend('agg')


class PLOTS(object):
	"""docstring for PLOTS"""
	def __init__(self, result_folder, n_test_pts, result_list=range(1,11)):
		super(PLOTS, self).__init__()
		self.result_folder = result_folder
		# self.n_results =  n_results
		self.n_test_pts = n_test_pts
		self.result_list = result_list
		self.n_results =  len(self.result_list)
		
	

		
	def load_data(self, sparse=False):
		if sparse == True: 
			# col1 = mse_gp, col2 = mse_aep, col3 = nll_gp, col4 = nll_aep, col5 = mse_sgp, col6 = nll_sgp
			results = np.zeros((self.n_test_pts,6,self.n_results))
			for j in range(self.n_results):
				results_file = self.result_folder + '/DGPS_' + str(self.result_list[j]) + '.txt'
				data = pd.read_csv(results_file, sep=",", header = None, index_col=0).sort_index()
				self.N_train = np.array(data.index)
				data = np.array(pd.read_csv(results_file, sep=",", header = None, index_col=0).sort_index())
				results[:,:,j] = np.pad(data, ((0,self.n_test_pts-data.shape[0]), (0,0)),
                                                      mode='constant', constant_values=np.NaN)
			self.results = results
			self.scores = np.zeros((self.n_test_pts,6,2))
			self.scores[:,:,0], self.scores[:,:,1] = np.nanmean(results,axis=2), np.nanstd(results,axis=2)
			scores = self.scores

		else: 			
			# col1 = mse_gp, col2 = mse_aep, col3 = nll_gp, col4 = nll_aep
			results = np.zeros((self.n_test_pts,4,self.n_results))
			for j in range(self.n_results):
				results_file = self.result_folder + '/DGPS_' + str(self.result_list[j]) + '.txt'
				data = pd.read_csv(results_file, sep=",", header = None, index_col=0).sort_index()
				self.N_train = np.array(data.index)
				data = np.array(pd.read_csv(results_file, sep=",", header = None, index_col=0).sort_index())
				results[:,:,j] = np.pad(data[:,0:4], ((0,self.n_test_pts-data.shape[0]), (0,0)),
                                                      mode='constant', constant_values=np.NaN)
			self.results = results
			self.scores = np.zeros((self.n_test_pts,4,2))
			self.scores[:,:,0], self.scores[:,:,1] = np.nanmean(results,axis=2), np.nanstd(results,axis=2)
			scores = self.scores
		return scores

  	def errorbars(self,save=True, sparse=False):
  		self.load_data(sparse)
  		mse_gp , mse_aep = self.scores[:,0,:], self.scores[:,1,:]
  		nll_gp , nll_aep = self.scores[:,2,:], self.scores[:,3,:]
  		if sparse == True:
  			mse_sgp , nll_sgp = self.scores[:,4,:], self.scores[:,5,:]	
	  		fig1 = plt.figure()
	  		plt.errorbar(self.N_train, mse_gp[:,0],yerr=mse_gp[:,1],color='r',label='GP')
	  		plt.errorbar(self.N_train, mse_aep[:,0],yerr=mse_aep[:,1],color='b',label='DGP')
	  		plt.errorbar(self.N_train, mse_sgp[:,0],yerr=mse_sgp[:,1],color='g',label='SGP')
	  		plt.legend()
	  		plt.title('MSEs')
	  		plt.ylim((0,0.06))
	  		plt.show()
	  		fig2 = plt.figure()
			plt.errorbar(self.N_train, nll_gp[:,0],yerr=nll_gp[:,1],color='r',label='GP')
			plt.errorbar(self.N_train, nll_aep[:,0],yerr=nll_aep[:,1], color='b',label='DGP')
			plt.errorbar(self.N_train, nll_sgp[:,0],yerr=nll_sgp[:,1],color='g',label='SGP')
			plt.ylim((-1,-0.4))
			plt.title('NLLs')
			plt.legend()

		else: 
			fig1 = plt.figure()
	  		plt.errorbar(self.N_train, mse_gp[:,0],yerr=mse_gp[:,1],color='r',label='GP')
	  		plt.errorbar(self.N_train, mse_aep[:,0],yerr=mse_aep[:,1],
	  			color='b',label='DGP')
	  		plt.legend()
	  		plt.title('MSEs')
	  		plt.ylim((0,0.06))
	  		plt.show()
	  		fig2 = plt.figure()
			plt.errorbar(self.N_train, nll_gp[:,0],yerr=nll_gp[:,1],color='r',label='GP')
			plt.errorbar(self.N_train, nll_aep[:,0],yerr=nll_aep[:,1], color='b',label='DGP')
			plt.ylim((-1,-0.4))
			plt.title('NLLs')
			plt.legend()

		plt.show()
		if save == True: 
			fig1.savefig(self.result_folder +'/mse_scores.png')
			fig2.savefig(self.result_folder + '/nll_scores.png')

	def DGP_sample(self, idx, save=True):
		# data_file = self.result_folder.strip('result_DGPS')+'data_DGPS/DGPS_' + str(idx) + '_test.txt'
		data_file = 'data_DGPS/DGPS_' + str(idx) + '_test.txt'

		Z,u = np.loadtxt(data_file, delimiter=',', unpack=True)
		Z = Z[:,None]
		u = u[:,None]
		fig = plt.figure()
		plt.plot(Z,u,'+b')
		plt.show()
		if save == True: 
			fig.savefig(data_file.strip('txt') + 'png')

	def indiv_scores(self, idx, DGP=True, save=True,sparse=False):
		self.load_data(sparse)
		print self.result_folder + '/DGPS_' + str(idx) + '.txt'

		if DGP==True:
			self.DGP_sample(idx, save)

		i = self.result_list.index(idx)
		plt.close('all')
		f, ((ax1), (ax2)) = plt.subplots(nrows=2, ncols=1)
		# f , ax = plt.subplots(2,1)
		ax1.plot(self.N_train, self.results[:,0,i],color='r',label='GP')
		ax1.plot(self.N_train, self.results[:,1,i],color='b',label='DGP')
		
		if sparse == True: 
			ax1.plot(self.N_train, self.results[:,4,i],color='g',label='SGP')
		ax1.legend()
		ax1.set_ylim((0,0.06))
		ax1.set_xlabel('Number training pts')
		ax1.set_ylabel('MSE')

		ax2.plot(self.N_train, self.results[:,2,i],color='r',label='GP')
		ax2.plot(self.N_train, self.results[:,3,i],color='b',label='DGP')
		if sparse == True: 
			ax2.plot(self.N_train, self.results[:,5,i],color='g',label='SGP')
		ax2.set_xlabel('Number training pts')
		ax2.set_ylabel('NLL')
		ax2.set_ylim((-1,-0.4))
		ax2.legend()
		plt.tight_layout()
		
		if save == True: 
			f.savefig(self.result_folder+ '/DGPS_' + str(idx) + '.png'