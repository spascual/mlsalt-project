import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import GPy
import testing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from context import dgpr
from context import gpr
from context import ei
from context import lbfgs_search
from context import base_task
from context import metrics
from context import aep

import time
import sys
import pdb
import os


np.random.seed(42)

def step(x, noise=False):
    y = x.copy()
    y[y < 0.0] = 0.0
    y[y > 0.0] = 1.0
    if noise: 
        y = y + 0.05 * np.random.randn(x.shape[0], 1)
    return y



def discont(x, noise=False): 
    interval = [(x>=-2.5) & (x<-1.5),
                (x>=-1.5) & (x<0),
                (x>=0) & (x<1.5),
                (x>=1.5) & (x<2.5)]
    functions = [lambda x: 0,
                 lambda x: -1.0,
                 lambda x: 2.0,
                 lambda x: 1.0]
    y = np.piecewise(x, interval, functions)
    return y
    if noise: 
        y + 0.005 * np.random.randn(x.shape[0], 1)
    return y

def piece(x):
    # interval = [x<-1.75, 
    #             (x>=-1.75) & (x<-1.0),
    #             (x>=-1.0) & (x<-0.5),
    #             x>=-0.5]
    # functions = [lambda x: -x,
    #              lambda x: +0.5*x+2,
    #              lambda x: -0.25*x+0.5,
    #              lambda x: 0.25*x]
    # y = np.piecewise(x, interval, functions)
    interval = [x<-1.75, 
                (x>=-1.75) & (x<-1.0),
                (x>=-1.0) & (x<-0.5),
                x>=-0.5]
    functions = [lambda x: -x,
                 lambda x: +x+2.5,
                 lambda x: -0.25*x,
                 lambda x: 0.25*x+2]
    y = np.piecewise(x, interval, functions)
    return y

def delete_blocks(x,y): 
    A = x<-0.75
    B = (x < 1) & (x > -0.25)
    C = x>2
    bod = (A+B+C).reshape(-1,1)
    return x[bod].reshape(-1,1), y[bod].reshape(-1,1)





print "create dataset ..."
N = 700
X = np.random.rand(N, 1) * 5 - 2.5
Y = step(X)
# Y = piece(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50, random_state=42)
X_train,y_train = delete_blocks(X_train,y_train)

# scaler = StandardScaler().fit(y_train)
# y_train = scaler.transform(y_train)
# y_test = scaler.transform(y_test)


def run_DGP(X_train, y_train, X_test, y_test, 
            M=30, hidden_size=[3,2], optimizer='adam'):
    N_train = X_train.shape[0]
    # FILL DATAFRAME
    df = testing.start_df(['DGP'], N_train)
    df['hidden_size'], df['optimizer'], df['M'] = str(hidden_size), optimizer, M
    # inference
    t0 = time.clock()
    print "create model and optimize ..."
    model = aep.SDGPR(X_train, y_train, M, hidden_size, lik='Gaussian')
    if optimizer=='adam':
        model.optimise(method='adam', adam_lr=0.02, mb_size=200, maxiter=500, disp=True)
        print "model.optimise(method='adam', adam_lr=0.02, mb_size=200, maxiter=1000, disp=False)"
    else: 
        model.optimise(method='L-BFGS-B', alpha=1, maxiter=800)
        print "model.optimise(method='L-BFGS-B', alpha=1, maxiter=800)"

    mean_aep, var_aep = model.predict_f(X_test.reshape([X_test.shape[0], 1]))
    results_aep = metrics.METRICS(y_test, mean_aep, var_aep)
    mse_aep , nll_aep = results_aep.mse(), results_aep.nll()
    df['mse'] = mse_aep
    df['nll'] = nll_aep

    fig = testing.plot_DGP(model, plot_sample=True)
    plt.title(testing.df_name(df))

    if np.max(model.sgp_layers[0].zu) > np.max(X_train):
        df['outlier'] = 'True'

        zu_ = np.max(model.sgp_layers[0].zu).reshape(-1,1)
        mean_u, var_u = model.predict_f(zu_)
        df['outlier_var'] = var_u

    t1 = time.clock()
    df['time'] = t1 - t0
    print 'test mse=%.3f, nll=%.3f' % (mse_aep, nll_aep)
    return df, fig


def run_GP(X_train, y_train, X_test, y_test): 
    N_train = X_train.shape[0]
    # FILL DATAFRAME
    df = testing.start_df(['GP'], N_train)
    # inference
    t0 = time.clock()
    print "create model and optimize ..."
    model_gp = GPy.models.GPRegression(X_train, y_train, GPy.kern.RBF(input_dim=1))
    # model_gp.optimize('scg',messages=True)
    model_gp.optimize('bfgs',messages=False)
        # optimizers = {'fmin_tnc': opt_tnc,
        # 'simplex': opt_simplex,
        # 'lbfgsb': opt_lbfgsb,
        # 'org-bfgs': opt_bfgs,
        # 'scg': opt_SCG,
        # 'adadelta':Opt_Adadelta,
        # 'rprop':RProp}

    testing.plot_GP(model_gp)
    

    mean_gp, var_gp = model_gp.predict(X_test.reshape([X_test.shape[0], 1]))
    results_gp = metrics.METRICS(y_test, mean_gp, var_gp)
    mse_gp , nll_gp = results_gp.mse(), results_gp.nll() 
    df['mse'] = mse_gp
    df['nll'] = nll_gp

    fig = testing.plot_GP(model_gp)
    plt.title(testing.df_name(df))

    t1 = time.clock()
    df['time'] = t1 - t0

    print 'test mse=%.3f, nll=%.3f' % (mse_gp, nll_gp)
    return df, fig


def run_SGP(X_train, y_train, X_test, y_test, M=25):
    N_train = X_train.shape[0]
    # FILL DATAFRAME
    df = testing.start_df(['SGP'], N_train)
    df['M'] =  M
    t0 = time.clock()
    # inference
    print "create model and optimize ..."
    # Z = np.random.rand(M,1)*10
    # model_sgp = GPy.models.SparseGPRegression(X_train,y_train, GPy.kern.RBF(input_dim=1),Z=Z)
    # model_sgp.Z.unconstrain()
    model_sgp = GPy.models.SparseGPRegression(X_train,y_train, kernel=GPy.kern.RBF(input_dim=1),num_inducing=M)
    model_sgp.optimize('bfgs',messages=True)

    testing.plot_GP(model_sgp)

    mean_sgp, var_sgp = model_sgp.predict(X_test.reshape([X_test.shape[0], 1]))
    results_sgp = metrics.METRICS(y_test, mean_sgp, var_sgp)
    mse_sgp , nll_sgp = results_sgp.mse(), results_sgp.nll() 
    df['mse'] = mse_sgp
    df['nll'] = nll_sgp

    fig = testing.plot_GP(model_sgp)
    plt.title(testing.df_name(df))

    t1 = time.clock()
    df['time'] = t1 - t0

    print 'test mse=%.3f, nll=%.3f' % (mse_sgp, nll_sgp)
    return df, fig

    

if __name__ == '__main__':


    exp_name = 'step_meeting'
    main_folder = 'thesis_work/exp_' + exp_name +'/'
    os.system('mkdir ' + main_folder )

    # Quick Run
    M = 50
    df, fig = run_DGP(X_train, y_train, X_test, y_test, M=M)
    df['exp'] = exp_name
    print main_folder
    testing.save_fig(main_folder,df,fig)
    testing.save_df(main_folder, df, use_df_name=True)



    # df = pd.DataFrame()
    # for M in [25,50,75]: 
    #     df_new, fig = run_DGP(X_train, y_train, X_test, y_test, M=M, hidden_size=[2])
    #     df_new['exp'] = exp_name
    #     testing.save_fig(main_folder,df_new,fig)
    #     df = pd.concat((df, df_new), axis=0)
    #     testing.save_df(main_folder, df, name='DGP_500it')
    #     # TWO LAYERS
    #     df_new, fig = run_DGP(X_train, y_train, X_test, y_test, M=M, hidden_size=[3,2])
    #     df_new['exp'] = exp_name
    #     testing.save_fig(main_folder,df_new,fig)
    #     df = pd.concat((df, df_new), axis=0)
    #     testing.save_df(main_folder, df, name='DGP_500it')

    # testing.save_df(main_folder, df, name='DGP_500it')



    # df, fig = run_GP(X_train, y_train, X_test, y_test)
    # df['exp'] = exp_name
    # testing.save_df(main_folder, df, name='DGP_350')
    # testing.save_fig(main_folder,df,fig)
    # for M in [25,50,75]: 
    #     df_new, fig = run_DGP(X_train, y_train, X_test, y_test, M=M, hidden_size=[2])
    #     df_new['exp'] = exp_name
    #     testing.save_fig(main_folder,df_new,fig)
    #     df = pd.concat((df, df_new), axis=0)
    #     testing.save_df(main_folder, df, name='DGP_350')
    #     # TWO LAYERS
    #     df_new, fig = run_DGP(X_train, y_train, X_test, y_test, M=M, hidden_size=[3,2])
    #     df_new['exp'] = exp_name
    #     testing.save_fig(main_folder,df_new,fig)
    #     df = pd.concat((df, df_new), axis=0)
    #     testing.save_df(main_folder, df, name='DGP_350')
    #     # SPARSE GPS
    #     df_new, fig = run_SGP(X_train, y_train, X_test, y_test, M=M)
    #     df_new['exp'] = exp_name
    #     testing.save_fig(main_folder,df_new,fig)
    #     df = pd.concat((df, df_new), axis=0)
    #     testing.save_df(main_folder, df, name='DGP_350')

    # testing.save_df(main_folder, df, name='DGP_350')









    # for M in [25,50]: 
    #     df_new, fig = run_DGP(X_train, y_train, X_test, y_test, 
    #     M=M, hidden_size=[3,2])
    #     df_new['exp'] = exp_name
    #     testing.save_fig(main_folder,df_new,fig)
    #     df = pd.concat((df, df_new), axis=0)


      
        



