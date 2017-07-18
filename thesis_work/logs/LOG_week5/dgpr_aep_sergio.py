print "importing stuff..."
import numpy as np
import pdb
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import special

from .context import aep

def load_data(data_idx):
    path = '/Users/sergiopascualdiaz/mlsalt-code/thesis_work/exp_DGP_samples/data/DGP[1]_'
    DGP_data_file_train = path + str(data_idx) + '_train.txt'
    DGP_data_file_test = path + str(data_idx) + '_test.txt'
    training_size = 1000
    X, y = np.loadtxt(DGP_data_file_train, delimiter=',', unpack=True)
    N = X.shape[0]
    list_indx = np.random.choice(range(N), training_size)
    X_train = X[list_indx, None]
    y_train = y[list_indx, None]
    X_test, y_test = np.loadtxt(DGP_data_file_test, delimiter=',', unpack=True)
    plt.figure()
    plt.plot(X_train, y_train, '+b', label='train')
    plt.plot(X_test, y_test, 'or', label='test')
    plt.legend()
    return X_train, y_train, X_test, y_test

def run_regression_DGP():
    np.random.seed(42)
    print "load dataset ..."
    data_idx = 3
    X_train, y_train, X_test, y_test = load_data(data_idx)
    def plot(m):
        xx = np.linspace(-2, 12, 200)[:, None]
        mean, var = m.predict_y(xx)
        zu = m.sgp_layers[0].zu
        mean_u, var_u = m.predict_f(zu)
        plt.figure()
        plt.plot(X_train, y_train, 'kx', mew=2)
        plt.plot(xx, mean, 'b', lw=2)
        plt.fill_between(
            xx[:, 0],
            mean[:, 0] - 2 * np.sqrt(var[:, 0]),
            mean[:, 0] + 2 * np.sqrt(var[:, 0]),
            color='blue', alpha=0.2)
        plt.errorbar(zu, mean_u, yerr=2 * np.sqrt(var_u), fmt='ro')
        plt.xlim(-2, 12)
        plt.savefig('/tmp/sergio_dgp.pdf')

        mean, var = m.predict_y(X_test.reshape([X_test.shape[0], 1]))
        mean, var = mean[:, 0], var[:, 0]
        rmse = np.mean((mean - y_test)**2)
        ll = np.mean(-0.5 * np.log(2 * np.pi * var) - 0.5 * (y_test - mean)**2 / var)
        print 'test mse=%.3f, ll=%.3f' % (rmse, ll)
    # inference
    print "create model and optimize ..."
    M = 40
    hidden_size = [2]
    model = aep.SDGPR(X_train, y_train, M, hidden_size, lik='Gaussian')
    # model.optimise(method='adam', adam_lr=0.01, mb_size=100, maxiter=2000)
    model.optimise(method='L-BFGS-B', maxiter=2000)
    plot(model)
    plt.show()
    plt.savefig('/tmp/sergio_dgpr.pdf')

def run_regression_GP():
    np.random.seed(42)
    print "load dataset ..."
    data_idx = 3
    X_train, y_train, X_test, y_test = load_data(data_idx)
    def plot(m):
        xx = np.linspace(-2, 12, 200)[:, None]
        mean, var = m.predict(xx)
        plt.figure()
        plt.plot(X_train, y_train, 'kx', mew=2)
        plt.plot(xx, mean, 'b', lw=2)
        plt.fill_between(
            xx[:, 0],
            mean[:, 0] - 2 * np.sqrt(var[:, 0]),
            mean[:, 0] + 2 * np.sqrt(var[:, 0]),
            color='blue', alpha=0.2)
        plt.xlim(-2, 12)
        plt.savefig('/tmp/sergio_gp.pdf')
        mean, var = m.predict(X_test.reshape([X_test.shape[0], 1]))
        mean, var = mean[:, 0], var[:, 0]
        rmse = np.mean((mean - y_test)**2)
        ll = np.mean(-0.5 * np.log(2 * np.pi * var) - 0.5 * (y_test - mean)**2 / var)
        print 'test mse=%.3f, ll=%.3f' % (rmse, ll)
    # inference
    print "create model and optimize ..."
    import GPy
    model = GPy.models.GPRegression(X_train, y_train, GPy.kern.RBF(input_dim=1))
    # print model
    # pdb.set_trace()
    model.Gaussian_noise.variance = 0.001
    # model.constrain_fixed('Gaussian_noise.variance',0.0025)
    model.optimize(messages=True)
    # model.unconstrain('')
    # model.optimize(messages=True)
    # model.optimize_restarts(messages=True, num_restarts=10)
    plot(model)
    plt.show()

# def run_regression_GPflow():
#     np.random.seed(42)
#     print "load dataset ..."
#     data_idx = 8
#     X_train, y_train, X_test, y_test = load_data(data_idx)
#     def plot(m):
#         xx = np.linspace(-2, 12, 200)[:, None]
#         mean, var = m.predict_y(xx)
#         plt.figure()
#         plt.plot(X_train, y_train, 'kx', mew=2)
#         plt.plot(xx, mean, 'b', lw=2)
#         plt.fill_between(
#             xx[:, 0],
#             mean[:, 0] - 2 * np.sqrt(var[:, 0]),
#             mean[:, 0] + 2 * np.sqrt(var[:, 0]),
#             color='blue', alpha=0.2)
#         plt.xlim(-2, 12)
#         mean, var = m.predict_y(X_test.reshape([X_test.shape[0], 1]))
#         mean, var = mean[:, 0], var[:, 0]
#         rmse = np.mean((mean - y_test)**2)
#         ll = np.mean(-0.5 * np.log(2 * np.pi * var) - 0.5 * (y_test - mean)**2 / var)
#         print 'test mse=%.3f, ll=%.3f' % (rmse, ll)
#     # inference
#     print "create model and optimize ..."
#     import GPflow
#     model = GPflow.gpr.GPR(X_train, y_train, GPflow.kernels.RBF(input_dim=1))
#     model.likelihood.variance.fixed = True
#     model.likelihood.variance = 0.0001
#     model.optimize(disp=True)
#     model.likelihood.variance.fixed = False
#     model.optimize(disp=True)
#     plot(model)
#     plt.show()


if __name__ == '__main__':
    run_regression_DGP()
    run_regression_GP()
