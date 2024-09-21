# -*- coding: utf-8 -*-

# Code adapted based on: https://github.com/fmfn/BayesianOptimization

import numpy as np
import GPy
from helper_funcs import UtilityFunction, unique_rows, PrintLog, acq_max
import pickle
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

class BayesianOptimization(object):

    def __init__(self, f, pbounds, gp_opt_schedule, gp_model, \
                 use_init=False, gp_mcmc=False, log_file=None, save_init=False, save_init_file=None, \
                 fix_gp_hypers=None, ARD=False, verbose=1):
        """
        f: the objective function
        pbounds: a dictionary containing the input parameters x; each key is the name of a parameter, 
            and the corresponding value is a tuple in the form (lower bound of search space, upper_bound of search space);
            in the current implementation, please use (0, 1) all values of the dictionary
        gp_opt_schedule: we update the GP hyper-parameters via maximum likelihood every "gp_opt_schedule" iterations
        gp_model: which GP library to use; only gp_model == 'gpy' is implemented in the current version
        use_init: whether to use existing initializations
        gp_mcmc: whether to use MCMC for the inference of the GP hyperparameters
        log_file: the log file in which the results are saved
        save_init: Boolean; whether to save the initialization
        save_init_file: the file name under which to save the initializations; only used if save_init==True
        fix_gp_hypers: whether to fix the GP hyperparameters; if yes, it should take the value of the desired hyperparameter values,
            otherwise, it should be set to None.
        ARD: whether to use Automatic Relevance Determination for the GP kernels
        verbose: verbosity
        """
        
        self.use_init = use_init    
        self.ARD = ARD # whether to use ARD for the kernel

        self.fix_gp_hypers = fix_gp_hypers # whether to fix the GP hyperparamters, instead of learning
    
        self.log_file = log_file
        
        # Store the original dictionary
        self.pbounds = pbounds

        # Get the name of the parameters
        self.keys = list(pbounds.keys())
        # Find the number of parameters
        self.dim = len(pbounds)

        self.incumbent = None

        # Create an array with parameters bounds
        self.bounds = []
        for key in self.pbounds.keys():
            self.bounds.append(self.pbounds[key])
        self.bounds = np.asarray(self.bounds)
        
        # The function to be optimized
        self.f = f

        # Initialization flag
        self.initialized = False

        # Initialization lists --- stores starting points before process begins
        self.init_points = []
        self.x_init = []
        self.y_init = []

        # Numpy array place holders
        self.X = None
        self.Y = None
        
        self.i = 0
        
        self.gp_mcmc = gp_mcmc
        
        # the gp model to use; only GPy is implemented for now
        self.gp_model = gp_model
        if self.gp_model == 'sklearn':
            pass
        elif self.gp_model == 'gpflow':
            pass
        elif self.gp_model == 'gpy':
            self.gp = None
            self.gp_params = None
            self.gp_opt_schedule = gp_opt_schedule

        self.util = None

        self.plog = PrintLog(self.keys)
        
        self.save_init = save_init
        self.save_init_file = save_init_file

        self.res = {}
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values':[], 'params':[], 'init_values':[], 'init_params':[], 'init':[], \
                          'eval_times':[], 'time_started':0, 'target_f_values':[]}

        self.verbose = verbose
        
    def init(self, init_points):
        """
        Function to draw random initialization
        """
        
        l = [np.random.uniform(x[0], x[1], size=init_points)
             for x in self.bounds]

        self.init_points += list(map(list, zip(*l)))

        y_init = []

        for x in self.init_points:
            print("[init point]: ", x)

            curr_y = self.f(x)

            y_init.append(curr_y)

            self.res['all']['init_values'].append(curr_y)
            self.res['all']['init_params'].append(dict(zip(self.keys, x)))
            
        self.X = np.asarray(self.init_points)
        self.Y = np.asarray(y_init)

        self.incumbent = np.max(y_init)
        print(self.incumbent)
        self.initialized = True

        print("[inits: ]", self.X)
        print("[init values: ]", self.Y)

        init = {"X":self.X, "Y":self.Y}
        self.res['all']['init'] = init
        if self.save_init:
            with open("init.p", "wb") as c:
                pickle.dump(init, open(self.save_init_file, "wb"))

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 use_fixed_kappa=True,
                 kappa_scale=0.2,
                 xi=0.0,):
        """
        Main optimization method.

        Parameters
        ----------
        :param init_points:
            Number of randomly chosen points to sample the
            target function before fitting the gp.

        :param n_iter:
            Total number of times the process is to repeated. Note that
            currently this methods does not have stopping criteria (due to a
            number of reasons), therefore the total number of points to be
            sampled must be specified.

        :param acq:
            Acquisition function to be used, defaults to Upper Confidence Bound.

        Returns
        -------
        :return: Nothing
        """
        # Reset timer
        self.plog.reset_timer()

        # Set acquisition function
        self.util = UtilityFunction(kind=acq, kappa=kappa, use_fixed_kappa=use_fixed_kappa, kappa_scale=kappa_scale, xi=xi, gp_model=self.gp_model)

        # Initialize x, y and find current y_max
        if not self.initialized:
            if self.verbose:
                self.plog.print_header()

            if self.use_init != None:
                init = pickle.load(open(self.use_init, "rb"))

                print("[loaded init: {0}]".format(init["X"]))

                self.X, self.Y = init["X"], init["Y"]
                self.incumbent = np.max(self.Y)
                self.initialized = True
                self.res['all']['init'] = init
                self.res['all']['init_values'] = list(self.Y)

                print("Using pre-existing initializations with {0} points".format(len(self.Y)))
            else:
                self.init(init_points)

        y_max = self.Y.max()

        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        if self.gp_model == 'sklearn':
            pass
        elif self.gp_model == 'gpflow':
            pass
        elif self.gp_model == 'gpy':
            if self.fix_gp_hypers is None:
                self.gp = GPy.models.GPRegression(self.X[ur], self.Y[ur].reshape(-1, 1), \
                        GPy.kern.Matern52(input_dim=self.X.shape[1], lengthscale=0.05, ARD=self.ARD))
            else:
                self.gp = GPy.models.GPRegression(self.X[ur], self.Y[ur].reshape(-1, 1), \
                        GPy.kern.Matern52(input_dim=self.X.shape[1], lengthscale=self.fix_gp_hypers[0], ARD=self.ARD))

            if self.gp_mcmc:
                self.gp.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                self.gp.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                self.gp.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                print("[Running MCMC for GP hyper-parameters]")
                hmc = GPy.inference.mcmc.HMC(self.gp, stepsize=5e-2)
                gp_samples = hmc.sample(num_samples=500)[-300:] # Burnin
                gp_samples_mean = np.mean(gp_samples, axis=0)
                print("Mean of MCMC hypers: {0}".format(gp_samples_mean))

                self.gp.kern.variance.fix(gp_samples_mean[0])
                self.gp.kern.lengthscale.fix(gp_samples_mean[1])
                self.gp.likelihood.variance.fix(gp_samples_mean[2])

            else:
                if self.fix_gp_hypers is None:
                    self.gp.optimize_restarts(num_restarts = 10, messages=False)
                    self.gp_params = self.gp.parameters

                gp_samples = None # set this flag variable to None, to indicate that MCMC is not used
                print("---Optimized hyper: ", self.gp)

        x_max = acq_max(ac=self.util.utility,
                        gp=self.gp,
                        y_max=y_max,
                        bounds=self.bounds, 
                        iteration=1,
                        gp_samples=gp_samples)
        
        if self.verbose:
            self.plog.print_header(initialization=False)

        for i in range(n_iter):
            if np.any(np.all(self.X - x_max == 0, axis=1)):
                x_max = np.random.uniform(self.bounds[:, 0],
                                            self.bounds[:, 1],
                                            size=self.bounds.shape[0])

            curr_y = self.f(x_max)

            self.Y = np.append(self.Y, curr_y)
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))

            if self.Y[-1] > y_max:
                y_max = self.Y[-1]
                self.incumbent = self.Y[-1]

            ur = unique_rows(self.X)
            if self.gp_model == 'sklearn':
                pass
            elif self.gp_model == 'gpflow':
                pass
            elif self.gp_model == 'gpy':
                self.gp.set_XY(X=self.X[ur], Y=self.Y[ur].reshape(-1, 1))
            
                if i >= self.gp_opt_schedule and i % self.gp_opt_schedule == 0:
                    if self.gp_mcmc:
                        self.gp.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                        self.gp.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                        self.gp.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                        print("[Running MCMC for GP hyper-parameters]")
                        hmc = GPy.inference.mcmc.HMC(self.gp, stepsize=5e-2)
                        gp_samples = hmc.sample(num_samples=500)[-300:] # Burnin

                        gp_samples_mean = np.mean(gp_samples, axis=0)
                        print("Mean of MCMC hypers: {0}".format(gp_samples_mean))

                        self.gp.kern.variance.fix(gp_samples_mean[0])
                        self.gp.kern.lengthscale.fix(gp_samples_mean[1])
                        self.gp.likelihood.variance.fix(gp_samples_mean[2])
                        
                    else:
                        self.gp.optimize_restarts(num_restarts = 10, messages=False)
                        self.gp_params = self.gp.parameters

                        gp_samples = None # set this flag variable to None, to indicate that MCMC is not used

                        print("---Optimized hyper: ", self.gp)

            # Maximize acquisition function to find next probing point
            x_max = acq_max(ac=self.util.utility,
                            gp=self.gp,
                            y_max=y_max,
                            bounds=self.bounds, 
                            iteration=i+2,
                            gp_samples=gp_samples)

            if self.verbose:
                self.plog.print_step(x_max, self.Y[-1], warning=False)

            # Keep track of total number of iterations
            self.i += 1

            x_max_param = self.X[self.Y.argmax(), :-1]

            self.res['max'] = {'max_val': self.Y.max(), 'max_params': dict(zip(self.keys, x_max_param))}
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(self.X[-1])

            if self.log_file is not None:
                pickle.dump(self.res, open(self.log_file, "wb"))

        if self.verbose:
            self.plog.print_summary()
