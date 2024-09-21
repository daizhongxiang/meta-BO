# -*- coding: utf-8 -*-

# Code adapted based on: https://github.com/fmfn/BayesianOptimization

import numpy as np
import GPy
from helper_funcs_rgpe import UtilityFunction, unique_rows, PrintLog, acq_max
import pickle
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

class BayesianOptimization_meta(object):

    def __init__(self, f, pbounds, gp_opt_schedule, gp_model, \
                 use_init=False, gp_mcmc=False, log_file=None, save_init=False, save_init_file=None, meta_functions=None, \
                 fix_gp_hypers=None, ARD=False, S=50, standardize=False, verbose=1):
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
        meta_functions: all information regarding the meta functions
        fix_gp_hypers: whether to fix the GP hyperparameters; if yes, it should take the value of the desired hyperparameter values,
            otherwise, it should be set to None.
        ARD: whether to use Automatic Relevance Determination for the GP kernels
        S: the number of posterior samples used to compute the ensemble weights in the RGPE algorithm
        standardize: whether to standardize the meta-observations of each meta-task
        verbose: verbosity
        """
        
        self.use_init = use_init

        self.ARD = ARD
        self.standardize = standardize
        
        self.S = S

        self.fix_gp_hypers = fix_gp_hypers

        self.log_file = log_file

        self.meta_functions = meta_functions
        
        self.w = self.meta_functions["w"]
        
        # Store the original dictionary
        self.pbounds = pbounds
        
        self.incumbent = None

        self.keys = list(pbounds.keys())

        # Find number of parameters
        self.dim = len(pbounds)

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
        d = len(pbounds)
        self.X = np.array([]).reshape(-1, d)
        self.Y = np.array([])
        
        self.time_started = 0

        # Counter of iterations
        self.i = 0
        
        self.gp_mcmc = gp_mcmc
        
        self.gp_model = gp_model
        # Internal GP regressor (sklearn)
        if self.gp_model == 'sklearn':
            pass
        elif self.gp_model == 'gpflow':
            pass
        elif self.gp_model == 'gpy':
            self.gp = None
            self.gp_params = None
            self.gp_opt_schedule = gp_opt_schedule
    
        self.meta_gps = []

        # Utility Function placeholder
        self.util = None

        # PrintLog object
        self.plog = PrintLog(self.keys)
        
        self.save_init = save_init
        self.save_init_file = save_init_file

        # Output dictionary
        self.res = {}
        # Output dictionary
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values':[], 'params':[], 'init_values':[], 'init_params':[], 'init':[], \
                          'eval_times':[], 'time_started':0, 'target_f_values':[], \
                          'all_ws':[], 'all_etas':[]}

        self.all_d_bars = []

        self.verbose = verbose
        
    def init(self, init_points):
        """
        Function to draw random initialization
        """
        
        # Generate random points
        self.time_started = time.time()
        self.res['all']['time_started'] = self.time_started

        l = [np.random.uniform(x[0], x[1], size=init_points)
             for x in self.bounds]

        self.init_points += list(map(list, zip(*l)))

        # Create empty list to store the new values of the function
        y_init = []

        all_init_lc = []
        fid_inits = []
        for x in self.init_points:
            print("[init point]: ", x)
            
            curr_y = self.f(x)

            self.res['all']['lc_all_init'].append(curr_y)

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
        self.plog.reset_timer()

        # Set acquisition function
        self.util = UtilityFunction(kind=acq, kappa=kappa, use_fixed_kappa=use_fixed_kappa, kappa_scale=kappa_scale, xi=xi, gp_model=self.gp_model)

        # draw random initial points
        if not self.initialized:
            if self.use_init != None:
                init = pickle.load(open(self.use_init, "rb"))
                
                print("[loaded init: {0}]".format(init["X"]))

                self.X, self.Y = init["X"], init["Y"]
                self.incumbent = np.max(self.Y)
                self.initialized = True
                self.res['all']['init'] = init
                self.res['all']['init_values'] = list(self.Y)
                
                self.time_started = time.time()
                self.res['all']['time_started'] = self.time_started

                print("Using pre-existing initializations with {0} points".format(len(self.Y)))
            else:
                if init_points > 0:
                    self.init(init_points)

        y_max = 0
        
        if self.X.shape[0] > 0: # if there is any initial points
            ur = unique_rows(self.X)

        if self.gp_model == 'sklearn':
            pass
        elif self.gp_model == 'gpflow':
            pass
        elif self.gp_model == 'gpy':
            M = len(self.meta_functions["Xs"])
            all_meta_samples = []
            for i in range(M):
                X_m = self.meta_functions["Xs"][i]
                Y_m = self.meta_functions["Ys"][i]

                if self.standardize:
                    Y_m = (Y_m - np.mean(Y_m)) / np.std(Y_m)

                if self.fix_gp_hypers is None:
                    gp_m = GPy.models.GPRegression(X_m, Y_m, GPy.kern.Matern52(input_dim=self.X.shape[1], lengthscale=0.05, ARD=self.ARD))
                else:
                    gp_m = GPy.models.GPRegression(X_m, Y_m, GPy.kern.Matern52(input_dim=self.X.shape[1], \
                            lengthscale=self.fix_gp_hypers[0], ARD=self.ARD))

                gp_m.optimize()
                self.meta_gps.append(gp_m)

            if self.X.shape[0] > 0: # if there is any initial points
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

                    self.gp_params = self.gp.parameters
                else:
                    self.gp.optimize_restarts(num_restarts = 10, messages=False)
                    self.gp_params = self.gp.parameters

                    gp_samples = None # set this flag variable to None, to indicate that MCMC is not used
                    print("---Optimized hyper: ", self.gp)


        if self.X.shape[0] > 0: # if there is any initial points
            #### derive the weights
            N_t = self.X.shape[0]
            losses_meta = np.zeros((M+1, self.S))
            for ind_1 in np.arange(N_t):
                mask = np.ones(self.X.shape[0], dtype=bool)
                mask[ind_1] = 0
                X_ = self.X[mask, :]
                Y_ = self.Y[mask]

                gp_ = self.gp.copy()
                gp_.set_XY(X=X_, Y=Y_.reshape(-1, 1))
                post_samples_target = gp_.posterior_samples_f(self.X, full_cov=True, size=self.S)
                post_samples_target = np.squeeze(post_samples_target) # shape: N_t * self.S

                for ind_2 in np.arange(N_t):
                    if ind_1 != ind_2:
                        for m in range(M):
                            post_samples_m_1 = self.meta_gps[m].posterior_samples_f(self.X[ind_1].reshape(1, -1), full_cov=False, size=self.S)
                            post_samples_m_1 = np.squeeze(post_samples_m_1)
                            post_samples_m_2 = self.meta_gps[m].posterior_samples_f(self.X[ind_2].reshape(1, -1), full_cov=False, size=self.S)
                            post_samples_m_2 = np.squeeze(post_samples_m_2)
                            for s in range(self.S):
                                losses_meta[m, s] += np.bitwise_xor(self.Y[ind_1] < self.Y[ind_2], post_samples_m_1[s] < post_samples_m_2[s])

                        # calculate the ranking loss for the target function
                        for s in range(self.S):
                            losses_meta[-1, s] += np.bitwise_xor(self.Y[ind_1] < self.Y[ind_2], post_samples_target[ind_1, s] < post_samples_target[ind_2, s])

#             print("losses_meta: ", losses_meta[:, :10])
            for m in range(M):
                if np.median(losses_meta[m, :]) > np.percentile(losses_meta[-1, :], 95):
                    losses_meta[m, :] = 1e5 # if this condition is satisfied, we artificially make the corresponding weight 0

            best_ind = np.argmin(losses_meta, axis=0)
            ws = np.zeros(M+1)
            for s in range(self.S):
                equ_inds = np.nonzero(losses_meta[:, s] == losses_meta[best_ind[s], s])[0]
                if len(equ_inds) == 1:
                    ws[best_ind[s]] += 1 / self.S
                else:
                    if M in equ_inds:
                        ws[M] += 1 / self.S
                    else:
                        ws[best_ind[s]] += 1 / self.S

            print("ws: ", ws)
            self.w = ws
            self.res['all']['all_ws'].append(self.w)

        # Finding argmax of the acquisition function.
        x_max = acq_max(ac=self.util.utility,
                        gp=self.gp,
                        meta_gps=self.meta_gps,
                        w=self.w,
                        tau=None,
                        eta=None,
                        y_max=y_max,
                        bounds=self.bounds, 
                        iteration=1,
                        gp_samples=None)

        print("x_max: ", x_max)

        if self.verbose:
            self.plog.print_header(initialization=False)
        for i in range(n_iter):
            if not self.X.shape[0] == 0: # since in the first iteration, there is nothing in "self.X"
                if np.any(np.all(self.X - x_max == 0, axis=1)):
                    x_max = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=self.bounds.shape[0])
                
            print("x_max: ", x_max)

            curr_y = self.f(x_max)

            self.Y = np.append(self.Y, curr_y)
            print("self.X.shape: ", self.X.shape)
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))

            if self.Y[-1] > y_max:
                y_max = self.Y[-1]
                self.incumbent = self.Y[-1]
            
            print("incumbent: ", np.max(self.Y))

            ur = unique_rows(self.X)
            if self.gp_model == 'sklearn':
                pass
            elif self.gp_model == 'gpflow':
                pass
            elif self.gp_model == 'gpy':
                if self.gp is None:
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

                        self.gp_params = self.gp.parameters
                    else:
                        self.gp.optimize_restarts(num_restarts = 10, messages=False)
                        self.gp_params = self.gp.parameters

                        gp_samples = None # set this flag variable to None, to indicate that MCMC is not used
                        print("---Optimized hyper: ", self.gp)

                else: # if there is any initial points
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
                        self.gp_params = self.gp.parameters
                        
                    else:
                        self.gp.optimize_restarts(num_restarts = 10, messages=False)
                        self.gp_params = self.gp.parameters

                        gp_samples = None # set this flag variable to None, to indicate that MCMC is not used

                        print("---Optimized hyper: ", self.gp)


            #### derive the weights
            print("[Calculating the weights]")
            N_t = self.X.shape[0]
            losses_meta = np.zeros((M+1, self.S))
            for ind_1 in np.arange(N_t):
                mask = np.ones(self.X.shape[0], dtype=bool)
                mask[ind_1] = 0
                X_ = self.X[mask, :]
                Y_ = self.Y[mask]

                gp_ = self.gp.copy()
                gp_.set_XY(X=X_, Y=Y_.reshape(-1, 1))
                post_samples_target = gp_.posterior_samples_f(self.X, full_cov=True, size=self.S)
                post_samples_target = np.squeeze(post_samples_target) # shape: N_t * self.S

                for ind_2 in np.arange(N_t):
                    if ind_1 != ind_2:
                        for m in range(M):
                            post_samples_m_1 = self.meta_gps[m].posterior_samples_f(self.X[ind_1].reshape(1, -1), full_cov=False, size=self.S)
                            post_samples_m_1 = np.squeeze(post_samples_m_1) # shape: 288 * self.S
                            post_samples_m_2 = self.meta_gps[m].posterior_samples_f(self.X[ind_2].reshape(1, -1), full_cov=False, size=self.S)
                            post_samples_m_2 = np.squeeze(post_samples_m_2) # shape: 288 * self.S
                            for s in range(self.S):
                                losses_meta[m, s] += np.bitwise_xor(self.Y[ind_1] < self.Y[ind_2], post_samples_m_1[s] < post_samples_m_2[s])

                        # calculate the ranking loss for the target function
                        for s in range(self.S):
                            losses_meta[-1, s] += np.bitwise_xor(self.Y[ind_1] < self.Y[ind_2], post_samples_target[ind_1, s] < post_samples_target[ind_2, s])

#             print("losses_meta: ", losses_meta[:, :10])
            for m in range(M):
                if np.median(losses_meta[m, :]) > np.percentile(losses_meta[-1, :], 95):
                    losses_meta[m, :] = 1e5 # if this condition is satisfied, we artificially make the corresponding weight 0

            best_ind = np.argmin(losses_meta, axis=0)
            ws = np.zeros(M+1)
            for s in range(self.S):
                equ_inds = np.nonzero(losses_meta[:, s] == losses_meta[best_ind[s], s])[0]
                if len(equ_inds) == 1:
                    ws[best_ind[s]] += 1 / self.S
                else:
                    if M in equ_inds:
                        ws[M] += 1 / self.S
                    else:
                        ws[best_ind[s]] += 1 / self.S

            print("ws: ", ws)
            self.w = ws
            self.res['all']['all_ws'].append(self.w)

            # Maximize acquisition function to find next probing point
            x_max = acq_max(ac=self.util.utility,
                            gp=self.gp,
                            meta_gps=self.meta_gps,
                            w=self.w,
                            tau=None,
                            eta=None,
                            y_max=y_max,
                            bounds=self.bounds,
                            iteration=i+2,
                            gp_samples=None)

            if self.verbose:
                self.plog.print_step(x_max, self.Y[-1], warning=False)

            # Keep track of total number of iterations
            self.i += 1

            self.curr_max_X = self.X[self.Y.argmax()]
            
            x_max_param = self.X[self.Y.argmax(), :-1]

            self.res['max'] = {'max_val': self.Y.max(), 'max_params': dict(zip(self.keys, x_max_param))}
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(self.X[-1])

            if self.log_file is not None:
                pickle.dump(self.res, open(self.log_file, "wb"))
            
        if self.verbose:
            self.plog.print_summary()

