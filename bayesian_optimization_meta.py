# -*- coding: utf-8 -*-

# Code adapted based on: https://github.com/fmfn/BayesianOptimization

"""
This script implements the RM-GP-UCB algorithm
"""

import numpy as np
import GPy
from helper_funcs_meta import UtilityFunction, unique_rows, PrintLog, acq_max
import pickle
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

class BayesianOptimization_meta(object):

    def __init__(self, f, pbounds, gp_opt_schedule, gp_model, \
                 use_init=False, gp_mcmc=False, log_file=None, save_init=False, save_init_file=None, meta_functions=None, \
                 online_learning_rate=0.1, fix_w=False, fix_gp_hypers=None, eps=0.5, min_decay=1.0, \
                 use_adaptive_eta=True, ARD=False, standardize=False, use_max=False, verbose=1):
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
        online_learning_rate: the learning rate used for online meta-weight optimization ($\eta$)
        fix_w: whether to fix the meta-weights to the uniform distribution
        fix_gp_hypers: whether to fix the GP hyperparameters; if yes, it should take the value of the desired hyperparameter values,
            otherwise, it should be set to None.
        eps: $\epsilon$ use in adaptive tuning of $\nu_t$
        min_decay: the minmum decaying rate imposed on $\nu_t$ to ensure that it is monotinically decreasing, 
            to make sure the algorithm is asymptotically no-regret
        use_adaptive_eta: whether to use the adaptive tuning of $\nu_t$
        ARD: whether to use Automatic Relevance Determination for the GP kernels
        standardize: whether to standardize the meta-observations of each meta-task
        use_max: whether to use the "max" operator, instead of the empirical mean, in calculating the upper bound on the function gap (Lemma 1)
        verbose: verbosity
        """
        
        self.use_init = use_init
    
        # Whether to use Automatic Relevance Determination for the kernel
        self.ARD = ARD
        
        # Whether to use the "max"  or "mean"  operator when estimaing upper bounds on the function gaps (Lemma 1)
        # By default, we use mean
        self.use_max = use_max
        
        self.standardize = standardize # whether to standardize the meta observations

        self.eps = eps
        self.min_decay = min_decay
        self.use_adaptive_eta = use_adaptive_eta

        self.fix_gp_hypers = fix_gp_hypers
    
        self.fix_w = fix_w
    
        self.log_file = log_file
        
        self.online_learning_rate = online_learning_rate
        
        self.meta_functions = meta_functions
        
        self.w = self.meta_functions["w"]
        self.eta = 1.0
        
        self.pbounds = pbounds
        self.incumbent = None

        self.keys = list(pbounds.keys())

        self.dim = len(pbounds)

        self.bounds = []
        for key in self.pbounds.keys():
            self.bounds.append(self.pbounds[key])
        self.bounds = np.asarray(self.bounds)
        
        # The function to be optimized
        self.f = f

        self.initialized = False

        self.init_points = []
        self.x_init = []
        self.y_init = []

        d = len(pbounds)
        self.X = np.array([]).reshape(-1, d)
        self.Y = np.array([])

        self.i = 0
        
        self.gp_mcmc = gp_mcmc
        
        self.gp_model = gp_model
        if self.gp_model == 'sklearn':
            pass
        elif self.gp_model == 'gpflow':
            pass
        elif self.gp_model == 'gpy':
            # for now, only GPy is implemented
            self.gp = None
            self.gp_params = None
            self.gp_opt_schedule = gp_opt_schedule
    
        self.meta_gps = []

        self.util = None

        self.plog = PrintLog(self.keys)

        self.save_init = save_init
        self.save_init_file = save_init_file

        self.res = {}
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values':[], 'params':[], 'init_values':[], 'init_params':[], 'init':[], 'target_f_values':[], 'all_ws':[], 'all_etas':[]}

        self.all_d_bars = []

        self.verbose = verbose
        
    def init(self, init_points):
        """
        Function to draw random initialization
        """
        
        l = [np.random.uniform(x[0], x[1], size=init_points)
             for x in self.bounds]

        self.init_points += list(map(list, zip(*l)))

        y_init = []

        all_init_lc = []
        fid_inits = []
        for x in self.init_points:
            print("[init point]: ", x)

            curr_y = self.f(x)

            y_init.append(curr_y) # we need to negate the error because the program assumes maximization

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
            # At this point, only GPy is implemented
            M = len(self.meta_functions["Xs"])
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

            if self.X.shape[0] > 0: # if there is any initial points, then fit the target surrogate function
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
            ####### update the weights w's
            iteration = 1
            d = self.X.shape[1]
            if use_fixed_kappa:
                beta_t = np.sqrt(kappa)
            else:
                beta_t = np.sqrt(kappa_scale * d * np.log(2 * iteration))

            M = len(self.meta_functions["Xs"])
            d_bars = []
            for k in range(M):
                meta_inputs = self.meta_functions["Xs"][k]
                meta_outputs = self.meta_functions["Ys"][k]

                if self.standardize:
                    meta_outputs = (meta_outputs - np.mean(meta_outputs)) / np.std(meta_outputs)

                N_meta_inputs = self.meta_functions["Xs"][k].shape[0]
                upper_bounds_all = []
                for j in range(N_meta_inputs):
                    meta_x = meta_inputs[j, :]
                    meta_x = meta_x.reshape(1, -1)

                    mean_m, var_m = self.gp.predict(meta_x)
                    var_m = max(1e-6, var_m)
                    std_m = np.sqrt(var_m)

                    upper_bound = mean_m + beta_t * std_m
                    lower_bound = mean_m - beta_t * std_m

                    diff_upper = upper_bound[0, 0] - meta_outputs[j, 0]
                    diff_lower = lower_bound[0, 0] - meta_outputs[j, 0]
                    max_diff = np.max([np.abs(diff_upper), np.abs(diff_lower)])
                    upper_bounds_all.append(max_diff)
                if self.use_max:
                    d_bars.append(np.max(upper_bounds_all))
                else:
                    d_bars.append(np.mean(upper_bounds_all))

            self.all_d_bars.append(d_bars)
            all_d_bars_np = np.array(self.all_d_bars)
            tmp = np.sum(all_d_bars_np, axis=0)
            tmp = np.exp(-self.online_learning_rate * tmp)

            if not self.fix_w:
                self.w = tmp / np.sum(tmp)
            else:
                self.w = np.repeat(1.0 / M, M)

            self.res['all']['all_ws'].append(self.w)
            print("updated w: ", self.w)

            if self.use_adaptive_eta:
                d_bars_eta = []
                for k in range(M):
                    meta_inputs = self.meta_functions["Xs"][k]
                    meta_outputs = self.meta_functions["Ys"][k]

                    if self.standardize:
                        meta_outputs = (meta_outputs - np.mean(meta_outputs)) / np.std(meta_outputs)

                    N_meta_inputs = self.meta_functions["Xs"][k].shape[0]
                    upper_bounds_all = []
                    for j in range(N_meta_inputs):
                        meta_x = meta_inputs[j, :]
                        meta_x = meta_x.reshape(1, -1)

                        mean_m, var_m = self.gp.predict(meta_x)
                        var_m = max(1e-6, var_m)
                        std_m = np.sqrt(var_m)

                        upper_bound = mean_m + beta_t * std_m
                        lower_bound = mean_m - beta_t * std_m

                        diff_upper = upper_bound[0, 0] - meta_outputs[j, 0]
                        diff_lower = lower_bound[0, 0] - meta_outputs[j, 0]
                        max_diff = np.max([np.abs(diff_upper), np.abs(diff_lower)])
                        upper_bounds_all.append(max_diff)
                    if self.use_max:
                        d_bars_eta.append(np.max(upper_bounds_all))
                    else:
                        d_bars_eta.append(np.mean(upper_bounds_all))

                eta_tmp = 0
                for m in range(M):
                    eta_tmp += d_bars_eta[m] * self.w[m]
                print("eta_tmp (prev): ", eta_tmp)
                eta_tmp = np.power(eta_tmp, -self.eps)
                print("eta_tmp (after): ", eta_tmp)
                self.eta = self.eta * np.min([self.min_decay, eta_tmp])
            else:
                self.eta = self.meta_functions["etas"][i]

            print("[current eta: {0}]".format(self.eta))
            self.res['all']['all_etas'].append(self.eta)

        x_max = acq_max(ac=self.util.utility,
                        gp=None,
                        meta_gps=self.meta_gps,
                        w=self.w,
                        tau=self.meta_functions["tau"],
                        eta=self.eta,
                        y_max=y_max,
                        bounds=self.bounds, 
                        iteration=1,
                        gp_samples=None)

        print("x_max: ", x_max)
        
        if self.verbose:
            self.plog.print_header(initialization=False)
        for i in range(n_iter):
            if not self.X.shape[0] == 0:
                if np.any(np.all(self.X - x_max == 0, axis=1)):
                    x_max = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=self.bounds.shape[0])

            curr_y = self.f(x_max)

            self.Y = np.append(self.Y, curr_y)
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))

            if self.Y[-1] > y_max:
                y_max = self.Y[-1]
                self.incumbent = self.Y[-1]

            # Updating the GP.
            ur = unique_rows(self.X)
            if self.gp_model == 'sklearn':
                pass
            elif self.gp_model == 'gpflow':
                pass
            elif self.gp_model == 'gpy':
                if self.gp is None: # if there is no initial target observations
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
                        
                    else:
                        self.gp.optimize_restarts(num_restarts = 10, messages=False)
                        self.gp_params = self.gp.parameters

                        gp_samples = None # set this flag variable to None, to indicate that MCMC is not used

                        print("---Optimized hyper: ", self.gp)

                    
            ####### update the weights w's
            iteration = i+1
            d = self.X.shape[1]
            if use_fixed_kappa:
                beta_t = np.sqrt(kappa)
            else:
                beta_t = np.sqrt(kappa_scale * d * np.log(2 * iteration))

            M = len(self.meta_functions["Xs"])
            d_bars = []
            for k in range(M):
                meta_inputs = self.meta_functions["Xs"][k]
                meta_outputs = self.meta_functions["Ys"][k]
                
                if self.standardize:
                    meta_outputs = (meta_outputs - np.mean(meta_outputs)) / np.std(meta_outputs)

                N_meta_inputs = self.meta_functions["Xs"][k].shape[0]
                upper_bounds_all = []
                for j in range(N_meta_inputs):
                    meta_x = meta_inputs[j, :]
                    meta_x = meta_x.reshape(1, -1)

                    mean_m, var_m = self.gp.predict(meta_x)
                    var_m = max(1e-6, var_m)
                    std_m = np.sqrt(var_m)

                    upper_bound = mean_m + beta_t * std_m
                    lower_bound = mean_m - beta_t * std_m

                    diff_upper = upper_bound[0, 0] - meta_outputs[j, 0]
                    diff_lower = lower_bound[0, 0] - meta_outputs[j, 0]
                    max_diff = np.max([np.abs(diff_upper), np.abs(diff_lower)])
                    upper_bounds_all.append(max_diff)
                
                if self.use_max:
                    d_bars.append(np.max(upper_bounds_all))
                else:
                    d_bars.append(np.mean(upper_bounds_all))

            self.all_d_bars.append(d_bars)
            all_d_bars_np = np.array(self.all_d_bars)
            tmp = np.sum(all_d_bars_np, axis=0)
            tmp = np.exp(-self.online_learning_rate * tmp)

            if not self.fix_w:
                self.w = tmp / np.sum(tmp)
            else:
                self.w = np.repeat(1.0 / M, M)

            self.res['all']['all_ws'].append(self.w)
            print("updated w: ", self.w)
            
            if self.use_adaptive_eta:
                d_bars_eta = []
                for k in range(M):
                    meta_inputs = self.meta_functions["Xs"][k]
                    meta_outputs = self.meta_functions["Ys"][k]

                    if self.standardize:
                        meta_outputs = (meta_outputs - np.mean(meta_outputs)) / np.std(meta_outputs)# scale everything into [0, 1]

                    N_meta_inputs = self.meta_functions["Xs"][k].shape[0]
                    upper_bounds_all = []
                    for j in range(N_meta_inputs):
                        meta_x = meta_inputs[j, :]
                        meta_x = meta_x.reshape(1, -1)

                        mean_m, var_m = self.gp.predict(meta_x)
                        var_m = max(1e-6, var_m)
                        std_m = np.sqrt(var_m)

                        upper_bound = mean_m + beta_t * std_m
                        lower_bound = mean_m - beta_t * std_m

                        diff_upper = upper_bound[0, 0] - meta_outputs[j, 0]
                        diff_lower = lower_bound[0, 0] - meta_outputs[j, 0]
                        max_diff = np.max([np.abs(diff_upper), np.abs(diff_lower)])
                        upper_bounds_all.append(max_diff)
                    if self.use_max:
                        d_bars_eta.append(np.max(upper_bounds_all))
                    else:
                        d_bars_eta.append(np.mean(upper_bounds_all))

                eta_tmp = 0
                for m in range(M):
                    eta_tmp += d_bars_eta[m] * self.w[m]
                print("eta_tmp (prev): ", eta_tmp)
                eta_tmp = np.power(eta_tmp, -self.eps)
                print("eta_tmp (after): ", eta_tmp)
                print("eta_tmp: ", eta_tmp)
                self.eta = self.eta * np.min([self.min_decay, eta_tmp])
            else:
                self.eta = self.meta_functions["etas"][i]

            print("[current eta: {0}]".format(self.eta))
            self.res['all']['all_etas'].append(self.eta)

            if self.verbose:
                self.plog.print_step(x_max, self.Y[-1], warning=False)
            
            x_max = acq_max(ac=self.util.utility,
                            gp=self.gp,
                            meta_gps=self.meta_gps,
                            w=self.w,
                            tau=self.meta_functions["tau"],
                            eta=self.eta,
                            y_max=y_max,
                            bounds=self.bounds,
                            iteration=i+2,
                            gp_samples=None)

            self.i += 1

            x_max_param = self.X[self.Y.argmax(), :-1]

            self.res['max'] = {'max_val': self.Y.max(), 'max_params': dict(zip(self.keys, x_max_param))}
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(self.X[-1])

            if self.log_file is not None:
                pickle.dump(self.res, open(self.log_file, "wb"))
            
        if self.verbose:
            self.plog.print_summary()

