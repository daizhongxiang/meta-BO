# -*- coding: utf-8 -*-

"""
This script implements the RM-GP-TS algorithm
"""

import numpy as np
import GPy
from helper_funcs_rm_gp_ts import UtilityFunction, unique_rows, PrintLog, acq_max
import pickle
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

class BayesianOptimization_meta(object):

    def __init__(self, f, pbounds, gp_opt_schedule, gp_model, ARD=False, \
                 use_init=False, gp_mcmc=False, log_file=None, save_init=False, save_init_file=None, fix_gp_hypers=None, \
                 M=50, N=50, meta_functions=None, \
                 online_learning_rate=0.1, fix_w=False, eps=0.5, \
                 min_decay=1.0, rho=1.0, use_max=False, use_adaptive_eta=True, verbose=1):
        self.eta = 0.0
        
        self.use_max = use_max
        self.eps = eps
        self.rho = rho
        self.min_decay = min_decay
        self.use_adaptive_eta = use_adaptive_eta
        
        self.fix_w = fix_w
        
        self.online_learning_rate = online_learning_rate
        self.meta_functions = meta_functions

        self.M = M
        self.N = N

        self.w = self.meta_functions["w"]
        
        self.use_init = use_init
        self.standardize = False

        self.meta_gps = []

        self.ARD = ARD
        self.fix_gp_hypers = fix_gp_hypers
    
        self.log_file = log_file
        
        self.pbounds = pbounds
        
        self.incumbent = None
        
        self.keys = list(pbounds.keys())

        self.dim = len(pbounds)

        self.bounds = []
        for key in self.pbounds.keys():
            self.bounds.append(self.pbounds[key])
        self.bounds = np.asarray(self.bounds)
        
        self.f = f

        self.initialized = False

        self.init_points = []
        self.x_init = []
        self.y_init = []

        self.X = np.array([]).reshape(-1, 1)
        self.Y = np.array([])
        
        self.i = 0

        self.gp_mcmc = gp_mcmc
        
        self.gp_model = gp_model
        # the gp model to use; only GPy is implemented for now
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
                          'all_ws':[], 'all_etas':[]}
        self.all_d_bars = []

        self.verbose = verbose
        
        
    def init(self, init_points):
        l = [np.random.uniform(x[0], x[1], size=init_points)
             for x in self.bounds]

        self.init_points += list(map(list, zip(*l)))
        y_init = []
        for x in self.init_points:
            print("[init point]: ", x)
            curr_y, target_value = self.f(x)
            self.res['all']['target_f_values'].append(target_value)
            
            y_init.append(curr_y)
            self.res['all']['init_values'].append(curr_y)
            self.res['all']['init_params'].append(dict(zip(self.keys, x)))

        self.X = np.asarray(self.init_points)
        self.Y = np.asarray(y_init)

        self.incumbent = np.max(y_init)
        self.initialized = True
        
        print("[inits: ]", self.X)
        print("[init values: ]", self.Y)

        init = {"X":self.X, "Y":self.Y}
        self.res['all']['init'] = init

        if self.save_init:
            pickle.dump(init, open(self.save_init_file, "wb"))

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 kappa=2.576,
                 use_fixed_kappa=True,
                 kappa_scale=0.2,
                 xi=0.0,):

        self.plog.reset_timer()

        # this utility function is used for the case of sampling a function from the GP posterior of the target function
        self.util_ts = UtilityFunction(kind="ts", kappa=kappa, use_fixed_kappa=use_fixed_kappa, kappa_scale=kappa_scale, xi=xi)
        
        # this utility function is used for the case of sampling functions from the GP posteriors of the meta-functions
        self.util_ts_new = UtilityFunction(kind="ts_new", kappa=kappa, use_fixed_kappa=use_fixed_kappa, kappa_scale=kappa_scale, xi=xi)

        if not self.initialized:
            if self.use_init != None:
                init = pickle.load(open(self.use_init, "rb"))

                print("[loaded init: {0}; {1}]".format(init["X"], init["Y"]))

                self.X, self.Y = init["X"], init["Y"]
                self.incumbent = np.max(self.Y)
                self.initialized = True
                self.res['all']['init'] = init
                self.res['all']['init_values'] = list(self.Y)
                
                print("Using pre-existing initializations with {0} points".format(len(self.Y)))
            else:
                if init_points > 0:
                    self.init(init_points)

        y_max = np.max(self.Y)
        ur = unique_rows(self.X)

        if self.gp_model == 'sklearn':
            pass
        elif self.gp_model == 'gpflow':
            pass
        elif self.gp_model == 'gpy':
            ### fit meta GPs
            M = len(self.meta_functions["Xs"])
            for i in range(M):
                X_m = self.meta_functions["Xs"][i]
                Y_m = self.meta_functions["Ys"][i]

                if self.standardize:
                    Y_m = (Y_m - np.mean(Y_m)) / np.std(Y_m)
                
                if self.fix_gp_hypers is None:
                    gp_m = GPy.models.GPRegression(X_m, Y_m, GPy.kern.RBF(input_dim=self.X.shape[1], lengthscale=0.05, ARD=self.ARD))
                else:
                    gp_m = GPy.models.GPRegression(X_m, Y_m, GPy.kern.RBF(input_dim=self.X.shape[1], \
                            lengthscale=self.fix_gp_hypers[0], ARD=self.ARD))

                gp_m.optimize()

                self.meta_gps.append(gp_m)
            

            if self.fix_gp_hypers is None:
                self.gp = GPy.models.GPRegression(self.X[ur], self.Y[ur].reshape(-1, 1), \
                        GPy.kern.RBF(input_dim=self.X.shape[1], lengthscale=1.0, variance=0.1, ARD=self.ARD))
            else:
                self.gp = GPy.models.GPRegression(self.X[ur], self.Y[ur].reshape(-1, 1), \
                        GPy.kern.RBF(input_dim=self.X.shape[1], lengthscale=self.fix_gp_hypers, ARD=self.ARD))

            if init_points > 1:
                if self.fix_gp_hypers is None:
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


        if self.X.shape[0] > 0:
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
                eta_tmp *= self.rho
                eta_tmp = np.power(eta_tmp, -self.eps)
                print("eta_tmp (after): ", eta_tmp)
                self.eta = self.eta * np.min([self.min_decay, eta_tmp])
            else:
                self.eta = self.meta_functions["etas"][i]

            print("[current eta: {0}]".format(self.eta))
            self.res['all']['all_etas'].append(self.eta)


        if np.random.random() < 1 - self.eta:
            M_target = self.M

            ls_target = self.gp["rbf.lengthscale"][0]
            v_kernel = self.gp["rbf.variance"][0]
            obs_noise = self.gp["Gaussian_noise.variance"][0]

            try:
                s = np.random.multivariate_normal(np.zeros(self.dim), 1 / (ls_target**2) * np.identity(self.dim), M_target)
            except np.linalg.LinAlgError:
                print("<----------------SVD Error------------------->")
                s = np.random.rand(M_target, self.dim) - 0.5

            b = np.random.uniform(0, 2 * np.pi, M_target)

            random_features_target = {"M":M_target, "length_scale":ls_target, "s":s, "b":b, "obs_noise":obs_noise, "v_kernel":v_kernel}

            Phi = np.zeros((self.X.shape[0], M_target))
            for i, x in enumerate(self.X):
                x = np.squeeze(x).reshape(1, -1)
                features = np.sqrt(2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
                features = features / np.sqrt(np.inner(features, features))
                features = np.sqrt(v_kernel) * features
                Phi[i, :] = features

            Sigma_t = np.dot(Phi.T, Phi) + obs_noise * np.identity(M_target)
            Sigma_t_inv = np.linalg.inv(Sigma_t)
            nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), self.Y.reshape(-1, 1))

            try:
                w_sample = np.random.multivariate_normal(np.squeeze(nu_t), obs_noise * Sigma_t_inv, 1)
            except np.linalg.LinAlgError:
                print("<----------------SVD Error------------------->")
                w_sample = np.random.rand(1, self.M) - 0.5

            x_max = acq_max(ac=self.util_ts.utility, gp=self.gp,
                        M=M_target, N=self.N, random_features=random_features_target, \
                        ws=self.w, w_sample=w_sample, bounds=self.bounds, list_random_features=None, list_w_sample=None)
        else:
            list_random_features = []
            list_w_sample = []
            for n in range(self.N):
                print("agent: ", n)
                xs = self.meta_functions["Xs"][n]
                ys = self.meta_functions["Ys"][n]
                dim = self.dim

                ls = self.meta_gps[n]["rbf.lengthscale"][0]
                v_kernel = self.meta_gps[n]["rbf.variance"][0]
                obs_noise = self.meta_gps[n]["Gaussian_noise.variance"][0]

                M = self.M

                try:
                    s = np.random.multivariate_normal(np.zeros(dim), 1 / (ls**2) * np.identity(dim), M)
                except np.linalg.LinAlgError:
                    print("<----------------SVD Error------------------->")
                    s = np.random.rand(M, dim) - 0.5
                b = np.random.uniform(0, 2 * np.pi, M)

                Phi = np.zeros((xs.shape[0], M))
                for i, x in enumerate(xs):
                    x = np.squeeze(x).reshape(1, -1)
                    features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)

                    features = features / np.sqrt(np.inner(features, features))
                    features = np.sqrt(v_kernel) * features

                    Phi[i, :] = features

                Sigma_t = np.dot(Phi.T, Phi) + obs_noise * np.identity(M)
                Sigma_t_inv = np.linalg.inv(Sigma_t)
                nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), ys)

                try:
                    w_sample = np.random.multivariate_normal(np.squeeze(nu_t), obs_noise * Sigma_t_inv, 1)
                except np.linalg.LinAlgError:
                    print("<----------------SVD Error------------------->")
                    w_sample = np.random.rand(1, self.M) - 0.5

                random_features = {"M":M, "length_scale":ls, "s":s, "b":b, "obs_noise":obs_noise, "v_kernel":v_kernel}

                list_random_features.append(random_features)
                list_w_sample.append(w_sample)

            x_max = acq_max(ac=self.util_ts_new.utility, gp=self.gp,
                            M=self.M, N=self.N, random_features=None, ws=self.w, w_sample=None, bounds=self.bounds, \
                             list_random_features=list_random_features, list_w_sample=list_w_sample)
        
        if self.verbose:
            self.plog.print_header(initialization=False)

        for i in range(n_iter):
            pwarning = False
            if not self.X.shape[0] == 0:
                if np.any(np.all(self.X - x_max == 0, axis=1)):
                    print("X repeated: ", x_max)
                    x_max = np.random.uniform(self.bounds[:, 0],
                                          self.bounds[:, 1],
                                          size=self.bounds.shape[0])
                    pwarning = True

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
                eta_tmp *= self.rho
                eta_tmp = np.power(eta_tmp, -self.eps)
                print("eta_tmp (after): ", eta_tmp)
                print("eta_tmp: ", eta_tmp)
                self.eta = self.eta * np.min([self.min_decay, eta_tmp])
            else:
                self.eta = self.meta_functions["etas"][i]

            if np.random.random() < 1 - self.eta:
                M_target = self.M

                ls_target = self.gp["rbf.lengthscale"][0]
                v_kernel = self.gp["rbf.variance"][0]
                obs_noise = self.gp["Gaussian_noise.variance"][0]

                try:
                    s = np.random.multivariate_normal(np.zeros(self.dim), 1 / (ls_target**2) * np.identity(self.dim), M_target)
                except np.linalg.LinAlgError:
                    print("<----------------SVD Error------------------->")
                    s = np.random.rand(M_target, self.dim) - 0.5

                b = np.random.uniform(0, 2 * np.pi, M_target)

                random_features_target = {"M":M_target, "length_scale":ls_target, "s":s, "b":b, "obs_noise":obs_noise, "v_kernel":v_kernel}

                Phi = np.zeros((self.X.shape[0], M_target))
                for i, x in enumerate(self.X):
                    x = np.squeeze(x).reshape(1, -1)
                    features = np.sqrt(2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)

                    features = features / np.sqrt(np.inner(features, features))
                    features = np.sqrt(v_kernel) * features

                    Phi[i, :] = features

                Sigma_t = np.dot(Phi.T, Phi) + obs_noise * np.identity(M_target)
                Sigma_t_inv = np.linalg.inv(Sigma_t)
                nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), self.Y.reshape(-1, 1))

                try:
                    w_sample = np.random.multivariate_normal(np.squeeze(nu_t), obs_noise * Sigma_t_inv, 1)
                except np.linalg.LinAlgError:
                    print("<----------------SVD Error------------------->")
                    w_sample = np.random.rand(1, self.M) - 0.5

                x_max = acq_max(ac=self.util_ts.utility, gp=self.gp,
                            M=M_target, N=self.N, random_features=random_features_target, \
                            ws=self.w, w_sample=w_sample, bounds=self.bounds, list_random_features=None, list_w_sample=None)

            else:
                list_random_features = []
                list_w_sample = []
                for n in range(self.N):
                    print("agent: ", n)
                    xs = self.meta_functions["Xs"][n]
                    ys = self.meta_functions["Ys"][n]
                    dim = self.dim

                    ls = self.meta_gps[n]["rbf.lengthscale"][0]
                    v_kernel = self.meta_gps[n]["rbf.variance"][0]
                    obs_noise = self.meta_gps[n]["Gaussian_noise.variance"][0]

                    M = self.M

                    try:
                        s = np.random.multivariate_normal(np.zeros(dim), 1 / (ls**2) * np.identity(dim), M)
                    except np.linalg.LinAlgError:
                        print("<----------------SVD Error------------------->")
                        s = np.random.rand(M, dim) - 0.5

                    b = np.random.uniform(0, 2 * np.pi, M)

                    Phi = np.zeros((xs.shape[0], M))
                    for i, x in enumerate(xs):
                        x = np.squeeze(x).reshape(1, -1)
                        features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)

                        features = features / np.sqrt(np.inner(features, features))
                        features = np.sqrt(v_kernel) * features

                        Phi[i, :] = features

                    Sigma_t = np.dot(Phi.T, Phi) + obs_noise * np.identity(M)
                    Sigma_t_inv = np.linalg.inv(Sigma_t)
                    nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), ys)

                    try:
                        w_sample = np.random.multivariate_normal(np.squeeze(nu_t), obs_noise * Sigma_t_inv, 1)
                    except np.linalg.LinAlgError:
                        print("<----------------SVD Error------------------->")
                        w_sample = np.random.rand(1, self.M) - 0.5

                    random_features = {"M":M, "length_scale":ls, "s":s, "b":b, "obs_noise":obs_noise, "v_kernel":v_kernel}

                    list_random_features.append(random_features)
                    list_w_sample.append(w_sample)


                x_max = acq_max(ac=self.util_ts_new.utility, gp=self.gp,
                            M=self.M, N=self.N, random_features=None, ws=self.w, w_sample=None, bounds=self.bounds,\
                             list_random_features=list_random_features, list_w_sample=list_w_sample)

            if self.verbose:
                self.plog.print_step(x_max, self.Y[-1], warning=pwarning)

            self.i += 1

            x_max_param = self.X[self.Y.argmax(), :-1]

            self.res['max'] = {'max_val': self.Y.max(), 'max_params': dict(zip(self.keys, x_max_param))}
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(self.X[-1])
            self.res['all']['params'].append(self.X[-1])

            if self.log_file is not None:
                pickle.dump(self.res, open(self.log_file, "wb"))

        if self.verbose:
            self.plog.print_summary()

