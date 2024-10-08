# Code adapted based on: https://github.com/fmfn/BayesianOptimization

#from __future__ import print_function
#from __future__ import division
import numpy as np
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipydirect import minimize as mini_direct

import pickle

# always set this to be True, since we only implemented the DIRECT optimizer to maximize the acquisition function
USE_DIRECT_OPTIMIZER = True

def acq_max(ac, gp, meta_gps, w, tau, eta, y_max, bounds, iteration, gp_samples=None, data_mask=None, all_meta_incs=[]):
    print("[Running the direct optimizer]")
    bound_list = []
    for b in bounds:
        bound_list.append(tuple(b))

    res = mini_direct(ac, bound_list, para_dict={"gp":gp, "meta_gps":meta_gps, "w":w, "tau":tau, "eta":eta, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples, "all_meta_incs":all_meta_incs})
    x_max = res["x"]

    return x_max


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, use_fixed_kappa, kappa_scale, xi, gp_model):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.kappa = kappa
        self.use_fixed_kappa = use_fixed_kappa
        self.kappa_scale = kappa_scale
        
        self.xi = xi
        self.gp_model = gp_model

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    # This function is defined to work with the DIRECT optimizer
    def utility(self, x, para_dict):
        gp, meta_gps, w, tau, eta, y_max, iteration, gp_samples, all_meta_incs = para_dict["gp"], para_dict["meta_gps"], para_dict["w"], \
                para_dict["tau"], para_dict["eta"], para_dict["y_max"], \
                para_dict["iteration"], para_dict["gp_samples"], para_dict["all_meta_incs"]
        
        if self.kind == 'ucb':
            return self._ucb(x, gp, meta_gps, w, tau, eta, self.kappa, self.use_fixed_kappa, \
                             self.kappa_scale, iteration, self.gp_model, gp_samples)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi, self.gp_model, gp_samples, meta_gps, w, all_meta_incs)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    @staticmethod
    def _ucb(x, gp, meta_gps, w, tau, eta, kappa, use_fixed_kappa, kappa_scale, iteration, gp_model, gp_samples):
        if USE_DIRECT_OPTIMIZER:
            x = x.reshape(1, -1)

        d = x.shape[1]
        
        post_mean, post_var = 0, 0
        
        M = len(meta_gps)
        for i in range(M):
            gp_m = meta_gps[i]
            mean_m, var_m = gp_m.predict(x)
            
            post_mean += mean_m * w[i]
            post_var += var_m * (w[i]**2)
        
        mean, var = gp.predict(x)
        post_mean += mean * w[-1]
        post_var += var * (w[-1]**2)

        post_std = np.sqrt(post_var)

        if use_fixed_kappa:
            ucb_overall = post_mean + kappa * post_std
        else:
            ucb_overall = post_mean + (kappa_scale * d * np.log(2 * iteration)) * post_std

        if USE_DIRECT_OPTIMIZER:
            optimizer_flag = -1
        else:
            optimizer_flag = 1
#         optimizer_flag = 1
        
        if use_fixed_kappa:
            return optimizer_flag * ucb_overall # beta_t value taken from the high-dimensional BO paper
        else:
            return optimizer_flag * ucb_overall
            # beta_t value taken from the high-dimensional BO paper

    @staticmethod
    def _ei(x, gp, y_max, xi, gp_model, gp_samples, meta_gps, w, all_meta_incs):
        if USE_DIRECT_OPTIMIZER:
            x = x.reshape(1, -1)

        ei_overall = 0

        M = len(meta_gps)
        for i in range(M):
            gp_m = meta_gps[i]
            mean_m, var_m = gp_m.predict(x)
            std_m = np.sqrt(var_m)
            
            y_max_m = all_meta_incs[i]

            pred = np.random.normal(mean_m, std_m, 1)[0]
            ei_overall += np.max([0, (pred - y_max_m - xi)]) * w[i]

        mean, var = gp.predict(x)
        std = np.sqrt(var)
        z = (mean - y_max - xi) / std
        ei_overall += ((mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)) * w[-1]

        ei_overall = ei_overall / np.sum(w)

        if USE_DIRECT_OPTIMIZER:
            optimizer_flag = -1
        else:
            optimizer_flag = 1

        return optimizer_flag * ei_overall

    @staticmethod
    def _poi(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        z = (mean - y_max - xi)/std
        return norm.cdf(z)


def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]


class BColours(object):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


class PrintLog(object):

    def __init__(self, params):

        self.ymax = None
        self.xmax = None
        self.params = params
        self.ite = 1

        self.start_time = datetime.now()
        self.last_round = datetime.now()

        # sizes of parameters name and all
        self.sizes = [max(len(ps), 7) for ps in params]

        # Sorted indexes to access parameters
        self.sorti = sorted(range(len(self.params)),
                            key=self.params.__getitem__)

    def reset_timer(self):
        self.start_time = datetime.now()
        self.last_round = datetime.now()

    def print_header(self, initialization=True):

        if initialization:
            print("{}Initialization{}".format(BColours.RED,
                                              BColours.ENDC))
        else:
            print("{}Bayesian Optimization{}".format(BColours.RED,
                                                     BColours.ENDC))

        print(BColours.BLUE + "-" * (29 + sum([s + 5 for s in self.sizes])) +
            BColours.ENDC)

        print("{0:>{1}}".format("Step", 5), end=" | ")
        print("{0:>{1}}".format("Time", 6), end=" | ")
        print("{0:>{1}}".format("Value", 10), end=" | ")

        for index in self.sorti:
            print("{0:>{1}}".format(self.params[index],
                                    self.sizes[index] + 2),
                  end=" | ")
        print('')

    def print_step(self, x, y, warning=False):

        print("{:>5d}".format(self.ite), end=" | ")

        m, s = divmod((datetime.now() - self.last_round).total_seconds(), 60)
        print("{:>02d}m{:>02d}s".format(int(m), int(s)), end=" | ")

        if self.ymax is None or self.ymax < y:
            self.ymax = y
            self.xmax = x
            print("{0}{2: >10.5f}{1}".format(BColours.MAGENTA,
                                             BColours.ENDC,
                                             y),
                  end=" | ")

            for index in self.sorti:
                print("{0}{2: >{3}.{4}f}{1}".format(
                            BColours.GREEN, BColours.ENDC,
                            x[index],
                            self.sizes[index] + 2,
                            min(self.sizes[index] - 3, 6 - 2)
                        ),
                      end=" | ")
        else:
            print("{: >10.5f}".format(y), end=" | ")
            for index in self.sorti:
                print("{0: >{1}.{2}f}".format(x[index],
                                              self.sizes[index] + 2,
                                              min(self.sizes[index] - 3, 6 - 2)),
                      end=" | ")

        if warning:
            print("{}Warning: Test point chose at "
                  "random due to repeated sample.{}".format(BColours.RED,
                                                            BColours.ENDC))

        print()

        self.last_round = datetime.now()
        self.ite += 1

    def print_summary(self):
        pass
