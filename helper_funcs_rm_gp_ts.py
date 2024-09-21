#from __future__ import print_function
#from __future__ import division
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipydirect import minimize as mini_direct
import pickle

# always set this to be True, since we only implemented the DIRECT optimizer to maximize the acquisition function
USE_DIRECT_OPTIMIZER = True

def acq_max(ac, gp, M, N, random_features, ws, w_sample, bounds, list_random_features=None, list_w_sample=None):
    print("[Running the direct optimizer]")
    para_dict={"gp":gp, "M":M, "N":N, "random_features":random_features, "ws":ws, "w_sample":w_sample,\
               "list_random_features":list_random_features, "list_w_sample":list_w_sample}

    bound_list = []
    for b in bounds:
        bound_list.append(tuple(b))

    res = mini_direct(ac, bound_list, para_dict=para_dict)
    x_max = res["x"]
    
    return x_max

class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, use_fixed_kappa, kappa_scale, xi):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.kappa = kappa
        self.use_fixed_kappa = use_fixed_kappa
        self.kappa_scale = kappa_scale
        
        self.xi = xi

        if kind not in ['ts', 'ts_new']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose ucb or ts.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    # This function is defined to work with the DIRECT optimizer
    def utility(self, x, para_dict):
        gp, M, N, random_features, ws, w_sample, list_random_features, list_w_sample = \
                para_dict["gp"], para_dict["M"], para_dict["N"], para_dict["random_features"], para_dict["ws"], \
                para_dict["w_sample"], para_dict["list_random_features"], para_dict["list_w_sample"]

        if self.kind == 'ts':
            return self._ts(x, gp, M, N, random_features, w_sample)
        elif self.kind == 'ts_new':
            return self._ts_new(x, gp, M, N, list_random_features, list_w_sample, ws)

    @staticmethod
    def _ts(x, gp, M, N, random_features, w_sample):
        if USE_DIRECT_OPTIMIZER:
            x = x.reshape(1, -1)
        d = x.shape[1]

        s = random_features["s"]
        b = random_features["b"]
        obs_noise = random_features["obs_noise"]
        v_kernel = random_features["v_kernel"]

        x = np.squeeze(x).reshape(1, -1)
        features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
        features = features.reshape(-1, 1)

        features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
        features = np.sqrt(v_kernel) * features

        f_value = np.squeeze(np.dot(w_sample, features))

        if USE_DIRECT_OPTIMIZER:
            optimizer_flag = -1
        else:
            optimizer_flag = 1
        
        return optimizer_flag * f_value

    @staticmethod
    def _ts_new(x, gp, M, N, list_random_features, list_w_sample, ws):
        if USE_DIRECT_OPTIMIZER:
            x = x.reshape(1, -1)
        d = x.shape[1]

        all_f_values = []
        for n in range(N):
            random_features = list_random_features[n]
            w_sample = list_w_sample[n]

            s = random_features["s"]
            b = random_features["b"]
            obs_noise = random_features["obs_noise"]
            v_kernel = random_features["v_kernel"]

            x = np.squeeze(x).reshape(1, -1)
            features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
            features = features.reshape(-1, 1)

            features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
            features = np.sqrt(v_kernel) * features

            f_value = np.squeeze(np.dot(w_sample, features))
            
            all_f_values.append(ws[n] * f_value)

        f_value = np.sum(all_f_values)

        if USE_DIRECT_OPTIMIZER:
            optimizer_flag = -1
        else:
            optimizer_flag = 1
        
        return optimizer_flag * f_value
    
    
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
