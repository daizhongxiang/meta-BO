'''
This script load and re-format the meta observations, to be used in the meta BO algorithms
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle

file_load = pickle.load(open("meta_tasks/cifar_10.p", "rb"))
func_values = file_load["all"]["values"]
init = file_load["all"]["init"]["Y"]
func_values_cifar_10 = np.array(list(init) + func_values).reshape(-1, 1)
init_X = file_load["all"]["init"]["X"]
init_X = [x for x in init_X]
params_cifar_10 = init_X + file_load["all"]["params"]

file_load = pickle.load(open("meta_tasks/cifar_100.p", "rb"))
func_values = file_load["all"]["values"]
init = file_load["all"]["init"]["Y"]
func_values_cifar_100 = np.array(list(init) + func_values).reshape(-1, 1)
init_X = file_load["all"]["init"]["X"]
init_X = [x for x in init_X]
params_cifar_100 = init_X + file_load["all"]["params"]

file_load = pickle.load(open("meta_tasks/svhn.p", "rb"))
func_values = file_load["all"]["values"]
init = file_load["all"]["init"]["Y"]
func_values_svhn = np.array(list(init) + func_values).reshape(-1, 1)
init_X = file_load["all"]["init"]["X"]
init_X = [x for x in init_X]
params_svhn = init_X + file_load["all"]["params"]

file_load = pickle.load(open("meta_tasks/mnist.p", "rb"))
func_values = file_load["all"]["values"]
init = file_load["all"]["init"]["Y"]
func_values_mnist = np.array(list(init) + func_values).reshape(-1, 1)
init_X = file_load["all"]["init"]["X"]
init_X = [x for x in init_X]
params_mnist = init_X + file_load["all"]["params"]


N_meta_obs = 50
params_cifar_10_np = np.array(params_cifar_10)
params_cifar_100_np = np.array(params_cifar_100)
params_svhn_np = np.array(params_svhn)
params_mnist_np = np.array(params_mnist)
params_meta = [params_cifar_10_np[:N_meta_obs], params_cifar_100_np[:N_meta_obs], params_svhn_np[:N_meta_obs], params_mnist_np[:N_meta_obs]]
func_values_meta = [func_values_cifar_10[:N_meta_obs], func_values_cifar_100[:N_meta_obs], func_values_svhn[:N_meta_obs], func_values_mnist[:N_meta_obs]]

meta_info = {"params_meta":params_meta, "func_values_meta":func_values_meta}


pickle.dump(meta_info, open("meta_obs_images.pkl", "wb"))

