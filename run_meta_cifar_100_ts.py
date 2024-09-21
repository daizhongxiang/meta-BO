import GPy
import numpy as np
import matplotlib.pyplot as plt
from bayesian_optimization_rm_gp_ts import BayesianOptimization_meta

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

import keras
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.datasets import mnist
from keras import backend as K

import time

import os

np.random.seed(0)

num_classes = 100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def obj_func_cifar(param):
    real_bounds = [[1e-7, 1e-2], [1e-7, 1e-2], [1e-7, 1e-2]]
    learning_rate_ = param[0]
    learning_rate = learning_rate_ * (real_bounds[0][1] - real_bounds[0][0]) + real_bounds[0][0]
    learning_rate_decay_ = param[1]
    learning_rate_decay = learning_rate_decay_ * (real_bounds[1][1] - real_bounds[1][0]) + real_bounds[1][0]
    l2_regular_ = param[2]
    l2_regular = l2_regular_ * (real_bounds[2][1] - real_bounds[2][0]) + real_bounds[2][0]
    
    dropout_rate = 0.0

    batch_size = 128
    conv_filters = 32
    dense_units = 64

    num_conv_layers = 2
    kernel_size = 3
    pool_size = 3

    model = Sequential()
    model.add(Conv2D(conv_filters, (kernel_size, kernel_size), padding='same',
                     input_shape=x_train.shape[1:], kernel_regularizer=regularizers.l2(l2_regular)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(conv_filters, (kernel_size, kernel_size), padding='same', kernel_regularizer=regularizers.l2(l2_regular)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout_rate))

    if num_conv_layers >= 3:
        model.add(Conv2D(conv_filters, (kernel_size, kernel_size), padding='same', kernel_regularizer=regularizers.l2(l2_regular)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(dense_units, kernel_regularizer=regularizers.l2(l2_regular)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=learning_rate, decay=learning_rate_decay)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=20,
              validation_data=(x_test, y_test),
              shuffle=True, verbose=0)
    scores = model.evaluate(x_test, y_test, verbose=0)
    val_acc = scores[1]

    return val_acc

load_meta_info = pickle.load(open("meta_obs_images.pkl", "rb"))
meta_x, meta_y = load_meta_info["params_meta"], load_meta_info["func_values_meta"]

### task order: cifar-10, cifar-100, svhn, mnist
meta_x = [meta_x[0], meta_x[2], meta_x[3]]
meta_y = [meta_y[0], meta_y[2], meta_y[3]]

N = len(meta_x)
w = np.repeat(1.0 / N, N)
tau = 1
etas = []
for i in range(200):
    etas.append((0.9**(i+1)))

meta_functions = {"Xs":meta_x, "Ys":meta_y, "w":w, "tau":tau, "etas":etas}
M = 120

run_list = np.arange(10)
for itr in run_list:
    lr_BO_meta_rm_gp_ts = BayesianOptimization_meta(f=obj_func_cifar, pbounds={'lr':(0, 1), 'lr_decay':(0, 1), 'l2':(0, 1)},\
              gp_opt_schedule=5, gp_model='gpy', use_init="initializations/cifar_100_init_" + str(itr) + ".p", \
              gp_mcmc=False, log_file="results/meta_cifar_100_" + str(itr) + ".p", save_init=False, \
              save_init_file=None, fix_gp_hypers=None, M=M, N=N,
              meta_functions=meta_functions, online_learning_rate=1.0, fix_w=False, eps=0.7, ARD=True, \
              min_decay=0.7, rho=1.0, use_max=False)

    lr_BO_meta_rm_gp_ts.maximize(n_iter=25, init_points=5, kappa=2, use_fixed_kappa=False, kappa_scale=0.2)
