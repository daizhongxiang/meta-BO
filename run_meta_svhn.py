import GPy
import numpy as np
import matplotlib.pyplot as plt
from bayesian_optimization_meta import BayesianOptimization_meta

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

from scipy.io import loadmat
import time

import os

np.random.seed(0)

num_classes = 10
# load the svhn dataset
svhn_path = '' # place the files "train_32x32.mat" and "test_32x32.mat" in this folder
train_data = loadmat(svhn_path + 'train_32x32.mat')
test_data = loadmat(svhn_path + 'test_32x32.mat')
y_train = keras.utils.to_categorical(train_data['y'][:,0])[:,1:]
y_test = keras.utils.to_categorical(test_data['y'][:,0])[:,1:]
x_train = np.zeros((73257, 32, 32, 3))
for i in range(len(x_train)):
    x_train[i] = train_data['X'].T[i].T.astype('float32')/255
x_test = np.zeros((26032, 32, 32, 3))
for i in range(len(x_test)):
    x_test[i] = test_data['X'].T[i].T.astype('float32')/255

# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)

def obj_func_svhn(param):
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

    # build the CNN model using Keras
    model = Sequential()
    model.add(Conv2D(conv_filters, (kernel_size, kernel_size), padding='same',
                     input_shape=x_train.shape[1:], kernel_regularizer=regularizers.l2(l2_regular)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
#     model.add(Dropout(dropout_rate))

    model.add(Conv2D(conv_filters, (kernel_size, kernel_size), padding='same', kernel_regularizer=regularizers.l2(l2_regular)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
#     model.add(Dropout(dropout_rate))

    if num_conv_layers >= 3:
        model.add(Conv2D(conv_filters, (kernel_size, kernel_size), padding='same', kernel_regularizer=regularizers.l2(l2_regular)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
#         model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(dense_units, kernel_regularizer=regularizers.l2(l2_regular)))
    model.add(Activation('relu'))
#     model.add(Dropout(dropout_rate))
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
meta_x = [meta_x[0], meta_x[1], meta_x[3]]
meta_y = [meta_y[0], meta_y[1], meta_y[3]]

M = len(meta_x)
w = np.repeat(1.0 / M, M)
tau = 1
etas = []
for i in range(200):
    etas.append((0.9**(i+1)))

meta_functions = {"Xs":meta_x, "Ys":meta_y, "w":w, "tau":tau, "etas":etas}


run_list = np.arange(10)
for itr in run_list:
    lr_BO_meta_online = BayesianOptimization_meta(f=obj_func_svhn,
        pbounds={'lr':(0, 1), 'lr_decay':(0, 1), 'l2':(0, 1)}, gp_opt_schedule=5, \
        gp_model='gpy', use_init="initializations/svhn_init_" + str(itr) + ".p", gp_mcmc=False, \
        log_file="results/meta_svhn_" + str(itr) + "_lr_1_eps_07_min_decay_07_tau_1.p", save_init=False, \
        save_init_file=None, meta_functions=meta_functions, online_learning_rate=1.0, fix_w=False, eps=0.7, ARD=True, \
        min_decay=0.7, use_max=False)
    lr_BO_meta_online.maximize(n_iter=25, init_points=5, kappa=2, use_fixed_kappa=False, kappa_scale=0.2, acq='ucb')
