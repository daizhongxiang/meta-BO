import GPy
import numpy as np
import matplotlib.pyplot as plt
from bayesian_optimization import BayesianOptimization

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.datasets import mnist
from keras import backend as K

import os

num_classes = 10
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def obj_func_mnist(param):
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


run_list = np.arange(10)
for itr in run_list:
    lr_BO = BayesianOptimization(f=obj_func_mnist,
            pbounds={'lr':(0, 1), 'lr_decay':(0, 1), 'l2':(0, 1)}, gp_opt_schedule=5, \
            gp_model='gpy', use_init=None, gp_mcmc=False, \
            log_file="results/mnist_" + str(itr) + ".p", save_init=True, \
            save_init_file="initializations/mnist_init_" + str(itr) + ".p", ARD=True)
    lr_BO.maximize(n_iter=25, init_points=5, kappa=2, use_fixed_kappa=False, kappa_scale=0.2, acq='ucb')
