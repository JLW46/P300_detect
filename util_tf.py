import numpy as np
import tensorflow as tf
from tensorflow import keras

def _conv2D(X, n_ch, k_size, strides, activation=None, padding='same', use_bias=True):
    model_regulizer = None
    kernel_initializer = tf.initializers.GlorotUniform()
    X = keras.layers.Conv2D(n_ch,
                            (k_size[0], k_size[1]),
                            kernel_regularizer=model_regulizer,
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            strides=(strides[0], strides[1]),
                            padding=padding,
                            use_bias=use_bias,)(X)
    return X

def _depth_conv2D(X, n_mult, k_size, strides, activation=None, padding='same', use_bias=True, weight_constraint=None):
    model_regulizer = None
    kernel_initializer = tf.initializers.GlorotUniform()
    X = keras.layers.DepthwiseConv2D(kernel_size=(k_size[0], k_size[1]),
                                     strides=(strides[0], strides[1]),
                                     padding=padding,
                                     depth_multiplier=n_mult,
                                     kernel_regularizer=model_regulizer,
                                     activation=activation,
                                     use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     depthwise_constraint=weight_constraint)(X)
    return X

def _cecotti_cnn1(in_shape, out_shape):
    # input = [chs, sample_pts]
    N = 10
    Nc = in_shape[0]
    Ns = in_shape[1]
    kernel_initializer = tf.initializers.GlorotUniform()
    input = keras.layers.Input(shape=in_shape)
    X = _conv2D(input, N, [Nc, 1], [1, 1], None, 'valid')
    # X = _depth_conv2D(X, n_mult=5, k_size=[1, 32], strides=[1, 32], activation=None, padding='valid')
    X = _depth_conv2D(X, n_mult=5, k_size=[1, 32], strides=[32, 32], activation=None, padding='valid')
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(100, kernel_initializer=kernel_initializer,
                           kernel_regularizer=None, activation=None)(X)
    X = keras.layers.Dense(out_shape, kernel_initializer=kernel_initializer,
                           kernel_regularizer=None, activation='sigmoid')(X)
    model = keras.models.Model(input, X, name="test_model")
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=keras.metrics.CategoricalAccuracy(),
                  loss_weights=[5, 1])
    return model


def _eegnet(in_shape, out_shape):
    weight_constraints_1 = keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=0)
    weight_constraints_2 = keras.constraints.MinMaxNorm(min_value=-0.25, max_value=0.25, rate=1.0, axis=0)
    kernel_initializer = tf.initializers.GlorotUniform()
    dropout_rate = 0.5
    F1 = 8
    D = 2
    F2 = F1*D
    input = keras.layers.Input(shape=in_shape)
    ### Block 1 ###
    X = _conv2D(input, F1, [1, int(0.5*in_shape[1])], [1, 1], activation=None, padding='same', use_bias=False)
    X = keras.layers.BatchNormalization()(X)
    X = _depth_conv2D(X, D, [in_shape[0], 1], [1, 1], activation=None, padding='valid', use_bias=False,
                      weight_constraint=weight_constraints_1)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.ELU()(X)
    X = keras.layers.AveragePooling2D(pool_size=(1, 4), strides=None, padding='valid')(X)
    X = keras.layers.Dropout(rate=dropout_rate)(X)
    ### Block 2 ###
    X = _depth_conv2D(X, 1, [1, 16], [1, 1], activation=None, padding='same', use_bias=False,
                      weight_constraint=weight_constraints_1)
    X = _conv2D(X, F2, [1, 1], [1, 1], activation=None, padding='same', use_bias=False)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.ELU()(X)
    X = keras.layers.AveragePooling2D(pool_size=(1, 8), strides=None, padding='valid')(X)
    X = keras.layers.Dropout(rate=dropout_rate)(X)
    ### Final ###
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(out_shape, kernel_initializer=kernel_initializer, use_bias=False,
                           kernel_constraint=weight_constraints_2,
                           kernel_regularizer=None, activation='Softmax')(X)
    model = keras.models.Model(input, X, name="test_model")
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=keras.metrics.CategoricalAccuracy(),
                  loss_weights=[5, 1])
    return model
# test
# model = _cecotti_cnn1([64, 192, 1], 2)
# model = _eegnet([64, 128, 1], 2)