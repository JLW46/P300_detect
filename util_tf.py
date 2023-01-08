import numpy as np
import tensorflow as tf
from tensorflow import keras

def _conv_2D(X, n_ch, k_size, strides, activation=None, padding='same'):
    model_regulizer = None
    kernel_initializer = tf.initializers.GlorotUniform()
    X = keras.layers.Conv2D(n_ch,
                            (k_size[0], k_size[1]),
                            kernel_regularizer=model_regulizer,
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            strides=(strides[0], strides[1]),
                            padding=padding)(X)
    return X

def _cecotti_cnn1(in_shape, out_shape):
    # input = [chs, sample_pts]
    N = 10
    Nc = in_shape[0]
    Ns = in_shape[1]
    kernel_initializer = tf.initializers.GlorotUniform()
    input = keras.layers.Input(shape=in_shape)
    X = _conv_2D(input, N, [Nc, 1], [1, 1], None, 'valid')
    X = _conv_2D(X, 5*N, [1, 32], [1, 32], None, 'valid')
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(100, kernel_initializer=kernel_initializer,
                           kernel_regularizer=None, activation=None)(X)
    X = keras.layers.Dense(out_shape, kernel_initializer=kernel_initializer,
                           kernel_regularizer=None, activation='sigmoid')(X)
    model = keras.models.Model(input, X, name="test_model")
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=keras.metrics.BinaryAccuracy(),
                  loss_weights=[5, 1])
    return model

# test
# model = _cecotti_cnn1([64, 192, 1], 2)