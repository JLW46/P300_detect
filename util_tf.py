import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
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

def _mbconv(X, ch_out, kern=(3,3), t=4, reduction=False, SE=0.25, dropout = 0.5):
    use_bias = False
    if reduction:
        strides = (2, 2)
    else:
        strides = (1, 1)
    ch_in = X.shape[-1]
    kernel_initializer = tf.initializers.GlorotUniform()
    Res = X
    X = keras.layers.Conv2D(ch_in*t,
                            kernel_size=(1, 1),
                            kernel_regularizer=None,
                            activation=None,
                            kernel_initializer=kernel_initializer,
                            strides=(1, 1),
                            padding='same',
                            use_bias=use_bias)(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.ReLU(max_value=6, negative_slope=0.0, threshold=0.0)(X)
    X = _depth_conv2D(X, n_mult=1, k_size=kern, strides=strides, activation=None, padding='same',
                      use_bias=False, weight_constraint=None)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.ReLU(max_value=6, negative_slope=0.0, threshold=0.0)(X)
    if SE is not None:
        SE_branch = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)(X)
        SE_size = SE_branch.shape[-1]
        SE_branch = tf.keras.layers.Dense(int(SE_size*SE), kernel_initializer=kernel_initializer,
                           kernel_regularizer=None, activation='relu')(SE_branch)
        SE_branch = tf.keras.layers.Dense(SE_size, kernel_initializer=kernel_initializer,
                                          kernel_regularizer=None, activation='relu')(SE_branch)
        scaler = tf.reshape(SE_branch, shape=[-1, 1, 1, SE_size], name='scaler')
        X = X*scaler
    else:
        pass
    X = keras.layers.Conv2D(ch_out,
                            kernel_size=(1, 1),
                            kernel_regularizer=None,
                            activation=None,
                            kernel_initializer=kernel_initializer,
                            strides=(1, 1),
                            padding='same',
                            use_bias=use_bias)(X)
    X = keras.layers.BatchNormalization()(X)
    if reduction is False and ch_out == ch_in:
        X = tfa.layers.StochasticDepth(survival_probability=dropout)([X, Res])
        # X = keras.layers.add([X, Res])
    else:
        pass
    return X

def _mbconvFused(X, ch_out, kern=(3, 3), t=4, reduction=False, SE=0.25, dropout=0.5):
    use_bias = False
    if reduction:
        strides = (2, 2)
    else:
        strides = (1, 1)
    ch_in = X.shape[-1]
    kernel_initializer = tf.initializers.GlorotUniform()
    Res = X
    X = keras.layers.Conv2D(ch_in*t, kernel_size=kern,
                            kernel_regularizer=None,
                            activation=None,
                            kernel_initializer=kernel_initializer,
                            strides=strides,
                            padding='same',
                            use_bias=use_bias)(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.ReLU(max_value=6, negative_slope=0.0, threshold=0.0)(X)
    if SE is not None:
        SE_branch = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)(X)
        SE_size = SE_branch.shape[-1]
        SE_branch = tf.keras.layers.Dense(int(SE_size*SE), kernel_initializer=kernel_initializer,
                           kernel_regularizer=None, activation='relu')(SE_branch)
        SE_branch = tf.keras.layers.Dense(SE_size, kernel_initializer=kernel_initializer,
                                          kernel_regularizer=None, activation='relu')(SE_branch)
        scaler = tf.reshape(SE_branch, shape=[-1, 1, 1, SE_size], name='scaler')
        X = X*scaler
    else:
        pass
    X = keras.layers.Conv2D(ch_out,
                            kernel_size=(1, 1),
                            kernel_regularizer=None,
                            activation=None,
                            kernel_initializer=kernel_initializer,
                            strides=(1, 1),
                            padding='same',
                            use_bias=use_bias)(X)
    X = keras.layers.BatchNormalization()(X)
    if reduction is False and ch_out == ch_in:
        X = tfa.layers.StochasticDepth(survival_probability=dropout)([X, Res])
        # X = keras.layers.add([X, Res])
    else:
        pass

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
    model = keras.models.Model(input, X, name="cecotti")
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=keras.metrics.CategoricalAccuracy(),
                  loss_weights=[5, 1])
    return model


def _eegnet(in_shape, out_shape, loss_weights=[0.5, 0.5], dropout_rate=0.2):
    weight_constraints_1 = keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=0)
    weight_constraints_2 = keras.constraints.MinMaxNorm(min_value=-0.25, max_value=0.25, rate=1.0, axis=0)
    kernel_initializer = tf.initializers.GlorotUniform()
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
    model = keras.models.Model(input, X, name="eegnet")
    model.summary()
    # acc = tf.keras.metrics.AUC(num_thresholds=50, curve='ROC', summation_method='interpolation')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  # loss=keras.losses.MeanSquaredError(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=keras.metrics.CategoricalAccuracy(),
                  # metrics=acc,
                  loss_weights=loss_weights,
                  weighted_metrics=[])
    return model


def _effnetV2(in_shape, out_shape, loss_weights=[0.5, 0.5]):
    kernel_initializer = tf.initializers.GlorotUniform()
    input = keras.layers.Input(shape=in_shape)
    X = _conv2D(input, n_ch=64, k_size=(3, 3), strides=(1, 2), activation=None, padding='same', use_bias=False)
    X = keras.layers.BatchNormalization()(X)
    X = _mbconvFused(X, ch_out=64, kern=(1, 3), t=2, reduction=False, SE=0.25)
    X = _mbconvFused(X, ch_out=80, kern=(1, 3), t=2, reduction=True, SE=0.25)
    X = _mbconvFused(X, ch_out=96, kern=(1, 3), t=2, reduction=True, SE=0.25)
    X = _mbconv(X, ch_out=112, kern=(1, 3), t=2, reduction=False, SE=0.25)
    X = _mbconv(X, ch_out=128, kern=(1, 3), t=2, reduction=True, SE=0.25)
    X = _mbconv(X, ch_out=144, kern=(1, 3), t=2, reduction=True, SE=0.25)
    X = _conv2D(X, n_ch=256, k_size=(1, 1), strides=(1, 1), activation=None, padding='same', use_bias=False)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.GlobalAveragePooling2D()(X)
    X = keras.layers.Dense(out_shape, kernel_initializer=kernel_initializer,
                           kernel_regularizer=None, activation='sigmoid')(X)
    model = keras.models.Model(input, X, name="efnetV2")
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  # loss=keras.losses.MeanSquaredError(),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=keras.metrics.CategoricalAccuracy(),
                  loss_weights=loss_weights,
                  weighted_metrics=[])
    return model


def _confusion_matrix(Y_pred, Y_true):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    P = 0
    N = 0
    for i in range(np.shape(Y_pred)[0]):
        if Y_true[i, 0] == 1:
            P = P + 1
            if Y_pred[i, 0] > Y_pred[i, 1]:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            N = N + 1
            if Y_pred[i, 0] < Y_pred[i, 1]:
                TN = TN + 1
            else:
                FP = FP + 1
    out = {
        'matrix': np.array([[TP/P, FN/P], [FP/N, TN/N]]),
        'TP': TP/P,
        'FN': FN/P,
        'TN': TN/N,
        'FP': FP/N
    }
    return out

# test
# model = _cecotti_cnn1([64, 192, 1], 2)
# model = _eegnet([60, 250, 1], 2)
# model = _effnetV2([1, 250, 60], 2)
