import numpy as np
import sklearn.metrics
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

class IterTracker(keras.callbacks.Callback):

    def __init__(self, X_test, Y_test):
        super(IterTracker, self).__init__()
        self.X_test = X_test
        self.Y_test = Y_test
        self.best_weights = None
        # self.save_path = SAVE_PATH
        self.wait_reset = 0
        self.wait_end = 0
        # self.Y_test = np.squeeze(Y_test)
        # self.Y_test = float(self.Y_test)
        self.best_scores = {
            'Y_pred': 0,
            'Y_true': np.squeeze(self.Y_test).tolist(),
            'epoch': 0,
            'auc': 0
        }

    def on_epoch_end(self, epoch, logs=None):
        out = self.model.predict(x=self.X_test)
        if self.Y_test.shape[-1] > 1:
            Y_pred = out[:, 1]
        else:
            Y_pred = out
        new_auc = sklearn.metrics.roc_auc_score(y_true=self.Y_test, y_score=Y_pred)
        # TN, FP, FN, TP = sklearn.metrics.confusion_matrix(y_true=self.Y_test, y_pred=Y_pred).ravel()
        # if TP == 0:
        #     precision = 0
        #     recall = 0
        #     f1 = 0
        # else:
        #     precision = TP / (TP + FP)
        #     recall = TP / (TP + FN)
        #     f1 = 2 * (precision * recall) / (precision + recall)
        # new_acc_weighted = 0.5 * TP / (TP + FN) + 0.5 * TN / (TN + FP)
        # if self.Y_test.shape[-1] > 1:
        #
        #     TP, TN, FP, FN = _confusion_matrix_2(Y_pred=Y_pred, Y_true=self.Y_test)
        #     P = TP + FN
        #     N = TN + FP
        #     # W_P = N/(P + N)
        #     # W_N = P/(P + N)
        #     # new_acc_weighted = W_P*TP/(TP + FN) + W_N*TN/(TN + FP)
        #     if TP == 0:
        #         precision = 0
        #         recall = 0
        #         f1 = 0
        #     else:
        #         precision = TP/(TP + FP)
        #         recall = TP/(TP + FN)
        #         f1 = 2*(precision*recall)/(precision + recall)
        #     new_acc_weighted = 0.5*TP/(TP + FN) + 0.5*TN/(TN + FP)
        # else:
        #
        #     sklearn.metrics.roc_auc_score(self.Y_test, Y_pred, )
        if new_auc > self.best_scores['auc']:
            self.best_scores['Y_pred'] = np.squeeze(Y_pred).tolist()
            self.best_scores['epoch'] = epoch
            self.best_scores['auc'] = new_auc
            # self.best_scores = {
            #     'Y_pred': Y_pred,
            #     'Y_true': self.Y_test,
            #     'epoch': epoch,
            #     'auc': new_auc
            # }
            self.best_weights = self.model.get_weights()
            self.wait_reset = 0
        else:
            self.wait_reset = self.wait_reset + 1
            if self.wait_reset > 5:
                self.model.set_weights(self.best_weights)
                self.wait_reset = 0
                self.wait_end = self.wait_end + 1
                print('+++++++++ Roll back:' + str(self.wait_end) + ' +++++++++++')
                if self.wait_end > 10:
                    self.model.stop_training = True
                    print('Not improving, terminate!!')
        print('Best AUC: ' + str(self.best_scores['auc']))



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
    Short = X
    X = keras.layers.Conv2D(int(ch_in*t),
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
        X = tfa.layers.StochasticDepth(survival_probability=dropout)([Short, X])
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
                  metrics=keras.metrics.CategoricalAccuracy())
    return model


def _eegnet(in_shape, out_shape, dropout_rate=0.2):
    weight_constraints_1 = keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=0)
    weight_constraints_2 = keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=0)
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
    X = keras.layers.AveragePooling2D(pool_size=(1, 4), strides=None, padding='valid')(X)
    X = keras.layers.Dropout(rate=2*dropout_rate)(X)
    ### Final ###
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(out_shape, kernel_initializer=kernel_initializer, use_bias=False,
                           kernel_constraint=weight_constraints_2,
                           kernel_regularizer=None, activation='sigmoid')(X)
    model = keras.models.Model(input, X, name="eegnet")
    model.summary()
    # acc = tf.keras.metrics.AUC(num_thresholds=50, curve='ROC', summation_method='interpolation')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss=keras.losses.MeanSquaredError(),
                  # loss=keras.losses.BinaryCrossentropy(),
                  # loss=keras.losses.CategoricalCrossentropy(),
                  metrics=keras.metrics.CategoricalAccuracy())
    return model


def _eegnet_1(in_shape, out_shape, dropout_rate=0.2):
    weight_constraints_1 = keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=0)
    weight_constraints_2 = keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=0)
    kernel_initializer = tf.initializers.GlorotUniform()
    input = keras.layers.Input(shape=in_shape)
    ### Block 1 ###
    X_1 = _conv2D(input, 8, [1, int(0.5*in_shape[1])], [1, 1], activation=None, padding='same', use_bias=False)
    X_2 = _conv2D(input, 8, [1, 15], [1, 1], activation=None, padding='same', use_bias=False)
    X_3 = _conv2D(input, 8, [1, 9], [1, 1], activation=None, padding='same', use_bias=False)
    X_4 = _conv2D(input, 8, [1, 5], [1, 1], activation=None, padding='same', use_bias=False)
    X = keras.layers.concatenate([X_1, X_2, X_3, X_4], axis=3)
    # X = _conv2D(input, 32, [1, int(0.5 * in_shape[1])], [1, 1], activation=None, padding='same', use_bias=False)
    X = keras.layers.BatchNormalization()(X)
    X = _depth_conv2D(X, 4, [in_shape[0], 1], [1, 1], activation=None, padding='valid', use_bias=False,
                      weight_constraint=weight_constraints_1)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.ELU()(X)
    X = keras.layers.AveragePooling2D(pool_size=(1, 2), strides=None, padding='valid')(X)
    X = keras.layers.Dropout(rate=2 * dropout_rate)(X)
    ### Block 2 ###
    X = _depth_conv2D(X, 1, [1, 4], [1, 1], activation=None, padding='same', use_bias=False,
                      weight_constraint=weight_constraints_1)
    X = _conv2D(X, 16, [1, 1], [1, 1], activation=None, padding='same', use_bias=False)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.ELU()(X)
    X = keras.layers.AveragePooling2D(pool_size=(1, 2), strides=None, padding='valid')(X)
    X = keras.layers.Dropout(rate=2 * dropout_rate)(X)
    ### Final ###
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(out_shape, kernel_initializer=kernel_initializer, use_bias=False,
                           kernel_constraint=weight_constraints_2,
                           kernel_regularizer=None, activation='sigmoid')(X)
    model = keras.models.Model(input, X, name="eegnet_1")
    model.summary()
    acc = tf.keras.metrics.AUC(num_thresholds=250, curve='ROC', summation_method='interpolation')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  # loss=keras.losses.MeanSquaredError(),
                  loss=keras.losses.BinaryCrossentropy(),
                  # loss=keras.losses.CategoricalCrossentropy(),
                  # metrics=keras.metrics.CategoricalAccuracy(),
                  metrics=acc)
    return model


def _eegnet_2(in_shape, out_shape, dropout_rate=0.2):
    weight_constraints_1 = keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=0)
    weight_constraints_2 = keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=0)
    kernel_initializer = tf.initializers.GlorotUniform()
    input = keras.layers.Input(shape=in_shape)
    ### Block 1 ###
    X_1_1 = _conv2D(input, 4, [1, 50], [1, 1], activation=None, padding='same', use_bias=False)
    X_1_2 = _conv2D(input, 4, [1, 25], [1, 1], activation=None, padding='same', use_bias=False)
    X_1_3 = _conv2D(input, 4, [1, 10], [1, 1], activation=None, padding='same', use_bias=False)
    X_1_4 = _conv2D(input, 4, [1, 5], [1, 1], activation=None, padding='same', use_bias=False)
    X = keras.layers.concatenate([X_1_1, X_1_2, X_1_3, X_1_4], axis=3)
    # X = keras.layers.BatchNormalization()(X)
    # X_2_1 = _depth_conv2D(X, 4, [50, 1], [1, 1], activation=None, padding='same', use_bias=False,
    #                   weight_constraint=weight_constraints_1)
    # X_2_2 = _depth_conv2D(X, 4, [25, 1], [1, 1], activation=None, padding='same', use_bias=False,
    #                       weight_constraint=weight_constraints_1)
    # X_2_3 = _depth_conv2D(X, 4, [10, 1], [1, 1], activation=None, padding='same', use_bias=False,
    #                       weight_constraint=weight_constraints_1)
    # X_2_4 = _depth_conv2D(X, 4, [5, 1], [1, 1], activation=None, padding='same', use_bias=False,
    #                       weight_constraint=weight_constraints_1)
    # X = keras.layers.concatenate([X_2_1, X_2_2, X_2_3, X_2_4], axis=3)
    # X = _conv2D(input, 32, [1, int(0.5 * in_shape[1])], [1, 1], activation=None, padding='same', use_bias=False)
    X = keras.layers.BatchNormalization()(X)
    X = _depth_conv2D(X, 2, [in_shape[0], 1], [1, 1], activation=None, padding='valid', use_bias=False,
                      weight_constraint=weight_constraints_1)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.ELU()(X)
    X = keras.layers.AveragePooling2D(pool_size=(1, 2), strides=None, padding='valid')(X)
    X = keras.layers.Dropout(rate=0.5)(X)
    ### Block 2 ###
    X = _mbconv(X, ch_out=64, kern=(1, 3), t=0.5, reduction=False, SE=0.25, dropout=0.5)
    X = _mbconv(X, ch_out=64, kern=(1, 3), t=0.5, reduction=True, SE=0.25, dropout=0.5)
    X = _mbconv(X, ch_out=64, kern=(1, 3), t=0.5, reduction=False, SE=0.25, dropout=0.5)

    # X = _depth_conv2D(X, 1, [1, 4], [1, 1], activation=None, padding='same', use_bias=False,
    #                   weight_constraint=weight_constraints_1)
    X = _conv2D(X, 32, [1, X.shape[-2]], [1, 1], activation=None, padding='valid', use_bias=False)

    # X = _conv2D(X, 8, [1, 1], [1, 1], activation=None, padding='same', use_bias=False)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.ELU()(X)
    # X = keras.layers.AveragePooling2D(pool_size=(1, 2), strides=None, padding='valid')(X)
    # X = keras.layers.Dropout(rate=0.5)(X)
    # X = _depth_conv2D(X, 2, [in_shape[0], 1], [1, 1], activation=None, padding='valid', use_bias=False,
    #                   weight_constraint=weight_constraints_1)
    ### Final ###
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dropout(rate=0.25)(X)
    if out_shape > 1:
        activation = 'softmax'
        acc = tf.keras.metrics.CategoricalAccuracy()
    else:
        activation = 'sigmoid'
        acc = tf.keras.metrics.AUC(num_thresholds=200, curve='ROC', summation_method='interpolation')
    X = keras.layers.Dense(out_shape, kernel_initializer=kernel_initializer, use_bias=False,
                           kernel_constraint=weight_constraints_2,
                           kernel_regularizer=None, activation=activation)(X)
    model = keras.models.Model(input, X, name="eegnet_2")
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  # loss=keras.losses.MeanSquaredError(),
                  loss=keras.losses.BinaryCrossentropy(),
                  # loss=keras.losses.CategoricalCrossentropy(),
                  metrics=acc)
    return model


def _effnetV2(in_shape, out_shape, loss_weights=[0.5, 0.5]):
    kernel_initializer = tf.initializers.GlorotUniform()
    input = keras.layers.Input(shape=in_shape)
    X = _conv2D(input, n_ch=64, k_size=(1, 3), strides=(1, 2), activation=None, padding='same', use_bias=False)
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
    P_val = []
    N_val = []
    threshold = 0.05
    if len(Y_pred[0]) > 1:
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
    else:
        for i in range(len(Y_pred)):
            if Y_true[i] == 1:
                P = P + 1
                P_val.append(float(Y_pred[i, 0]))
                if Y_pred[i] > threshold:
                    TP = TP + 1
                else:
                    FN = FN + 1
            else:
                N = N + 1
                N_val.append(float(Y_pred[i, 0]))
                if Y_pred[i] < threshold:
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
    return out, P_val, N_val


def _confusion_matrix_2(Y_pred, Y_true):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    if len(Y_pred[0]) > 1:
        for i in range(np.shape(Y_pred)[0]):
            if Y_true[i, 0] == 1:
                if Y_pred[i, 0] > Y_pred[i, 1]:
                    TP = TP + 1
                else:
                    FN = FN + 1
            else:
                if Y_pred[i, 0] < Y_pred[i, 1]:
                    TN = TN + 1
                else:
                    FP = FP + 1
    return TP, TN, FP, FN
# test
# model = _cecotti_cnn1([64, 192, 1], 2)
# model = _eegnet([60, 250, 1], 2)
# model = _effnetV2([1, 250, 60], 2)
