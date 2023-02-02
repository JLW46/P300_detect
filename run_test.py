import json
import keras.backend
import numpy as np
import os
import scipy
import mne
import matplotlib.pyplot as plt
import util_preprocessing
import util_tf

# TRAIN = [
#     '01_01.set',
#     '01_02.set',
#     '01_03.set',
#     '01_04.set',
#     '01_05.set',
#     '01_06.set',
# ]

# TRAIN = [
#     '02_01.set',
#     '02_02.set',
#     '02_03.set',
#     '02_04.set',
#     '02_05.set',
#     '02_06.set',
# ]

# TRAIN = [
#     '03_01.set',
#     '03_02.set',
#     '03_03.set',
#     '03_04.set',
#     '03_05.set',
#     '03_06.set',
# ]

# TRAIN = [
#     '04_01.set',
#     '04_02.set',
#     '04_03.set',
#     '04_04.set',
#     '04_05.set',
#     '04_06.set',
# ]

# TRAIN = [
#     '06_01.set',
#     '06_02.set',
#     '06_03.set',
#     '06_04.set',
#     '06_05.set',
#     '06_06.set',
# ]

# TRAIN = [
#     '07_01.set',
#     '07_02.set',
#     '07_03.set',
#     '07_04.set',
#     '07_05.set',
#     '07_06.set',
# ]

# TRAIN = [
#     '08_01.set',
#     '08_02.set',
#     '08_03.set',
#     '08_04.set',
#     '08_05.set',
#     '08_06.set',
# ]

TRAIN = [
    '09_01.set',
    '09_02.set',
    '09_03.set',
    '09_04.set',
    '09_05.set',
    '09_06.set',
]

CLASS = {
        '4': [0], # nt estim
        '8': [1], # t estim
        '16': [0], # nt astim
        '32': [0], # t astim
        '64': [0], # nt vstim
        '128': [0] # t vstim
    }

FOLDER = r'D:/Code/PycharmProjects/P300_detect/data/SEP BCI 125 0-20 no ICA'
max_iter = 150
converge_threshold = 0.0001
for item in TRAIN:
    TEST = [item]
    X_train, Y_train, X_test, Y_test, class_weights, events_train, \
    sample_weights_train, sample_weights_test = util_preprocessing._build_dataset_eeglab(FOLDER=FOLDER, CLASS=CLASS,
                                                                                         TRAIN=TRAIN, TEST=TEST,
                                                                                         ch_last=False,
                                                                                         trainset_ave=3,
                                                                                         testset_ave=3,
                                                                                         for_plot=False)
    print(np.shape(X_train))
    print(np.shape(Y_train))
    print(np.shape(X_test))
    print(np.shape(Y_test))

    best_loss = 1
    best_acc = 0
    acc_old = 0
    record_iter = []
    record_loss_train = []
    record_loss_test = []
    record_acc_train = []
    record_acc_test = []
    record_P_val = []
    record_N_val = []
    dict_save = {}

    keras.backend.clear_session()
    model = util_tf._eegnet_1(in_shape=np.shape(X_train)[-3:], out_shape=np.shape(Y_train)[-1])

    for i in range(max_iter):
        record_iter.append(i)
        hist = model.fit(x=X_train, y=Y_train, epochs=1, batch_size=16)
        loss_train = hist.history['loss'][-1]
        acc_train = hist.history['auc'][-1]
        pred = model.evaluate(x=X_test, y=Y_test)
        loss_test = pred[0]
        acc_test = pred[1]
        record_loss_train.append(loss_train)
        record_loss_test.append(loss_test)
        record_acc_train.append(acc_train)
        record_acc_test.append(acc_test)
        Y_pred = model.predict(x=X_test)
        results, P_val, N_val = util_tf._confusion_matrix(Y_pred, Y_test)
        record_P_val.append(P_val)
        record_N_val.append(N_val)
        if loss_test < best_loss:
            best_loss = loss_test
        if acc_test > best_acc:
            model.save('results_noICA_epoch_3/' + TEST[0].split('.')[0] + '_iter_' + str(i))
            best_acc = acc_test
        print('Iter: ' + str(i) + '. TP: ' + str(results['matrix'][0, 0]) + '. TN: ' + str(results['matrix'][1, 1])
              + '. Test loss: ' + str(loss_test) + '. Test acc: ' + str(acc_test))
        if i > 25:
            if np.max(np.array(record_loss_train[-10:])) < converge_threshold:
                break
    dict_save = {
            'best_test_acc': best_acc,
            'iter': record_iter,
            'train_loss': record_loss_train,
            'train_acc': record_acc_train,
            'test_loss': record_loss_test,
            'test_acc': record_acc_test,
            'test_P_val': record_P_val,
            'test_N_val': record_N_val
        }
    with open('results_noICA_epoch_3/' + TEST[0].split('.')[0] + '.json', "w") as json_file:
        json.dump(dict_save, json_file)
    print('best: ')
    print(best_loss)
    print(best_acc)

print('DONE!')