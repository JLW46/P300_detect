import json
import keras.backend
import numpy as np
import os
import scipy
import mne
import matplotlib.pyplot as plt
import sklearn.metrics

import util_preprocessing
import util_tf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

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

TRAIN = [
    '07_01.set',
    '07_02.set',
    '07_03.set',
    '07_04.set',
    '07_05.set',
    '07_06.set',
]

# TRAIN = [
#     '08_01.set',
#     '08_02.set',
#     '08_03.set',
#     '08_04.set',
#     '08_05.set',
#     '08_06.set',
# ]

# TRAIN = [
#     '09_01.set',
#     '09_02.set',
#     '09_03.set',
#     '09_04.set',
#     '09_05.set',
#     '09_06.set',
# ]


# TRAIN = [
#     # '01_01.set',
#     # '01_02.set',
#     # '01_03.set',
#     # '01_04.set',
#     # '01_05.set',
#     # '01_06.set',
#     '02_01.set',
#     '02_02.set',
#     '02_03.set',
#     '02_04.set',
#     '02_05.set',
#     '02_06.set',
#     '03_01.set',
#     '03_02.set',
#     '03_03.set',
#     '03_04.set',
#     '03_05.set',
#     '03_06.set',
#     '04_01.set',
#     '04_02.set',
#     '04_03.set',
#     '04_04.set',
#     '04_05.set',
#     '04_06.set',
#     '06_01.set',
#     '06_02.set',
#     '06_03.set',
#     '06_04.set',
#     '06_05.set',
#     '06_06.set',
#     '07_01.set',
#     '07_02.set',
#     '07_03.set',
#     '07_04.set',
#     '07_05.set',
#     '07_06.set',
#     '08_01.set',
#     '08_02.set',
#     '08_03.set',
#     '08_04.set',
#     '08_05.set',
#     '08_06.set',
#     '09_01.set',
#     '09_02.set',
#     '09_03.set',
#     '09_04.set',
#     '09_05.set',
#     '09_06.set',
# ]

TEST = [
    # '01_01.set',
#     '01_02.set',
#     '01_03.set',
#     '01_04.set',
#     '01_05.set',
#     '01_06.set',
#     '02_01.set',
    # '02_02.set',
    # '02_03.set',
    # '02_04.set',
    # '02_05.set',
    # '02_06.set',
    # '03_01.set',
    # '03_02.set',
    # '03_03.set',
    # '03_04.set',
    # '03_05.set',
    # '03_06.set',
    # '04_01.set',
    # '04_02.set',
    # '04_03.set',
    # '04_04.set',
    # '04_05.set',
    # '04_06.set',
    # '06_01.set',
    # '06_02.set',
    # '06_03.set',
    # '06_04.set',
    # '06_05.set',
    # '06_06.set',
    # '07_01.set',
    # '07_02.set',
    # '07_03.set',
    # '07_04.set',
    # '07_05.set',
    # '07_06.set',
    # '08_01.set',
    # '08_02.set',
    # '08_03.set',
    # '08_04.set',
    # '08_05.set',
    # '08_06.set',
    # '09_01.set',
    # '09_02.set',
    # '09_03.set',
    # '09_04.set',
    # '09_05.set',
    # '09_06.set',
]



FOLDER = r'D:/Code/PycharmProjects/P300_detect/data/SEP BCI 125 0-20 no ICA'
# FOLDER = r'D:/Code/PycharmProjects/P300_detect/data/SEP BCI 125 0-20 ICA'
def _run_cnn_test():
    # out_len = 1, AUC
    CLASS = {
            '4': [0], # nt estim
            '8': [1], # t estim
            '16': [0], # nt astim
            '32': [0], # t astim
            '64': [0], # nt vstim
            '128': [0] # t vstim
        }
    max_iter = 20
    converge_threshold = 0.0001
    # for item in TRAIN:
    if True:
        # TEST = [item]
        # X_train = [n_sample, eeg_ch, time_series, 1]
        # X_train = [n_sample, 1, time_series, eeg_ch]
        # Y_test = [n_sample, 1]
        X_train, Y_train, X_test, Y_test, class_weights, events_train, \
        sample_weights_train, sample_weights_test = util_preprocessing._build_dataset_eeglab(FOLDER=FOLDER, CLASS=CLASS,
                                                                                             TRAIN=TRAIN, TEST=TEST,
                                                                                             ch_last=False,
                                                                                             trainset_ave=1,
                                                                                             testset_ave=1)
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
        model = util_tf._eegnet_2(in_shape=np.shape(X_train)[-3:], out_shape=np.shape(Y_train)[-1])
        for i in range(max_iter):
            record_iter.append(i)
            hist = model.fit(x=X_train, y=Y_train, epochs=50, batch_size=32)
            model.save('temp_model')
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
                model.save('results_ICA_epoch_3/' + TEST[0].split('.')[0] + '_iter_' + str(i))
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
        with open('results_ICA_epoch_3/' + TEST[0].split('.')[0] + '.json', "w") as json_file:
            json.dump(dict_save, json_file)
        print('best: ')
        print(best_loss)
        print(best_acc)

    print('DONE!')

def _run_cnn_test2(epochs=1):
    # out_len = 2, CategoricalCrossEntropy
    CLASS = {
        '4': [0],  # nt estim
        '8': [1],  # t estim
        '16': [0],  # nt astim
        '32': [0],  # t astim
        '64': [0],  # nt vstim
        '128': [0]  # t vstim
    }
    # CH_SELECT = [9, 27, 45, 59, 43, 47, 50, 56]
    # CH_SELECT = [9, 27, 45]
    CH_SELECT = False
    for sbj in ['01', '02', '03', '04', '06', '07', '08', '09']:
    # for sbj in ['04', '06', '07', '08', '09']:
        # create TRAIN
        TRAIN = []
        for set in ['_01', '_02', '_03', '_04', '_05', '_06']:
            TRAIN.append(sbj + set + '.set')
        # run
        for item in TRAIN:
        # if True:
        #     TEST = ['07_01.set']
            TEST = [item]
            X_train, Y_train, X_test, Y_test, class_weights, events_train, \
            sample_weights_train, sample_weights_test = util_preprocessing._build_dataset_eeglab(FOLDER=FOLDER, CLASS=CLASS,
                                                                                                 TRAIN=TRAIN, TEST=TEST,
                                                                                                 ch_last=False,
                                                                                                 trainset_ave=epochs,
                                                                                                 testset_ave=epochs,
                                                                                                 ch_select=CH_SELECT,
                                                                                                 rep=4)
            print(np.shape(X_train))
            print(np.shape(Y_train))
            print(np.shape(X_test))
            print(np.shape(Y_test))
            print(np.sum(Y_train, axis=0))
            print(np.sum(Y_test, axis=0))

            keras.backend.clear_session()
            callback_1 = util_tf.IterTracker(X_test=X_test, Y_test=Y_test)
            # model = util_tf._eegnet(in_shape=np.shape(X_train)[-3:], out_shape=np.shape(Y_train)[-1])
            # model = util_tf._custom(in_shape=np.shape(X_train)[-3:], out_shape=np.shape(Y_train)[-1])
            # model = util_tf._effnetV2(in_shape=np.shape(X_train)[-3:], out_shape=np.shape(Y_train)[-1])
            model = util_tf._vit(in_shape=np.shape(X_train)[-3:], out_shape=np.shape(Y_train)[-1])
            model.fit(x=X_train, y=Y_train, epochs=200, batch_size=16, callbacks=[callback_1])
            print('---------++++++++++++______________')
            print(callback_1.best_scores)
            print('DONE!')
            # SAVE_PATH = r'results_noICA_loss_eegnet_8ch_epoch_' + str(epochs) + '/'
            # SAVE_PATH = r'results_noICA_loss_custom_8ch_epoch_' + str(epochs) + '/'
            # SAVE_PATH = r'results_noICA_effnetv2_epoch_1/'
            SAVE_PATH = r'results_noICA_loss_vit_0ch_epoch_' + str(epochs) + '/'
            if os.path.exists(SAVE_PATH):
                pass
            else:
                os.makedirs(SAVE_PATH)
            FILE_NAME = SAVE_PATH + TEST[0].split('.')[0] + '.json'
            if os.path.isfile(FILE_NAME):
                with open(FILE_NAME) as json_file:
                    data = json.load(json_file)
                    old_loss = data['loss']
                if callback_1.best_scores['loss'] < old_loss:
                    with open(FILE_NAME, "w") as json_file:
                        json.dump(callback_1.best_scores, json_file)
                    model.set_weights(callback_1.best_weights)
                    # model.save(SAVE_PATH + TEST[0].split('.')[0] + '_iter_' + str(callback_1.best_scores['epoch']))
            else:
                with open(FILE_NAME, "w") as json_file:
                    json.dump(callback_1.best_scores, json_file)
                model.set_weights(callback_1.best_weights)
                # model.save(SAVE_PATH + TEST[0].split('.')[0] + '_iter_' + str(callback_1.best_scores['epoch']))
            # if os.path.isfile(FILE_NAME):
            #     with open(FILE_NAME) as json_file:
            #         data = json.load(json_file)
            #         old_auc = data['auc']
            #     if callback_1.best_scores['auc'] > old_auc:
            #         with open(FILE_NAME, "w") as json_file:
            #             json.dump(callback_1.best_scores, json_file)
            #         model.set_weights(callback_1.best_weights)
            #         model.save(SAVE_PATH + TEST[0].split('.')[0] + '_iter_' + str(callback_1.best_scores['epoch']))
            # else:
            #     with open(FILE_NAME, "w") as json_file:
            #         json.dump(callback_1.best_scores, json_file)
            #     model.set_weights(callback_1.best_weights)
            #     model.save(SAVE_PATH + TEST[0].split('.')[0] + '_iter_' + str(callback_1.best_scores['epoch']))

    return

def _run_csp_lda(display=False, epochs=1):
    CLASS = {
        '4': [0],  # nt estim
        '8': [1],  # t estim
        '16': [0],  # nt astim
        '32': [0],  # t astim
        '64': [0],  # nt vstim
        '128': [0]  # t vstim
    }
    # CH_SELECT = [9, 27, 45, 59, 43, 47, 50, 56]
    # CH_SELECT = [9, 27, 45]
    CH_SELECT = False
    for sbj in ['01', '02', '03', '04', '06', '07', '08', '09']:
    # for sbj in ['08', '09']:
        # create TRAIN
        TRAIN = []
        for set in ['_01', '_02', '_03', '_04', '_05', '_06']:
            TRAIN.append(sbj + set + '.set')
        # run
        for item in TRAIN:
            TEST = [item]
            X_train, Y_train, X_test, Y_test, class_weights, events_train, \
            sample_weights_train, sample_weights_test = util_preprocessing._build_dataset_eeglab(FOLDER=FOLDER, CLASS=CLASS,
                                                                                                 TRAIN=TRAIN, TEST=TEST,
                                                                                                 ch_last=False,
                                                                                                 trainset_ave=epochs,
                                                                                                 testset_ave=epochs,
                                                                                                 ch_select=CH_SELECT,
                                                                                                 rep=4)

            X_train = np.reshape(X_train, (np.shape(X_train)[0], np.shape(X_train)[1], np.shape(X_train)[2]))
            X_test = np.reshape(X_test, (np.shape(X_test)[0], np.shape(X_test)[1], np.shape(X_test)[2]))
            Y_train = np.squeeze(Y_train)
            Y_test = np.squeeze(Y_test)
            print(np.shape(X_train))
            print(np.shape(Y_train))
            print(np.shape(X_test))
            print(np.shape(Y_test))
            lda = LinearDiscriminantAnalysis()
            csp = CSP(n_components=16, reg=None, log=True, norm_trace=False)
            Feature_train = csp.fit_transform(X_train, Y_train)
            lda.fit(Feature_train, Y_train)
            Feature_test = csp.transform(X_test)
            # print(sample_weights_test)
            # print(lda.score(Feature_test, Y_test, sample_weight=sample_weights_test))
            proba = lda.predict_proba(Feature_test)
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=Y_test, y_score=proba[:, 1])
            roc_auc = sklearn.metrics.auc(fpr, tpr)
            best_scores = {
                'loss': 0,
                'auc': roc_auc,
                'Y_pred': np.squeeze(proba[:, 1]).tolist(),
                'Y_true': Y_test.tolist()
            }
            SAVE_PATH = r'results_noICA_loss_csplda_0ch_epoch_' + str(epochs) + '/'
            if os.path.exists(SAVE_PATH):
                pass
            else:
                os.makedirs(SAVE_PATH)
            FILE_NAME = SAVE_PATH + TEST[0].split('.')[0] + '.json'
            if os.path.isfile(FILE_NAME):
                with open(FILE_NAME) as json_file:
                    data = json.load(json_file)
                    old_auc = data['auc']
                if best_scores['auc'] > old_auc:
                    with open(FILE_NAME, "w") as json_file:
                        json.dump(best_scores, json_file)
            else:
                with open(FILE_NAME, "w") as json_file:
                    json.dump(best_scores, json_file)
    # if display is True:
    #     # fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=Y_test, y_score=proba[:, 1], sample_weight=sample_weights_test)
    #     fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=Y_test, y_score=proba[:, 1])
    #     roc_auc = sklearn.metrics.auc(fpr, tpr)
    #     display = sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    #     display.plot()
    #     plt.show()

    return

for i in [1, 2, 3, 4, 5, 6]:
# for i in [5, 6]:
#     _run_csp_lda(display=False, epochs=i)
    _run_cnn_test2(epochs=i)
# _run_cnn_test2(epochs=6)