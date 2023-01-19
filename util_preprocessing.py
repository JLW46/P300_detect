import numpy as np
import scipy
import os
import mne
import random
import matplotlib.pyplot as plt


P300_SPELLER = {
    'A'
}


def _build_dataset_p300(PATH, WIN, CH, epochs=1, ch_last=True):
    TARGET = np.array([1, 0])
    NONTARGET = np.array([0, 1])
    data_pkg = {}
    data = scipy.io.loadmat(PATH)
    Signal = data['Signal']
    Flashing = data['Flashing']
    StimType = data['StimulusType']
    X = []
    X_norm = []
    Y = []
    grand_ave_target = []
    grand_ave_nontarget = []
    target_ind = []
    for i in range(np.shape(Flashing)[0]):
        X_sample = []
        X_sample_norm = []
        Y_sample = []
        for j in range(np.shape(Flashing)[1] - 1):
            if Flashing[i, j + 1] == 1 and Flashing[i, j] == 0:
                x = np.reshape(Signal[i, j + 1:j + WIN + 1, :], (WIN, CH)).T
                x_norm = (x - np.repeat(np.reshape(np.mean(x, axis=1), (CH, 1)), np.shape(x)[1], axis=1)) / (
                        np.repeat(np.reshape(np.std(x, axis=1), (CH, 1)), np.shape(x)[1], axis=1))
                if ch_last:
                    X_sample.append(np.reshape(x.T, (1, WIN, CH)))
                    X_sample_norm.append(np.reshape(x_norm.T, (1, WIN, CH)))
                else:
                    X_sample.append(np.reshape(x, (CH, WIN, 1)))
                    X_sample_norm.append(np.reshape(x_norm, (CH, WIN, 1)))
                target_ind.append(i)
                if StimType[i, j + 1] == 1:
                    grand_ave_target.append(X_sample[-1])
                    Y_sample.append(TARGET)
                else:
                    grand_ave_nontarget.append(X_sample[-1])
                    Y_sample.append(NONTARGET)
        X.append(np.array(X_sample))
        X_norm.append(np.array(X_sample_norm))
        Y.append(np.array(Y_sample))
    X = np.array(X)
    X_norm = np.array(X_norm)
    Y = np.array(Y)
    target_ind = np.array(target_ind)
    if epochs > 1 and epochs <= 15:
        X_multi_epochs = []
        X_multi_epochs_norm = []
        Y_multi_epochs = []
        for i in range(np.shape(X)[0]):
            x_target = []
            x_target_norm = []
            x_target_cnt = 0
            x_nontarget = []
            x_nontarget_norm = []
            x_nontarget_cnt = 0
            for j in range(np.shape(X)[1]):
                if Y[i, j, 0] == 1:
                    # target
                    x_target_cnt = x_target_cnt + 1
                    x_target.append(X[i, j, :, :, :])
                    x_target_norm.append(X_norm[i, j, :, :, :])
                else:
                    # nontarget
                    x_nontarget_cnt = x_nontarget_cnt + 1
                    x_nontarget.append(X[i, j, :, :, :])
                    x_nontarget_norm.append(X_norm[i, j, :, :, :])
                if x_target_cnt >= epochs*2 and x_nontarget_cnt >= epochs*10:
                    X_multi_epochs.append(np.mean(np.array(x_target), axis=0))
                    X_multi_epochs_norm.append(np.mean(np.array(x_target_norm), axis=0))
                    Y_multi_epochs.append(TARGET)
                    X_multi_epochs.append(np.mean(np.array(x_nontarget), axis=0))
                    X_multi_epochs_norm.append(np.mean(np.array(x_nontarget_norm), axis=0))
                    Y_multi_epochs.append(NONTARGET)
                    break
        X = np.array(X_multi_epochs)
        X_norm = np.array(X_multi_epochs_norm)
        Y = np.array(Y_multi_epochs)
    data_pkg['X'] = X
    data_pkg['X_norm'] = X_norm
    data_pkg['Y'] = Y
    data_pkg['target_ind'] = target_ind
    data_pkg['grand_ave_target'] = np.mean(np.array(grand_ave_target), axis=0)
    data_pkg['grand_ave_nontarget'] = np.mean(np.array(grand_ave_nontarget), axis=0)

    return data_pkg


def _build_dataset_p300_test(PATH, WIN, CH, LABEL, epochs=1, ch_last=True):
    TARGET = np.array([1, 0])
    NONTARGET = np.array([0, 1])
    data_pkg = {}
    data = scipy.io.loadmat(PATH)
    Signal = data['Signal']
    Flashing = data['Flashing']
    StimCode = data['StimulusCode']
    X = []
    X_norm = []
    Y = []
    target_ind = []
    for i in range(np.shape(Flashing)[0]):
        X_sample = []
        X_sample_norm = []
        Y_sample = []
        r, c = _P300_speller(LABEL[i])
        for j in range(np.shape(Flashing)[1] - 1):
            if Flashing[i, j + 1] == 1 and Flashing[i, j] == 0:
                x = np.reshape(Signal[i, j + 1:j + WIN + 1, :], (WIN, CH)).T
                x_norm = (x - np.repeat(np.reshape(np.mean(x, axis=1), (CH, 1)), np.shape(x)[1], axis=1)) / (
                        np.repeat(np.reshape(np.std(x, axis=1), (CH, 1)), np.shape(x)[1], axis=1))
                if ch_last:
                    X_sample.append(np.reshape(x.T, (1, WIN, CH)))
                    X_sample_norm.append(np.reshape(x_norm.T, (1, WIN, CH)))
                else:
                    X_sample.append(np.reshape(x, (CH, WIN, 1)))
                    X_sample_norm.append(np.reshape(x_norm, (CH, WIN, 1)))
                target_ind.append(i)
                if int(StimCode[i, j + 1]) == int(r) or int(StimCode[i, j + 1]) == int(c):
                    Y_sample.append(TARGET)
                else:
                    Y_sample.append(NONTARGET)
        X.append(np.array(X_sample))
        X_norm.append(np.array(X_sample_norm))
        Y.append(np.array(Y_sample))
    X = np.array(X)
    X_norm = np.array(X_norm)
    Y = np.array(Y)
    target_ind = np.array(target_ind)
    if epochs > 1 and epochs <= 15:
        X_multi_epochs = []
        X_multi_epochs_norm = []
        Y_multi_epochs = []
        for i in range(np.shape(X)[0]):
            x_target = []
            x_target_norm = []
            x_target_cnt = 0
            x_nontarget = []
            x_nontarget_norm = []
            x_nontarget_cnt = 0
            for j in range(np.shape(X)[1]):
                if Y[i, j, 0] == 1:
                    # target
                    x_target_cnt = x_target_cnt + 1
                    x_target.append(X[i, j, :, :, :])
                    x_target_norm.append(X_norm[i, j, :, :, :])
                else:
                    # nontarget
                    x_nontarget_cnt = x_nontarget_cnt + 1
                    x_nontarget.append(X[i, j, :, :, :])
                    x_nontarget_norm.append(X_norm[i, j, :, :, :])
                if x_target_cnt >= epochs*2 and x_nontarget_cnt >= epochs*10:
                    X_multi_epochs.append(np.mean(np.array(x_target), axis=0))
                    X_multi_epochs_norm.append(np.mean(np.array(x_target_norm), axis=0))
                    Y_multi_epochs.append(TARGET)
                    X_multi_epochs.append(np.mean(np.array(x_nontarget), axis=0))
                    X_multi_epochs_norm.append(np.mean(np.array(x_nontarget_norm), axis=0))
                    Y_multi_epochs.append(NONTARGET)
                    break
        X = np.array(X_multi_epochs)
        X_norm = np.array(X_multi_epochs_norm)
        Y = np.array(Y_multi_epochs)
    data_pkg['X'] = X
    data_pkg['X_norm'] = X_norm
    data_pkg['Y'] = Y
    data_pkg['target_ind'] = target_ind

    return data_pkg


def _P300_speller(char):
    # SPELLER = [['A', 'B', 'C', 'D', 'E', 'F'],
    #            ['G', 'H', 'I', 'J', 'K', 'L'],
    #            ['M', 'N', 'O', 'P', 'Q', 'R'],
    #            ['S', 'T', 'U', 'V', 'W', 'X'],
    #            ['Y', 'Z', '1', '2', '3', '4'],
    #            ['5', '6', '7', '8', '9', '_']]
    SPELLER = ['ABCDEF',
               'GHIJKL',
               'MNOPQR',
               'STUVWX',
               'YZ1234',
               '56789_']
    r = 7
    for row in SPELLER:
        if char in row:
            c = row.find(char) + 1
            break
        else:
            r = r + 1
    return r, c


def _build_dataset_eeglab(FOLDER, TRAIN, TEST, CLASS, ch_last=False,
                          trainset_ave=False, trainset_copy=False, testset_ave=False, for_plot=False):
    files = os.listdir(FOLDER)
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    events_train = None
    events_test = None
    for file_name in files:
        # if file_name.endswith('.set'):
        if file_name in TRAIN or file_name in TEST:
            PATH = os.path.join(FOLDER, file_name)
            X, Y, events = _read_data_eeglab(PATH=PATH, CLASS=CLASS, epochs=1, ch_last=ch_last, norm=True)
            if file_name in TRAIN:
                if trainset_ave is not False:
                    X, Y = _random_average(X, Y, fold=trainset_ave)
                elif trainset_copy is True:
                    X, Y = _copy_balance(X, Y)
                if X_train is None:
                    X_train = X
                    Y_train = Y
                    events_train = events
                else:
                    X_train = np.concatenate([X_train, X], axis=0)
                    Y_train = np.concatenate([Y_train, Y], axis=0)
                    events_train = np.concatenate([events_train, events])
            elif file_name in TEST:
                if testset_ave is not False:
                    X, Y = _consec_average(X, Y, epochs=testset_ave)
                if X_test is None:
                    X_test = X
                    Y_test = Y
                    events_test = events
                else:
                    X_test = np.concatenate([X_test, X], axis=0)
                    Y_test = np.concatenate([Y_test, Y], axis=0)
                    events_test = np.concatenate([events_test, events])
    summation = np.sum(Y_train, axis=0)
    summation_test = np.sum(Y_test, axis=0)
    class_weights = []
    class_weights_test = []
    for i in range(len(summation)):
        w_inv = 0
        for j in range(len(summation)):
            w_inv = w_inv + summation[i]/summation[j]
        class_weights.append(1/w_inv)
    for i in range(len(summation_test)):
        w_inv = 0
        for j in range(len(summation_test)):
            w_inv = w_inv + summation_test[i]/summation_test[j]
        class_weights_test.append(1/w_inv)
    class_weights = np.array(class_weights)
    class_weights_test = np.array(class_weights_test)
    # loss_weights = loss_weights/np.sum(loss_weights)
    sample_weights_train = np.ones(np.shape(Y_train)[0])
    sample_weights_train[np.where(Y_train[:, 0] == 1)[0]] = class_weights[0]
    sample_weights_train[np.where(Y_train[:, 1] == 1)[0]] = class_weights[1]
    sample_weights_test = np.ones(np.shape(Y_test)[0])
    sample_weights_test[np.where(Y_test[:, 0] == 1)[0]] = class_weights_test[0]
    sample_weights_test[np.where(Y_test[:, 1] == 1)[0]] = class_weights_test[1]
    if for_plot:
        X_train = np.concatenate([X_train, X_test], axis=0)
        Y_train = np.concatenate([Y_train, Y_test], axis=0)
        events_train = np.concatenate([events_train, events_test])
    return X_train, Y_train, X_test, Y_test, class_weights, events_train, sample_weights_train, sample_weights_test


def _get_event(events, event_id):
    out = events[:, 2]
    key_list = list(event_id.keys())
    val_list = list(event_id.values())
    for i in range(len(out)):
        out[i] = int(key_list[val_list.index(out[i])].split('/')[0])
    return out


def _read_data_eeglab(PATH, CLASS, epochs=1, ch_last=False, norm=True):
    data_pkg = mne.read_epochs_eeglab(PATH)
    events = _get_event(data_pkg.events, data_pkg.event_id)
    data = data_pkg._data
    CH = np.shape(data)[1]
    T = np.shape(data)[2]
    X = []
    X_norm = []
    Y = []
    for i in range(np.shape(data)[0]):
        Y.append(CLASS[str(events[i])])
        x = data[i, :, :]
        if norm:
            # remove baseline by [-0.2, 0.0]s
            x_norm = (x - np.repeat(np.reshape(np.mean(x[:, :50], axis=1), (CH, 1)), T, axis=1))/(
                    np.repeat(np.reshape(np.std(x[:, -175:], axis=1), (CH, 1)), T, axis=1))
            x = x_norm
            # x = x[:, -175] #######
        if ch_last:
            X.append(np.reshape(x.T, (1, T, CH)))
        else:
            X.append(np.reshape(x, (CH, T, 1)))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y, events


def _random_average(X, Y, fold=1):
    target_ind = list(np.where(Y[:, 0]==1)[0])
    X_new = []
    Y_new = []
    for i in range(fold):
        for j in range(len(target_ind)):
            ind = random.sample(target_ind, i + 2)
            X_new.append(np.mean(X[ind], axis=0))
            Y_new.append(Y[ind[0], :])
    X = np.concatenate([X, np.array(X_new)], axis=0)
    Y = np.concatenate([Y, np.array(Y_new)], axis=0)

    return X, Y


def _copy_balance(X, Y):
    target_ind = list(np.where(Y[:, 0] == 1)[0])
    label_sum = np.sum(Y, axis=0)
    fold = round(label_sum[1]/label_sum[0])
    X = np.concatenate([X, np.repeat(X[target_ind], fold - 1, axis=0)], axis=0)
    Y = np.concatenate([Y, np.repeat(Y[target_ind], fold - 1, axis=0)], axis=0)
    return X, Y


def _consec_average(X, Y, epochs=2):
    target_ind = list(np.where(Y[:, 0] == 1)[0])
    nontarget_ind = list(np.where(Y[:, 1] == 1)[0])
    X_new = []
    Y_new = []
    time_axis = np.linspace(-0.2, 0.8, 1 * 250, endpoint=False)
    for i in range(len(target_ind)):
        if (i + epochs) <= len(target_ind):
            ind = target_ind[i: i + epochs]
            # print(ind)
            # for k in range(len(target_ind)):
            #     plt.plot(time_axis, X[target_ind[k], 0, :, 31], 'b')
            # plt.plot(time_axis, np.mean(X[ind], axis=0)[0, :, 31], 'r')
            # plt.grid()
            # plt.show()
            X_new.append(np.mean(X[ind], axis=0))
            Y_new.append(Y[ind[0], :])
    X = np.concatenate([X[nontarget_ind], np.array(X_new)], axis=0)
    Y = np.concatenate([Y[nontarget_ind], np.array(Y_new)], axis=0)

    return X, Y