import numpy as np
import scipy
import os
import mne
import random
import itertools
import matplotlib.pyplot as plt
import json

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
                          trainset_ave=False, testset_ave=False, ch_select=False, rep=1):
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
            # X, Y, events = _read_data_eeglab(PATH=PATH, CLASS=CLASS, ch_last=ch_last, norm=True, epochs=3, sampling='random')
            if (file_name in TRAIN) and (file_name not in TEST):
                X, Y, events = _read_data_eeglab(PATH=PATH, CLASS=CLASS, ch_last=ch_last, norm=True, epochs=trainset_ave,
                                                 sampling='random', plot=False, ch_select=ch_select, rep=rep)
                if X_train is None:
                    X_train = X
                    Y_train = Y
                    events_train = events
                else:
                    X_train = np.concatenate([X_train, X], axis=0)
                    Y_train = np.concatenate([Y_train, Y], axis=0)
                    events_train = np.concatenate([events_train, events])
            elif file_name in TEST:
                X, Y, events = _read_data_eeglab(PATH=PATH, CLASS=CLASS, ch_last=ch_last, norm=True, epochs=testset_ave,
                                                 sampling='consecutive', plot=False, ch_select=ch_select, rep=1)
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
    sample_weights_train = []
    sample_weights_test = []
    # if len(np.shape(summation)) > 1:
    #     for i in range(len(summation)):
    #         w_inv = 0
    #         for j in range(len(summation)):
    #             w_inv = w_inv + summation[i]/summation[j]
    #         class_weights.append(1/w_inv)
    #     for i in range(len(summation_test)):
    #         w_inv = 0
    #         for j in range(len(summation_test)):
    #             w_inv = w_inv + summation_test[i]/summation_test[j]
    #         class_weights_test.append(1/w_inv)
    #     class_weights = np.array(class_weights)
    #     class_weights_test = np.array(class_weights_test)
    #     # loss_weights = loss_weights/np.sum(loss_weights)
    #     sample_weights_train = np.ones(np.shape(Y_train)[0])
    #     sample_weights_train[np.where(Y_train[:, 0] == 1)[0]] = class_weights[0]
    #     sample_weights_train[np.where(Y_train[:, 1] == 1)[0]] = class_weights[1]
    #     sample_weights_test = np.ones(np.shape(Y_test)[0])
    #     sample_weights_test[np.where(Y_test[:, 0] == 1)[0]] = class_weights_test[0]
    #     sample_weights_test[np.where(Y_test[:, 1] == 1)[0]] = class_weights_test[1]
    # else:
    #     t_cnt = 0
    #     nt_cnt = 0
    #     for i in range(np.shape(Y_test)[0]):
    #         if Y_test[i] == 0:
    #             nt_cnt = nt_cnt + 1
    #         else:
    #             t_cnt = t_cnt + 1
    #     for i in range(np.shape(Y_test)[0]):
    #         if Y_test[i] == 0:
    #             sample_weights_test.append(t_cnt/(t_cnt + nt_cnt))
    #         else:
    #             sample_weights_test.append(nt_cnt/(t_cnt + nt_cnt))
    return X_train, Y_train, X_test, Y_test, class_weights, events_train, sample_weights_train, sample_weights_test
    # return X_train, Y_train, X_test, Y_test, events_train

def _build_dataset_strat2(FOLDER, TRAIN, TEST, CLASS, ch_select=False, rep_train=1, rep_test=1, mult=1, from_rep0=False):
    files = os.listdir(FOLDER)
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    events_train = None
    events_test = None
    for file_name in files:
        if file_name in TRAIN or file_name in TEST:
            PATH = os.path.join(FOLDER, file_name)
            if (file_name in TRAIN) and (file_name not in TEST):
                X, Y, events = _read_data_strat2(PATH=PATH, CLASS=CLASS,
                                                 ch_select=ch_select, norm=True, plot=False,
                                                 test=False, from_rep0=from_rep0,
                                                 num_reps_train=rep_train, num_reps_test=rep_test, mult=mult)
                if X_train is None:
                    X_train = X
                    Y_train = Y
                    events_train = events
                else:
                    X_train = np.concatenate([X_train, X], axis=0)
                    Y_train = np.concatenate([Y_train, Y], axis=0)
                    events_train = np.concatenate([events_train, events])
            elif file_name in TEST:
                X, Y, events = _read_data_strat2(PATH=PATH, CLASS=CLASS,
                                                 ch_select=ch_select, norm=True, plot=False,
                                                 test=True, from_rep0=from_rep0,
                                                 num_reps_train=rep_train, num_reps_test=rep_test, mult=mult)
                if X_test is None:
                    X_test = X
                    Y_test = Y
                    events_test = events
                else:
                    X_test = np.concatenate([X_test, X], axis=0)
                    Y_test = np.concatenate([Y_test, Y], axis=0)
                    events_test = np.concatenate([events_test, events])

    return X_train, Y_train, X_test, Y_test


def _build_dataset_strat3(FOLDER, TRAIN, TEST, num_reps=1):
    files = os.listdir(FOLDER)
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    X_test_ext = None
    Y_test_ext = None
    events_train = None
    events_test = None
    for file_name in files:
        if file_name in TRAIN or file_name in TEST:
            PATH = os.path.join(FOLDER, file_name)
            if (file_name in TRAIN) and (file_name not in TEST):
                out = _read_data_strat3(PATH=PATH, norm=True, plot=False, test=False,
                                        num_reps=num_reps)
                if X_train is None:
                    X_train = out['X_train']
                    Y_train = out['Y_train']
                else:
                    X_train = np.concatenate([X_train, out['X_train']], axis=0)
                    Y_train = np.concatenate([Y_train, out['Y_train']], axis=0)
            elif file_name in TEST:
                out = _read_data_strat3(PATH=PATH, norm=True, plot=False, test=True,
                                        num_reps=num_reps)
                if X_test is None:
                    X_test = out['X_test']
                    Y_test = out['Y_test']
                    X_test_ext = out['X_test_ext']
                    Y_test_ext = out['Y_test_ext']
                else:
                    X_test = np.concatenate([X_test, out['X_test']], axis=0)
                    Y_test = np.concatenate([Y_test, out['Y_test']], axis=0)
                    X_test_ext = np.concatenate([X_test, out['X_test_ext']], axis=0)
                    Y_test_ext = np.concatenate([Y_test, out['Y_test_ext']], axis=0)

    return X_train, Y_train, X_test, Y_test, X_test_ext, Y_test_ext


def _build_dataset_eeglab_plot(FOLDER, TRAIN, TEST, CLASS):
    files = os.listdir(FOLDER)
    X_out = None
    Y_out = None
    events_out = None
    for file_name in files:
        if file_name in TRAIN or file_name in TEST:
            PATH = os.path.join(FOLDER, file_name)
            X, Y, events = _read_data_eeglab(PATH=PATH, CLASS=CLASS, ch_last=False, norm=True, epochs=1,
                                             sampling='consecutive', plot=True, ch_select=False, rep=1)
            if X_out is None:
                X_out = X
                Y_out = Y
                events_out = events
            else:
                X_out = np.concatenate([X_out, X], axis=0)
                Y_out = np.concatenate([Y_out, Y], axis=0)
                events_out = np.concatenate([events_out, events])
    print('X_shape:' + str(np.shape(X_out)))
    print('Y_shape:' + str(np.shape(Y_out)))
    print('Events:' + str(np.shape(events_out)))
    return X_out, Y_out, events_out

def _get_event(events, event_id):
    out = events[:, 2]
    key_list = list(event_id.keys())
    val_list = list(event_id.values())
    for i in range(len(out)):
        out[i] = int(key_list[val_list.index(out[i])].split('/')[0])
    return out

def _read_data_eeglab(PATH, CLASS, ch_last=False, norm=True, epochs=1, sampling='consecutive', plot=False, ch_select=False, rep=1):
    data_pkg = mne.read_epochs_eeglab(PATH)
    events = _get_event(data_pkg.events, data_pkg.event_id)
    data = data_pkg._data
    if ch_select is not False:
        CH = len(ch_select)
    else:
        CH = np.shape(data)[1]
    T = np.shape(data)[2]
    # T_re = 175
    # T_re = T
    T_re = 100
    X = []
    Y = []
    events_new = []
    for i in range(np.shape(data)[0]):
        Y.append(CLASS[str(events[i])])
        x = data[i, :, :]
        if ch_select is False:
            pass
        else:
            x = x[ch_select, :]
        if norm:
            # remove baseline by [-0.2, 0.0]s
            x_norm = (x - np.repeat(np.reshape(np.mean(x[:, :25], axis=1), (CH, 1)), T, axis=1))/(
                    # np.repeat(np.reshape(np.std(x[:, -T_re:], axis=1), (CH, 1)), T, axis=1))
                    # np.repeat(np.reshape(np.std(x, axis=1), (CH, 1)), T, axis=1))
                    0.000001)
            x = x_norm
            if plot is False:
                x = x[:, -T_re:-T_re+75]
            elif plot is True:
                pass
        if ch_last:
            X.append(np.reshape(x.T, (1, np.shape(x)[1], np.shape(x)[0])))
        else:
            X.append(np.reshape(x, (np.shape(x)[0], np.shape(x)[1], 1)))
    X = np.array(X)
    Y = np.array(Y)
    X_ave = []
    Y_ave = []
    for label in [4, 16, 32, 64, 128, 8]:
    # for label in [4, 8, 16, 20, 24, 32, 36, 40, 64, 68, 72, 80, 84, 88, 96, 100, 104, 128, 132, 136, 144, 148, 152, 160, 164, 168]:
        inds = list(np.where(events == label)[0])
        for _ in range(rep):
            for i in range(len(inds)):
                if sampling == 'random':
                    if epochs > 1:
                        inds_popped = inds.copy()
                        inds_popped.remove(inds[i])
                        rand_ind = random.sample(inds_popped, epochs - 1)
                        rand_ind.append(inds[i])
                        X_ave.append(np.mean(X[rand_ind], axis=0))
                    else:
                        picked_ind = inds[i]
                        X_ave.append(X[picked_ind])
                elif sampling == 'consecutive':
                    if epochs > 1:
                        if i + epochs > len(inds):
                            continue
                        else:
                            picked_ind = inds[i:i + epochs]
                            X_ave.append(np.mean(X[picked_ind], axis=0))
                    else:
                        picked_ind = inds[i]
                        X_ave.append(X[picked_ind])
                Y_ave.append(CLASS[str(label)])
                events_new.append(label)
    X_ave = np.array(X_ave)
    Y_ave = np.array(Y_ave)
    if np.shape(Y_ave)[1] > 1:
        pos_ind = np.where(Y_ave[:, 0] == 1)[0]
    else:
        pos_ind = np.where(Y_ave == 1)[0]
    scale = int((np.shape(Y_ave)[0] - len(pos_ind)) / len(pos_ind)) - 1
    if scale > 0:
        for i in range(scale):
            X_ave = np.concatenate([X_ave, X_ave[pos_ind]])
            Y_ave = np.concatenate([Y_ave, Y_ave[pos_ind]])
    # print('total: ' + str(len(Y_ave)) + '. target: ' + str(np.sum(Y_ave)))
    print('total: ' + str(np.shape(Y_ave)[0]) + '. target: ' + str(np.sum(Y_ave, axis=0)[0]))
    if plot is True:
        X_out = X
        Y_out = Y
        events_out = np.array(events)
    else:
        X_out = X_ave
        Y_out = Y_ave
        events_out = np.array(events_new)
    return X_out, Y_out, events_out

def _read_data_strat2(PATH, CLASS, ch_select=False, norm=True, plot=False, test=False, from_rep0=False,
                      num_reps_train=1, num_reps_test=1, mult=1):
    data_pkg = mne.read_epochs_eeglab(PATH)
    events = _get_event(data_pkg.events, data_pkg.event_id)
    data = data_pkg._data
    if ch_select is not False:
        CH = len(ch_select)
    else:
        CH = np.shape(data)[1]
    T = np.shape(data)[2]
    T_re = 100
    X = []
    Y = []
    events_new = []
    for i in range(np.shape(data)[0]):
        Y.append(CLASS[str(events[i])])
        x = data[i, :, :]
        if ch_select is False:
            pass
        else:
            x = x[ch_select, :]
        if norm:
            # remove baseline by [-0.2, 0.0]s
            x_norm = (x - np.repeat(np.reshape(np.mean(x[:, :25], axis=1), (CH, 1)), T, axis=1)) / (
                # np.repeat(np.reshape(np.std(x[:, -T_re:], axis=1), (CH, 1)), T, axis=1))
                # np.repeat(np.reshape(np.std(x, axis=1), (CH, 1)), T, axis=1))
                0.000001)
            x = x_norm
            if plot is False:
                # x = x[:, -T_re:-T_re + 75]
                x = x[:, -125:]
            elif plot is True:
                pass
        X.append(np.reshape(x, (1, np.shape(x)[0], np.shape(x)[1])))
    X = np.array(X)
    Y = np.array(Y)
    X_ave = []
    Y_ave = []
    for label in [4, 16, 32, 64, 128, 8]:
        inds = list(np.where(events == label)[0])
        if test is False:
            if from_rep0 is True:
                for rep in range(num_reps_train):
                    combinations = list(itertools.combinations(inds, rep + 1))
                    if rep == 0:
                        old = combinations.copy()
                        for i in range(mult - 1):
                            combinations.extend(old)
                    else:
                        random.shuffle(combinations)
                        combinations = combinations[:len(inds)*mult]
                    print(f'label: {label}. rep: {rep}. len: {len(combinations)}.')
                    for combo in combinations:
                        X_ave.append(np.mean(X[list(combo)], axis=0))
                        Y_ave.append(CLASS[str(label)])
                        events_new.append(label)
            else:
                combinations = list(itertools.combinations(inds, num_reps_train))
                random.shuffle(combinations)
                combinations = combinations[:len(inds) * mult]
                for combo in combinations:
                    X_ave.append(np.mean(X[list(combo)], axis=0))
                    Y_ave.append(CLASS[str(label)])
                    events_new.append(label)
                print(f'label: {label}. rep: {num_reps_train}. len: {len(combinations)}.')
        else:
            for i in range(len(inds) - num_reps_test):
                X_ave.append(np.mean(X[inds[i:]], axis=0))
                Y_ave.append(CLASS[str(label)])
                events_new.append(label)
    X_ave = np.array(X_ave)
    Y_ave = np.array(Y_ave)
    events_new = np.array(events_new)
    if np.shape(Y_ave)[1] > 1:
        pos_ind = np.where(Y_ave[:, 0] == 1)[0]
    else:
        pos_ind = np.where(Y_ave == 1)[0]
    scale = int((np.shape(Y_ave)[0] - len(pos_ind)) / len(pos_ind)) - 1
    if scale > 0:
        for i in range(scale):
            X_ave = np.concatenate([X_ave, X_ave[pos_ind]])
            Y_ave = np.concatenate([Y_ave, Y_ave[pos_ind]])
            events_new = np.concatenate([events_new, events_new[pos_ind]])
    print('total: ' + str(len(Y_ave)) + '. target: ' + str((scale + 1)*len(pos_ind)))
    if plot is True:
        X_out = X
        Y_out = Y
        events_out = np.array(events)
    else:
        X_out = X_ave
        Y_out = Y_ave
        events_out = events_new
    return X_out, Y_out, events_out

def _read_data_strat3(PATH, norm=True, plot=False, test=False, num_reps=1):
    data_pkg = mne.read_epochs_eeglab(PATH)
    events = _get_event(data_pkg.events, data_pkg.event_id)
    event_types = np.unique(np.array(events)).tolist()
    to_exclude = [4, 8, 32, 128, 16, 64]
    targets = [8]
    non_targets = [4, 16, 64]
    non_targets_extended = event_types
    for event in to_exclude:
        if event in non_targets_extended:
            non_targets_extended.remove(event)
    data = data_pkg._data
    CH = np.shape(data)[1]
    T = np.shape(data)[2]
    X = []
    # Y = []
    events_new = []
    for i in range(np.shape(data)[0]):
        # if events[i] in targets:
        #     Y.append(CLASS[str(events[i])])
        x = data[i, :, :]
        if norm:
            # remove baseline by [-0.2, 0.0]s
            x_norm = (x - np.repeat(np.reshape(np.mean(x[:, :25], axis=1), (CH, 1)), T, axis=1)) / (
                # np.repeat(np.reshape(np.std(x[:, -T_re:], axis=1), (CH, 1)), T, axis=1))
                # np.repeat(np.reshape(np.std(x, axis=1), (CH, 1)), T, axis=1))
                0.000001)
            x = x_norm
            if plot is False:
                # x = x[:, -T_re:-T_re + 75]
                x = x[:, -125:]
            elif plot is True:
                pass
        X.append(np.reshape(x, (1, np.shape(x)[0], np.shape(x)[1])))
    X = np.array(X)

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    X_test_ext = []
    Y_test_ext = []
    if test is False:
        for label in targets:
            # inds = list(np.where(events == label)[0])
            # combinations = list(itertools.combinations(inds, num_reps))
            # random.shuffle(combinations)
            # # combinations = combinations[:int(len(combinations) * ratio)]
            # combinations = combinations[:1000]
            combinations = _combos(events, label, num_reps)
            for combo in combinations:
                X_train.append(np.mean(X[list(combo)], axis=0))
                Y_train.append([1, 0])
                events_new.append(label)
        for label in non_targets:
            # inds = list(np.where(events == label)[0])
            # combinations = list(itertools.combinations(inds, num_reps))
            # random.shuffle(combinations)
            # # combinations = combinations[:int(len(combinations) * ratio)]
            # combinations = combinations[:1000]
            combinations = _combos(events, label, num_reps)
            for combo in combinations:
                X_train.append(np.mean(X[list(combo)], axis=0))
                Y_train.append([0, 1])
                events_new.append(label)
    elif test is True:
        for label in targets:
            inds = list(np.where(events == label)[0])
            for i in range(len(inds) - num_reps):
                X_test.append(np.mean(X[list(inds[i:i + num_reps])], axis=0))
                Y_test.append([1, 0])
            for label in non_targets:
                inds = list(np.where(events == label)[0])
                for i in range(len(inds) - num_reps):
                    X_test.append(np.mean(X[list(inds[i:i + num_reps])], axis=0))
                    Y_test.append([0, 1])
            for label in non_targets_extended:
                inds = list(np.where(events == label)[0])
                for i in range(len(inds) - num_reps):
                    X_test_ext.append(np.mean(X[list(inds[i:i + num_reps])], axis=0))
                    Y_test_ext.append([0, 1])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_test_ext = np.array(X_test_ext)
    Y_test_ext = np.array(Y_test_ext)
    out = {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_test': X_test,
        'Y_test': Y_test,
        'X_test_ext': X_test_ext,
        'Y_test_ext': Y_test_ext
    }
    return out


def _combos(events, label, num_reps):
    inds = list(np.where(events == label)[0])
    n_0 = 2000
    if num_reps < 3:
        combinations = list(itertools.combinations(inds, num_reps))
        random.shuffle(combinations)
        combinations_out = combinations[:n_0]
    elif num_reps < 5:
        n_1 = n_0//(len(inds) - 1)
        combinations_out = []
        for i in range(len(inds) - 1):
            a = inds[i:i + 2]
            inds_removed = inds.copy()
            for ele in a:
                inds_removed.remove(ele)
            combinations = list(itertools.combinations(inds_removed, num_reps - 2))
            random.shuffle(combinations)
            combinations = combinations[:n_1]
            for combo in combinations:
                combinations_out.append(np.concatenate([np.array(a), np.array(combo)]))
    else:
        n_1 = n_0 // (len(inds) - 2)
        combinations_out = []
        for i in range(len(inds) - 2):
            a = inds[i:i + 3]
            inds_removed = inds.copy()
            for ele in a:
                inds_removed.remove(ele)
            combinations = list(itertools.combinations(inds_removed, num_reps - 3))
            random.shuffle(combinations)
            combinations = combinations[:n_1]
            for combo in combinations:
                combinations_out.append(np.concatenate([np.array(a), np.array(combo)]))
    # print(len(combinations_out))
    return combinations_out


def _make_average(X, Y, CLASS, events, fold=1, epochs=1, consec=False):
    # for i in range(len(CLASS)):

    aaa = [4, 8, 16, 32, 64, 128]

    summation = np.sum(Y, axis=0)
    X_new = []
    Y_new = []
    num_sample = fold*max(summation)
    num_sample = 20*fold
    # for i in range(np.shape(Y)[1]):
    for i in range(len(CLASS)):
        class_ind = list(np.where(events == aaa[i])[0])
        # class_ind = list(np.where(Y[:, i] == 1)[0])
        for j in range(num_sample):
            if consec is False:
                ind = random.sample(class_ind, epochs)
                X_new.append(np.mean(X[ind], axis=0))
                # Y_new.append(Y[ind[0]])
                Y_new.append(CLASS[str(aaa[i])])
            elif consec is True:
                if (j + epochs) <= len(class_ind):
                    ind = class_ind[j: j + epochs]
                    X_new.append(np.mean(X[ind], axis=0))
                    Y_new.append(Y[ind[0]])
                else:
                    break
    X_new = np.array(X_new)
    Y_new = np.array(Y_new)
    return X_new, Y_new


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


# PATH = r'D:\Code\PycharmProjects\P300_detect\data\a\01_01.set'
# data_pkg = mne.read_epochs_eeglab(PATH)
# print('a')
