import numpy as np
import scipy


P300_SPELLER = {
    'A'
}

def _build_dataset_p300(PATH, WIN, CH, epochs=1):
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
                X_sample.append(x)
                X_sample_norm.append(x_norm)
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
                    x_target.append(X[i, j, :, :])
                    x_target_norm.append(X_norm[i, j, :, :])
                else:
                    # nontarget
                    x_nontarget_cnt = x_nontarget_cnt + 1
                    x_nontarget.append(X[i, j, :, :])
                    x_nontarget_norm.append(X_norm[i, j, :, :])
                if x_target_cnt >= epochs*2 and x_nontarget_cnt >= epochs*5:
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
    data_pkg['X'] = np.reshape(X, (np.shape(X)[0], np.shape(X)[1], np.shape(X)[2], np.shape(X)[3], 1))
    data_pkg['X_norm'] = np.reshape(X_norm, (np.shape(X_norm)[0], np.shape(X_norm)[1], np.shape(X_norm)[2], np.shape(X_norm)[3], 1))
    data_pkg['Y'] = Y
    data_pkg['target_ind'] = target_ind
    data_pkg['grand_ave_target'] = np.mean(np.array(grand_ave_target), axis=0)
    data_pkg['grand_ave_nontarget'] = np.mean(np.array(grand_ave_nontarget), axis=0)

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


def _build_dataset_p300_test(PATH, WIN, CH, LABEL, epochs=1):
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
                X_sample.append(x)
                X_sample_norm.append(x_norm)
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
                    x_target.append(X[i, j, :, :])
                    x_target_norm.append(X_norm[i, j, :, :])
                else:
                    # nontarget
                    x_nontarget_cnt = x_nontarget_cnt + 1
                    x_nontarget.append(X[i, j, :, :])
                    x_nontarget_norm.append(X_norm[i, j, :, :])
                if x_target_cnt >= epochs*2 and x_nontarget_cnt >= epochs*5:
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
    data_pkg['X'] = np.reshape(X, (np.shape(X)[0], np.shape(X)[1], np.shape(X)[2], np.shape(X)[3], 1))
    data_pkg['X_norm'] = np.reshape(X_norm, (np.shape(X_norm)[0], np.shape(X_norm)[1], np.shape(X_norm)[2], np.shape(X_norm)[3], 1))
    data_pkg['Y'] = Y
    data_pkg['target_ind'] = target_ind

    return data_pkg
#
# def _multi_epochs(data_pkg, epochs=1):
#     X = data_pkg['X']
#     Y = data_pkg['Y']
#     target_ind = data_pkg['target_ind']
#     prev_ind = -1
#     for i in range(np.shape(X)[0]):
#         if target_ind[i] > prev_ind:
#             # start