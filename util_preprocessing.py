import numpy as np
import scipy

def _build_dataset_p300(PATH, WIN, CH, epochs=1):
    TARGET = np.array([1, 0])
    NONTARGET = np.array([0, 1])
    data_pkg = {}
    data = scipy.io.loadmat(PATH)
    Signal = data['Signal']
    Flashing = data['Flashing']
    StimType = data['StimulusType']
    X = []
    Y = []
    grand_ave_target = []
    grand_ave_nontarget = []
    target_ind = []
    for i in range(np.shape(Flashing)[0]):
        X_sample = []
        Y_sample = []
        for j in range(np.shape(Flashing)[1] - 1):
            if Flashing[i, j + 1] == 1 and Flashing[i, j] == 0:
                x = np.reshape(Signal[i, j + 1:j + WIN + 1, :], (WIN, CH)).T
                x = (x - np.repeat(np.reshape(np.mean(x, axis=1), (CH, 1)), np.shape(x)[1], axis=1)) / (
                        np.repeat(np.reshape(np.std(x, axis=1), (CH, 1)), np.shape(x)[1], axis=1))
                X_sample.append(x)
                target_ind.append(i)
                if StimType[i, j + 1] == 1:
                    grand_ave_target.append(X_sample[-1])
                    Y_sample.append(TARGET)
                else:
                    grand_ave_nontarget.append(X_sample[-1])
                    Y_sample.append(NONTARGET)
        X.append(np.array(X_sample))
        Y.append(np.array(Y_sample))
    X = np.array(X)
    Y = np.array(Y)
    target_ind = np.array(target_ind)
    if epochs > 1 and epochs <= 15:
        X_multi_epochs = []
        Y_multi_epochs = []
        for i in range(np.shape(X)[0]):
            x_target = []
            x_target_cnt = 0
            x_nontarget = []
            x_nontarget_cnt = 0
            for j in range(np.shape(X)[1]):
                if Y[i, j, 0] == 1:
                    # target
                    x_target_cnt = x_target_cnt + 1
                    x_target.append(X[i, j, :, :])
                else:
                    # nontarget
                    x_nontarget_cnt = x_nontarget_cnt + 1
                    x_nontarget.append(X[i, j, :, :])
                if x_target_cnt >= epochs*2 and x_nontarget_cnt >= epochs*5:
                    X_multi_epochs.append(np.mean(np.array(x_target), axis=0))
                    Y_multi_epochs.append(TARGET)
                    X_multi_epochs.append(np.mean(np.array(x_nontarget), axis=0))
                    Y_multi_epochs.append(NONTARGET)
                    break
        X = np.array(X_multi_epochs)
        Y = np.array(Y_multi_epochs)
    data_pkg['X'] = X
    data_pkg['Y'] = Y
    data_pkg['target_ind'] = target_ind
    data_pkg['grand_ave_target'] = np.mean(np.array(grand_ave_target), axis=0)
    data_pkg['grand_ave_nontarget'] = np.mean(np.array(grand_ave_nontarget), axis=0)

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