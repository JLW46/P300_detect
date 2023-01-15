import numpy as np
import os
import scipy
import mne
import matplotlib.pyplot as plt
import util_preprocessing
import util_tf

TRAIN = [
    'P01_01.set',
    'P01_02.set',
    # 'P01_03.set',
    'P01_04.set',
    'P01_05.set',
    'P01_06.set',
]
TEST = [
    'P01_03.set',
]

CLASS = {
        '4': [0, 1], # nt estim
        '8': [1, 0], # t estim
        '16': [0, 1], # nt astim
        '32': [0, 1], # t astim
        '64': [0, 1], # nt vstim
        '128': [0, 1] # t vstim
    }

FOLDER = r'D:/Code/PycharmProjects/P300_detect/data/SEP BCI'

X_train, Y_train, X_test, Y_test, loss_weights = util_preprocessing._build_dataset_eeglab(FOLDER=FOLDER, CLASS=CLASS,
                                                                                          TRAIN=TRAIN, TEST=TEST,
                                                                                          ch_last=True,
                                                                                          trainset_ave=True,
                                                                                          testset_ave=True,
                                                                                          for_plot=False)
print(np.shape(X_train))
print(np.shape(Y_train))
print(np.shape(X_test))
print(np.shape(Y_test))

# tgt_cnt = 0
# nontgt_cnt = 0
# x_target = []
# x_nontarget = []
# for i in range(np.shape(X_train)[0]):
#     if Y_train[i, 0] == 1:
#         x_target.append(np.squeeze(X_train[i, :, :, 0]))
#         tgt_cnt = tgt_cnt + 1
#     elif Y_train[i, 1] == 1:
#         x_nontarget.append(np.squeeze(X_train[i, :, :, 0]))
#         nontgt_cnt = nontgt_cnt + 1
# print('target: ' + str(tgt_cnt))
# print('nontarget: ' + str(nontgt_cnt))
# x_target = np.array(x_target)
# x_nontarget = np.array(x_nontarget)
# x_target_mean = np.mean(x_target, axis=0)
# x_nontarget_mean = np.mean(x_nontarget, axis=0)
# time_axis = np.linspace(-0.2, 0.8, 1*250, endpoint=False)
# plot_ch = 10
# plt.plot(time_axis, x_target_mean[plot_ch, :])
# plt.plot(time_axis, x_nontarget_mean[plot_ch, :])
# plt.grid()
# plt.legend(['target', 'nontarget'])
# plt.show()

# model = util_tf._eegnet(in_shape=np.shape(X_train)[-3:], out_shape=np.shape(Y_train)[-1],
#                         loss_weights=loss_weights, dropout_rate=0.4)
model = util_tf._effnetV2(in_shape=np.shape(X_train)[-3:], out_shape=np.shape(Y_train)[-1],
                        loss_weights=None)
print(loss_weights)
class_weights = {
    0: loss_weights[0],
    1: loss_weights[1]
}
model.fit(x=X_train, y=Y_train, epochs=50, batch_size=16,
          class_weight=None)
print('Training Complete')
Y_pred = model.predict(x=X_test)
results = util_tf._confusion_matrix(Y_pred, Y_test)
print(results['matrix'])
