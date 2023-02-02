import json
import numpy as np
import os
import scipy
import mne
import matplotlib.pyplot as plt
import util_preprocessing
import util_tf

# TRAIN = [
#     'P01_01_raw.set',
#     # 'P01_02_raw.set',
#     'P01_03_raw.set',
#     'P01_04_raw.set',
#     'P01_05_raw.set',
#     'P01_06_raw.set',
# ]
# TEST = [
#     'P01_02_raw.set',
# ]

# TRAIN = [
#     'P02_01_raw.set',
#     'P02_02_raw.set',
#     # 'P02_03_raw.set',
#     'P02_04_raw.set',
#     'P02_05_raw.set',
#     'P02_06_raw.set',
# ]
# TEST = [
#     'P02_03_raw.set',
# ]

# TRAIN = [
#     'P03_01_raw.set',
#     'P03_02_raw.set',
#     # 'P03_03_raw.set',
#     'P03_04_raw.set',
#     'P03_05_raw.set',
#     'P03_06_raw.set',
# ]
# TEST = [
#     'P03_03_raw.set',
# ]

# TRAIN = [
#     'P04_01_raw.set',
#     # 'P04_02_raw.set',
#     'P04_03_raw.set',
#     'P04_04_raw.set',
#     'P04_05_raw.set',
#     'P04_06_raw.set',
# ]
# TEST = [
#     'P04_02_raw.set',
# ]

# TRAIN = [
#     'P06_01_raw.set',
#     # 'P06_02_raw.set',
#     'P06_03_raw.set',
#     'P06_04_raw.set',
#     'P06_05_raw.set',
#     'P06_06_raw.set',
# ]
# TEST = [
#     'P06_02_raw.set',
# ]

# TRAIN = [
#     'P07_01_raw.set',
#     # 'P07_02_raw.set',
#     'P07_03_raw.set',
#     'P07_04_raw.set',
#     'P07_05_raw.set',
#     'P07_06_raw.set',
# ]
# TEST = [
#     'P07_02_raw.set',
# ]

# TRAIN = [
#     'P08_01_raw.set',
#     'P08_02_raw.set',
#     'P08_03_raw.set',
#     # 'P08_04_raw.set',
#     'P08_05_raw.set',
#     'P08_06_raw.set',
# ]
# TEST = [
#     # 'P08_03_raw.set',
#     'P08_04_raw.set',
# ]

# TRAIN = [
#     'P09_01_raw.set',
#     'P09_02_raw.set',
#     'P09_03_raw.set',
#     'P09_04_raw.set',
#     'P09_05_raw.set',
#     # 'P09_06_raw.set',
# ]
# TEST = [
#     'P09_06_raw.set',
# ]

# TRAIN = [
#     '04_01.set',
#     '04_02.set',
#     '04_03.set',
#     '04_04.set',
#     '04_05.set',
#     # '04_06.set',
# ]
# TEST = [
#     '04_06.set',
# ]

TRAIN = [
    '01_01.set',
    '01_02.set',
    '01_03.set',
    '01_04.set',
    '01_05.set',
    # '01_06.set',
]
TEST = [
    '01_06.set',
]

# TRAIN = [
#     '02_01.set',
#     '02_02.set',
#     '02_03.set',
#     '02_04.set',
#     '02_05.set',
#     # '02_06.set',
# ]
# TEST = [
#     '02_06.set',
# ]

# TRAIN = [
#     '03_01.set',
#     '03_02.set',
#     '03_03.set',
#     '03_04.set',
#     '03_05.set',
#     # '03_06.set',
# ]
# TEST = [
#     '03_06.set',
# ]

# TRAIN = [
#     '06_01.set',
#     '06_02.set',
#     '06_03.set',
#     '06_04.set',
#     '06_05.set',
#     # '06_06.set',
# ]
# TEST = [
#     '06_06.set',
# ]

# TRAIN = [
#     '07_01.set',
#     '07_02.set',
#     '07_03.set',
#     '07_04.set',
#     '07_05.set',
#     # '07_06.set',
# ]
# TEST = [
#     '07_06.set',
# ]

# TRAIN = [
#     '08_01.set',
#     '08_02.set',
#     '08_03.set',
#     '08_04.set',
#     '08_05.set',
#     # '08_06.set',
# ]
# TEST = [
#     '08_06.set',
# ]

# TRAIN = [
#     # '09_01.set',
#     '09_02.set',
#     '09_03.set',
#     '09_04.set',
#     '09_05.set',
#     '09_06.set',
# ]
# TEST = [
#     '09_01.set',
# ]

# CLASS = {
#         '4': [0, 1], # nt estim
#         '8': [1, 0], # t estim
#         '16': [0, 1], # nt astim
#         '32': [0, 1], # t astim
#         '64': [0, 1], # nt vstim
#         '128': [0, 1] # t vstim
#     }

CLASS = {
        '4': [0], # nt estim
        '8': [1], # t estim
        '16': [0], # nt astim
        '32': [0], # t astim
        '64': [0], # nt vstim
        '128': [0] # t vstim
    }

# FOLDER = r'D:/Code/PycharmProjects/P300_detect/data/SEP BCI'
FOLDER = r'D:/Code/PycharmProjects/P300_detect/data/SEP BCI 125 0-20 no ICA'
# FOLDER = r'D:\Code\MATLAB\eeglab2022.1'

# X_plot, Y_plot, events_plot = util_preprocessing._build_dataset_eeglab_plot(FOLDER, TRAIN, TEST, CLASS)


X_train, Y_train, X_test, Y_test, class_weights, events_train, \
sample_weights_train, sample_weights_test = util_preprocessing._build_dataset_eeglab(FOLDER=FOLDER, CLASS=CLASS,
                                                                                          TRAIN=TRAIN, TEST=TEST,
                                                                                          ch_last=False,
                                                                                          trainset_ave=2,
                                                                                          testset_ave=2,
                                                                                          for_plot=False)

# X_train, Y_train, X_test, Y_test, events_train = util_preprocessing._build_dataset_eeglab(FOLDER=FOLDER, CLASS=CLASS,
#                                                                                           TRAIN=TRAIN, TEST=TEST,
#                                                                                           ch_last=False,
#                                                                                           trainset_ave=4,
#                                                                                           trainset_copy=True,
#                                                                                           testset_ave=3,
#                                                                                           for_plot=False)
# X_train2, Y_train2, X_test_2, Y_test_2, class_weights2, events_train2, \
# sample_weights_train2, sample_weights_test_2 = util_preprocessing._build_dataset_eeglab(FOLDER=FOLDER, CLASS=CLASS,
#                                                                                           TRAIN=TRAIN, TEST=TEST,
#                                                                                           ch_last=False,
#                                                                                           trainset_ave=4,
#                                                                                           trainset_copy=False,
#                                                                                           testset_ave=2,
#                                                                                           for_plot=False)
# X_train3, Y_train3, X_test_3, Y_test_3, class_weights3, events_train3, \
# sample_weights_train3, sample_weights_test_3 = util_preprocessing._build_dataset_eeglab(FOLDER=FOLDER, CLASS=CLASS,
#                                                                                           TRAIN=TRAIN, TEST=TEST,
#                                                                                           ch_last=False,
#                                                                                           trainset_ave=4,
#                                                                                           trainset_copy=False,
#                                                                                           testset_ave=3,
#                                                                                           for_plot=False)


print(np.shape(X_train))
print(np.shape(Y_train))
print(np.shape(X_test))
print(np.shape(Y_test))
PLOT = False
# if PLOT:
    # tgt_cnt_e = 0
    # nontgt_cnt_e = 0
    # tgt_cnt_a = 0
    # nontgt_cnt_a = 0
    # tgt_cnt_v = 0
    # nontgt_cnt_v = 0
    # x_target_e = []
    # x_nontarget_e = []
    # x_target_a = []
    # x_nontarget_a = []
    # x_target_v = []
    # x_nontarget_v = []
    # for i in range(np.shape(X_plot)[0]):
    #     if events_plot[i] == 4:
    #         x_nontarget_e.append(np.squeeze(X_plot[i, :, :, 0]))
    #         nontgt_cnt_e = nontgt_cnt_e + 1
    #     elif events_plot[i] == 8:
    #         x_target_e.append(np.squeeze(X_plot[i, :, :, 0]))
    #         tgt_cnt_e = tgt_cnt_e + 1
    #     elif events_plot[i] == 16:
    #         x_nontarget_a.append(np.squeeze(X_plot[i, :, :, 0]))
    #         nontgt_cnt_a = nontgt_cnt_a + 1
    #     elif events_plot[i] == 32:
    #         x_target_a.append(np.squeeze(X_plot[i, :, :, 0]))
    #         tgt_cnt_a = tgt_cnt_a + 1
    #     elif events_plot[i] == 64:
    #         x_nontarget_v.append(np.squeeze(X_plot[i, :, :, 0]))
    #         nontgt_cnt_v = nontgt_cnt_v + 1
    #     elif events_plot[i] == 128:
    #         x_target_v.append(np.squeeze(X_plot[i, :, :, 0]))
    #         tgt_cnt_v = tgt_cnt_v + 1
    # print('target_e: ' + str(tgt_cnt_e))
    # print('nontarget_e: ' + str(nontgt_cnt_e))
    # print('target_a: ' + str(tgt_cnt_a))
    # print('nontarget_a: ' + str(nontgt_cnt_a))
    # print('target_v: ' + str(tgt_cnt_v))
    # print('nontarget_v: ' + str(nontgt_cnt_v))
    # x_target_e = np.array(x_target_e)
    # x_nontarget_e = np.array(x_nontarget_e)
    # x_target_e_mean = np.mean(x_target_e, axis=0)
    # x_nontarget_e_mean = np.mean(x_nontarget_e, axis=0)
    # x_target_a = np.array(x_target_a)
    # x_nontarget_a = np.array(x_nontarget_a)
    # x_target_a_mean = np.mean(x_target_a, axis=0)
    # x_nontarget_a_mean = np.mean(x_nontarget_a, axis=0)
    # x_target_v = np.array(x_target_v)
    # x_nontarget_v = np.array(x_nontarget_v)
    # x_target_v_mean = np.mean(x_target_v, axis=0)
    # x_nontarget_v_mean = np.mean(x_nontarget_v, axis=0)
    # # time_axis = np.linspace(-0.2, 0.8, 1*250, endpoint=False)
    # time_axis = np.linspace(0.2, 0.8, 75, endpoint=False)
    #
    # plot_ch = 28
    # plt.plot(time_axis, x_target_e_mean[plot_ch, :])
    # plt.plot(time_axis, x_nontarget_e_mean[plot_ch, :])
    # plt.plot(time_axis, x_target_a_mean[plot_ch, :])
    # plt.plot(time_axis, x_nontarget_a_mean[plot_ch, :])
    # plt.plot(time_axis, x_target_v_mean[plot_ch, :])
    # plt.plot(time_axis, x_nontarget_v_mean[plot_ch, :])
    # plt.grid()
    # plt.title('Grand Ave. all data - Subject 4')
    # plt.legend(['target_electric', 'nontarget_electric', 'target_audio', 'nontarget_audio', 'target_vibration', 'nontarget_vibration'])
    # plt.show()

model = util_tf._eegnet_1(in_shape=np.shape(X_train)[-3:], out_shape=np.shape(Y_train)[-1],
                        )
# model = util_tf._effnetV2(in_shape=np.shape(X_train)[-3:], out_shape=np.shape(Y_train)[-1],
#                         loss_weights=None)
print(class_weights)
# class_weights = {
#     0: class_weights[0],
#     1: class_weights[1]
# }
max_iter = 200
acc_old = 0
record_iter = []
record_loss_train = []
record_loss_test = []
record_acc_train = []
record_acc_test = []
record_P_val = []
record_N_val = []

# record_loss_train_2 = []
# record_loss_test_2 = []
# record_loss_train_3 = []
# record_loss_test_3 = []
converge_threshold = 0.0001
best_loss = 1
best_acc = 0

for i in range(max_iter):
    record_iter.append(i)
    model.save_weights('Temp_saved_weights')
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
        # best_acc = [results['matrix'][0, 0], results['matrix'][1, 1]]
    if acc_test > best_acc:
        model.save('results_noICA_epoch_3/' + TEST[0] + '_iter_' + str(i))
        best_acc = acc_test
        print('model saved ' + TEST[0])
    # pred_2 = model.evaluate(x=X_test_2, y=Y_test_2, sample_weight=sample_weights_test_2)
    # loss_test_2 = pred_2[0]
    # record_loss_test_2.append(loss_test_2)
    # Y_pred_2 = model.predict(x=X_test_2)
    # results_2 = util_tf._confusion_matrix(Y_pred_2, Y_test_2)
    #
    # pred_3 = model.evaluate(x=X_test_3, y=Y_test_3, sample_weight=sample_weights_test_3)
    # loss_test_3 = pred_3[0]
    # record_loss_test_3.append(loss_test_3)
    # Y_pred_3 = model.predict(x=X_test_3)
    # results_3 = util_tf._confusion_matrix(Y_pred_3, Y_test_3)

    # acc = 0.5*(results['matrix'][0, 0] + results['matrix'][1, 1])
    # print('Iter: ' + str(i) + '. Test acc: ' + str(acc_test) + '. Test loss: ' + str(loss_test))
    print('Iter: ' + str(i) + '. TP: ' + str(results['matrix'][0, 0]) + '. TN: ' + str(results['matrix'][1, 1])
          + '. Test loss: ' + str(loss_test) + '. Test acc: ' + str(acc_test))
    # print('Iter: ' + str(i) + '. TP: ' + str(results_2['matrix'][0, 0]) + '. TN: ' + str(results_2['matrix'][1, 1])
    #       + '. Test loss: ' + str(loss_test_2))
    # print('Iter: ' + str(i) + '. TP: ' + str(results_3['matrix'][0, 0]) + '. TN: ' + str(results_3['matrix'][1, 1])
    #       + '. Test loss: ' + str(loss_test_3))
    if i > 25:
        if np.max(np.array(record_loss_train[-10:])) < converge_threshold:
            break
    # if acc > acc_old:
    #     print('weights updated')
    #     acc_old = acc
    #     pass
    # else:
    #     model.load_weights('Temp_saved_weights')
    #     print('weights discarded')
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
with open('results_noICA_epoch_3' + TEST[0] + '.json', "w") as json_file:
    json.dump(dict_save, json_file)

print('best: ')
print(best_loss)
print(best_acc)
plt.plot(np.array(record_iter), np.array(record_loss_train))
plt.plot(np.array(record_iter), np.array(record_loss_test))
# plt.plot(np.array(record_iter), np.array(record_loss_test_2))
# plt.plot(np.array(record_iter), np.array(record_loss_test_3))
plt.axhline(y=converge_threshold, linestyle='--')
plt.grid()
plt.xlabel(['Iteration'])
plt.xlim([0, max_iter])
# plt.ylim([0, 0.12])
plt.ylabel(['Weighted loss'])
plt.legend(['train loss', 'test loss 1', 'test loss 2', 'test loss 3', 'convergence threshold'])
plt.show()
# model.fit(x=X_train, y=Y_train, epochs=100, batch_size=16,
#               class_weight=class_weights)
# print('Training Complete')
# Y_pred = model.predict(x=X_test)
# results = util_tf._confusion_matrix(Y_pred, Y_test)
# print(results['matrix'])
