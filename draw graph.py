import numpy as np
import os
import json
import matplotlib.pyplot as plt
import util_preprocessing
import csv
import sklearn
from sklearn.metrics import confusion_matrix
import matplotlib.font_manager as font_manager
import mne


FOLDER_1 = r'D:/Code/PycharmProjects/P300_detect/results_noICA_epoch_1'
FOLDER_2 = r'D:/Code/PycharmProjects/P300_detect/results_noICA_epoch_2'
FOLDER_3 = r'D:/Code/PycharmProjects/P300_detect/results_noICA_epoch_3'
# FOLDER_1 = r'D:/Code/PycharmProjects/P300_detect/results_ICA_epoch_1'
# FOLDER_2 = r'D:/Code/PycharmProjects/P300_detect/results_ICA_epoch_2'
# FOLDER_3 = r'D:/Code/PycharmProjects/P300_detect/results_ICA_epoch_3'
# FOLDER_DATA = r'D:\Code\PycharmProjects\P300_detect\data\SEP BCI 125 0-20 no ICA'
# FOLDER_DATA = r'D:\Code\PycharmProjects\P300_detect\data\SEP BCI 125 0-20 ICA'
FOLDER_DATA = r'D:\Code\PycharmProjects\P300_detect\data\SEP BCI 125 0-20 no ICA 2'

# CLASS = {
#         '4': [0], # nt estim
#         '8': [1], # t estim
#         '16': [0], # nt astim
#         '32': [0], # t astim
#         '64': [0], # nt vstim
#         '128': [0] # t vstim
#     }

CLASS = {
        '4': [0], # nt estim
        '8': [1], # t estim
        '16': [0], # nt astim
        '32': [1], # t astim
        '64': [0], # nt vstim
        '128': [1], # t vstim
        '20': [0],
        '24': [0],
        '36': [0],
        '40': [0],
        '68': [0],
        '72': [0],
        '80': [0],
        '84': [0],
        '88': [0],
        '96': [0],
        '100': [0],
        '104': [0],
        '132': [0],
        '136': [0],
        '144': [0],
        '148': [0],
        '152': [0],
        '160': [0],
        '164': [0],
        '168': [0],
    }
def _read_auc(FOLDER):
    files = os.listdir(FOLDER)
    sbj_old = ''
    result_acc = {}
    for file_name in files:
        if file_name.endswith('.json'):
            with open(os.path.join(FOLDER, file_name)) as json_file:
                data = json.load(json_file)
            sbj = file_name.split('_')[0]
            trial = file_name.split('_')[1].split('.')[0]
            if sbj != sbj_old:
                sbj_old = sbj
                result_acc[sbj] = [data['auc']]
            else:
                result_acc[sbj].append(data['auc'])
    return result_acc


def _read_acc(FOLDER):
    SBJ = ['01', '02', '03', '04', '06', '07', '08', '09']
    SESSION = ['01', '02', '03', '04', '05', '06']
    sbj_result = []
    for sbj in SBJ:
        Y_PRED = {}
        Y_TRUE = {}
        AUC = {}
        mean_ses_acc = 0
        mean_ses_precision = 0
        mean_ses_recall = 0
        mean_ses_f1 = 0
        mean_ses_auc = 0
        for ses in SESSION:
            print(sbj + '_' + ses)
            file_name = sbj + '_' + ses + '.json'
            with open(os.path.join(FOLDER, file_name)) as json_file:
                data = json.load(json_file)
            Y_pred = data['Y_pred']
            Y_true = data['Y_true']
            if len(np.shape(Y_pred)) > 1:
                Y_pred = Y_pred[:, 1]
                Y_true = Y_true[:, 1]
            Y_pred = np.array(Y_pred)
            Y_true = np.array(Y_true)
            ##### balance #####
            target_ind = np.where(Y_true == 1)[0]
            ratio = int((len(Y_true) - len(target_ind)) / len(target_ind))
            for i in range(ratio):
                Y_pred = np.concatenate([Y_pred, Y_pred[target_ind]])
                Y_true = np.concatenate([Y_true, Y_true[target_ind]])
            # print('total: ' + str(len(Y_true)) + '. targets: ' + str(np.sum(Y_true)))
            ### balance end ###
            Y_PRED[ses] = Y_pred
            Y_TRUE[ses] = Y_true
            AUC[ses] = data['auc']
            ####### find best threshold #######
            best_ses_acc = 0
            for threshold in np.linspace(0, 1, num=100, endpoint=False):
                pred = np.zeros(np.shape(Y_PRED[ses])[0])
                pred[np.where(Y_PRED[ses] > threshold)[0]] = 1
                TN, FP, FN, TP = confusion_matrix(y_true=Y_TRUE[ses], y_pred=pred).ravel()
                ses_acc = (TP + TN) / (TP + TN + FP + FN)
                if ses_acc > best_ses_acc and TP != 0:
                    best_ses_acc = ses_acc
                    best_ses_precision = TP / (TP + FP)
                    best_ses_recall = TP / (TP + FN)
                    best_ses_f1 = 2 * (best_ses_precision * best_ses_recall) / (best_ses_precision + best_ses_recall)
            ###### find best threshold end ######
            mean_ses_acc = mean_ses_acc + best_ses_acc
            mean_ses_precision = mean_ses_precision + best_ses_precision
            mean_ses_recall = mean_ses_recall + best_ses_recall
            mean_ses_f1 = mean_ses_f1 + best_ses_f1
            mean_ses_auc = mean_ses_auc + data['auc']
        mean_ses_acc = mean_ses_acc/len(SESSION)
        mean_ses_precision = mean_ses_precision/len(SESSION)
        mean_ses_recall = mean_ses_recall/len(SESSION)
        mean_ses_f1 = mean_ses_f1/len(SESSION)
        mean_ses_auc = mean_ses_auc/len(SESSION)
        sbj_result.append([mean_ses_acc, mean_ses_precision, mean_ses_recall, mean_ses_f1, mean_ses_auc])

    return sbj_result


def _read_prediction(FOLDER, SBJ_PLOT):
    files = os.listdir(FOLDER)
    sbj_old = ''
    result_pred = {}
    for file_name in files:
        sbj = file_name.split('_')[0]
        if file_name.endswith('.json') and sbj in SBJ_PLOT:
            with open(os.path.join(FOLDER, file_name)) as json_file:
                data = json.load(json_file)
            trial = file_name.split('_')[1].split('.')[0]
            best_ind = np.where(np.array(data['test_acc']) == data['best_test_acc'])[0]
            if len(best_ind) > 1:
                best_ind = best_ind[-1]
            best_ind = int(best_ind)
            print(best_ind)
            if sbj != sbj_old:
                result_pred[str(sbj + 'P')] = data['test_P_val'][best_ind]
                result_pred[sbj + 'N'] = data['test_N_val'][best_ind]
            else:
                result_pred[sbj + 'P'].append(data['test_P_val'][best_ind])
                result_pred[sbj + 'N'].append(data['test_N_val'][best_ind])
    return result_pred


def _plot_prediction(SBJ_PLOT):
    data_to_box = []
    label_to_box = []
    result_pred_1 = _read_prediction(FOLDER_1, SBJ_PLOT=SBJ_PLOT)
    result_pred_2 = _read_prediction(FOLDER_2, SBJ_PLOT=SBJ_PLOT)
    result_pred_3 = _read_prediction(FOLDER_3, SBJ_PLOT=SBJ_PLOT)
    fig = plt.figure()
    for sbj_plot in SBJ_PLOT:
        data_to_box.append(result_pred_1[sbj_plot + 'N'])
        label_to_box.append(str('1-N'))
        data_to_box.append(result_pred_1[sbj_plot + 'P'])
        label_to_box.append(str('1-T'))
        data_to_box.append(result_pred_2[sbj_plot + 'N'])
        label_to_box.append(str('2-N'))
        data_to_box.append(result_pred_2[sbj_plot + 'P'])
        label_to_box.append(str('2-T'))
        data_to_box.append(result_pred_3[sbj_plot + 'N'])
        label_to_box.append(str('3-N'))
        data_to_box.append(result_pred_3[sbj_plot + 'P'])
        label_to_box.append(str('3-T'))
    x_spacing = [1, 2, 4, 5, 7, 8]
    ax1 = fig.add_subplot(121)
    ax1.set_title(str('Subject ' + SBJ_PLOT[0]))
    ax1.set_ylabel('Prediction')
    ax1.set_xlabel('Repetition')
    box_dict_1 = ax1.boxplot(data_to_box[:6], labels=label_to_box[:6],
                             sym='+', positions=x_spacing, patch_artist=True, showmeans=True)
    ax1.yaxis.grid(True)
    ax2 = fig.add_subplot(122)
    ax2.set_title(str('Subject ' + SBJ_PLOT[1]))
    box_dict_2 = ax2.boxplot(data_to_box[6:12], labels=label_to_box[6:12],
                             sym='+', positions=x_spacing, patch_artist=True, showmeans=True)
    ax2.yaxis.grid(True)
    ax2.set_xlabel('Repetition')
    # ax3 = fig.add_subplot(133)
    # ax3.set_title(str('Subject ' + SBJ_PLOT[2]))
    # box_dict_3 = ax3.boxplot(data_to_box[12:], labels=label_to_box[12:],
    #                          sym='+', positions=x_spacing, patch_artist=True, showmeans=True)
    # ax3.yaxis.grid(True)
    COLOR = ['gold', 'mediumpurple', 'gold', 'mediumpurple', 'gold', 'mediumpurple']
    for i in range(6):
        box_dict_1.get('boxes')[i].set_facecolor(COLOR[i])
        box_dict_2.get('boxes')[i].set_facecolor(COLOR[i])
        # box_dict_3.get('boxes')[i].set_facecolor(COLOR[i])
    # plt.legend(['Non-target', 'Target'])
    plt.show()

    return


def _epoch_plot(SBJ_PLOT):
    result_acc_1 = _read_auc(FOLDER_1)
    result_acc_2 = _read_auc(FOLDER_2)
    result_acc_3 = _read_auc(FOLDER_3)
    result_mean = {}
    for sbj in SBJ_PLOT:
        result_mean[sbj] = [np.mean(np.array(result_acc_1[sbj])),
                            np.mean(np.array(result_acc_2[sbj])),
                            np.mean(np.array(result_acc_3[sbj]))]
        stds = [np.mean(np.std(result_acc_1[sbj])),
                            np.std(np.array(result_acc_2[sbj])),
                            np.std(np.array(result_acc_3[sbj]))]
        print(result_mean[sbj])
        print([stds])
        print('  ')
        plt.plot(['1', '2', '3'], result_mean[sbj], 'o-')
    for k in range(len(SBJ_PLOT)):
        SBJ_PLOT[k] = str('Subject ' + SBJ_PLOT[k])
    # plt.legend(SBJ_PLOT)
    plt.legend(['SBJ01', 'SBJ02', 'SBJ03', 'SBJ04', 'SBJ05', 'SBJ06', 'SBJ07', 'SBJ08'])
    plt.ylim([0.8, 1.0])
    plt.xlabel('Number of Samples Averaged')
    plt.ylabel('AUC')
    plt.grid(axis='y')
    plt.show()
    return


def _epoch_plot_methods(SBJ_PLOT):
    fig = plt.figure(1)
    result_mean_all_subs = {}
    result_mean_all_subs8 = {}
    y_lim = [0.5, 1.0]
    COLOR = {
        '01': 'darkcyan',
        '02': 'steelblue',
        '03': 'rebeccapurple',
        '04': 'deeppink',
        '06': 'brown',
        '07': 'mediumseagreen',
        '08': 'olive',
        '09': 'gray',
    }

    ax0 = fig.add_subplot(151)
    ax0.set_title('CSP-LDA')
    ax0.set_ylabel('AUC')
    ax0.set_xlabel('Repetition')
    ax0.set_ylim(y_lim)
    ax0.yaxis.grid(True)
    result_acc_1 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_epoch_1')
    result_acc_2 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_epoch_2')
    result_acc_3 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_epoch_3')
    result_acc_4 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_epoch_4')
    result_acc_5 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_epoch_5')
    result_acc_6 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_epoch_6')
    result_acc8_1 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_8ch_epoch_1')
    result_acc8_2 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_8ch_epoch_2')
    result_acc8_3 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_8ch_epoch_3')
    result_acc8_4 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_8ch_epoch_4')
    result_acc8_5 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_8ch_epoch_5')
    result_acc8_6 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_8ch_epoch_6')
    result_mean = {}
    result_mean8 = {}
    result_mean_all_subs['csplda'] = np.zeros(6)
    result_mean_all_subs8['csplda'] = np.zeros(6)
    for sbj in SBJ_PLOT:
        result_mean[sbj] = [np.mean(np.array(result_acc_1[sbj])),
                            np.mean(np.array(result_acc_2[sbj])),
                            np.mean(np.array(result_acc_3[sbj])),
                            np.mean(np.array(result_acc_4[sbj])),
                            np.mean(np.array(result_acc_5[sbj])),
                            np.mean(np.array(result_acc_6[sbj])),
                            ]
        result_mean_all_subs['csplda'] = result_mean_all_subs['csplda'] + np.squeeze(np.array(result_mean[sbj]))
        result_mean8[sbj] = [np.mean(np.array(result_acc8_1[sbj])),
                            np.mean(np.array(result_acc8_2[sbj])),
                            np.mean(np.array(result_acc8_3[sbj])),
                            np.mean(np.array(result_acc8_4[sbj])),
                            np.mean(np.array(result_acc8_5[sbj])),
                            np.mean(np.array(result_acc8_6[sbj])),
                            ]
        result_mean_all_subs8['csplda'] = result_mean_all_subs8['csplda'] + np.squeeze(np.array(result_mean8[sbj]))
        stds = [np.mean(np.std(result_acc_1[sbj])),
                np.std(np.array(result_acc_2[sbj])),
                np.std(np.array(result_acc_3[sbj]))]
        print(result_mean[sbj])
        print([stds])
        print('  ')
        ax0.plot(['1', '2', '3', '4', '5', '6'], result_mean[sbj], 'o-', color=COLOR[sbj])
        ax0.plot(['1', '2', '3', '4', '5', '6'], result_mean8[sbj], 's--', color=COLOR[sbj])
    result_mean_all_subs['csplda'] = result_mean_all_subs['csplda'] / 8
    result_mean_all_subs8['csplda'] = result_mean_all_subs8['csplda'] / 8
    ax0.legend(['SBJ01', 'SBJ01_8', 'SBJ02', 'SBJ02_8',
                'SBJ03', 'SBJ03_8', 'SBJ04', 'SBJ04_8',
                'SBJ05', 'SBJ05_8', 'SBJ06', 'SBJ06_8',
                'SBJ07', 'SBJ07_8', 'SBJ08', 'SBJ08_8'])
    ax1 = fig.add_subplot(152)
    ax1.set_title('EEGNET')
    ax1.set_ylabel('AUC')
    ax1.set_xlabel('Repetition')
    ax1.set_ylim(y_lim)
    ax1.yaxis.grid(True)
    result_acc_1 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet_epoch_1')
    result_acc_2 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet_epoch_2')
    result_acc_3 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet_epoch_3')
    result_acc_4 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet_epoch_4')
    result_acc_5 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet_epoch_5')
    result_acc_6 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet_epoch_6')
    result_acc8_1 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet_8ch_epoch_1')
    result_acc8_2 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet_8ch_epoch_2')
    result_acc8_3 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet_8ch_epoch_3')
    result_acc8_4 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet_8ch_epoch_4')
    result_acc8_5 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet_8ch_epoch_5')
    result_acc8_6 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet_8ch_epoch_6')
    result_mean = {}
    result_mean_all_subs['eegnet'] = np.zeros(6)
    result_mean8 = {}
    result_mean_all_subs8['eegnet'] = np.zeros(6)
    for sbj in SBJ_PLOT:
        result_mean[sbj] = [np.mean(np.array(result_acc_1[sbj])),
                            np.mean(np.array(result_acc_2[sbj])),
                            np.mean(np.array(result_acc_3[sbj])),
                            np.mean(np.array(result_acc_4[sbj])),
                            np.mean(np.array(result_acc_5[sbj])),
                            np.mean(np.array(result_acc_6[sbj]))
                            ]
        result_mean_all_subs['eegnet'] = result_mean_all_subs['eegnet'] + np.squeeze(np.array(result_mean[sbj]))
        result_mean8[sbj] = [np.mean(np.array(result_acc8_1[sbj])),
                             np.mean(np.array(result_acc8_2[sbj])),
                             np.mean(np.array(result_acc8_3[sbj])),
                             np.mean(np.array(result_acc8_4[sbj])),
                             np.mean(np.array(result_acc8_5[sbj])),
                             np.mean(np.array(result_acc8_6[sbj])),
                             ]
        result_mean_all_subs8['eegnet'] = result_mean_all_subs8['eegnet'] + np.squeeze(np.array(result_mean8[sbj]))
        stds = [np.mean(np.std(result_acc_1[sbj])),
                np.std(np.array(result_acc_2[sbj])),
                np.std(np.array(result_acc_3[sbj]))]
        print(result_mean[sbj])
        print([stds])
        print('  ')
        ax1.plot(['1', '2', '3', '4', '5', '6'], result_mean[sbj], 'o-', color=COLOR[sbj])
        ax1.plot(['1', '2', '3', '4', '5', '6'], result_mean8[sbj], 's--', color=COLOR[sbj])
    result_mean_all_subs['eegnet'] = result_mean_all_subs['eegnet'] / 8
    result_mean_all_subs8['eegnet'] = result_mean_all_subs8['eegnet'] / 8
    ax1.legend(['SBJ01', 'SBJ01_8', 'SBJ02', 'SBJ02_8',
                'SBJ03', 'SBJ03_8', 'SBJ04', 'SBJ04_8',
                'SBJ05', 'SBJ05_8', 'SBJ06', 'SBJ06_8',
                'SBJ07', 'SBJ07_8', 'SBJ08', 'SBJ08_8'])
    ax2 = fig.add_subplot(153)
    ax2.set_title('EFFNET_V2')
    ax2.set_ylabel('AUC')
    ax2.set_xlabel('Repetition')
    ax2.set_ylim(y_lim)
    ax2.yaxis.grid(True)
    result_acc_1 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_effnetv2_epoch_1')
    result_acc_2 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_effnetv2_epoch_2')
    result_acc_3 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_effnetv2_epoch_3')
    result_acc_4 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_effnetv2_epoch_4')
    result_acc_5 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_effnetv2_epoch_5')
    result_acc_6 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_effnetv2_epoch_6')
    result_acc8_1 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_effnetv2_8ch_epoch_1')
    result_acc8_2 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_effnetv2_8ch_epoch_2')
    result_acc8_3 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_effnetv2_8ch_epoch_3')
    result_acc8_4 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_effnetv2_8ch_epoch_4')
    result_acc8_5 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_effnetv2_8ch_epoch_5')
    result_acc8_6 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_effnetv2_8ch_epoch_6')
    result_mean = {}
    result_mean_all_subs['effnetv2'] = np.zeros(6)
    result_mean8 = {}
    result_mean_all_subs8['effnetv2'] = np.zeros(6)
    for sbj in SBJ_PLOT:
        result_mean[sbj] = [np.mean(np.array(result_acc_1[sbj])),
                            np.mean(np.array(result_acc_2[sbj])),
                            np.mean(np.array(result_acc_3[sbj])),
                            np.mean(np.array(result_acc_4[sbj])),
                            np.mean(np.array(result_acc_5[sbj])),
                            np.mean(np.array(result_acc_6[sbj]))
                            ]
        result_mean_all_subs['effnetv2'] = result_mean_all_subs['effnetv2'] + np.squeeze(np.array(result_mean[sbj]))
        result_mean8[sbj] = [np.mean(np.array(result_acc8_1[sbj])),
                             np.mean(np.array(result_acc8_2[sbj])),
                             np.mean(np.array(result_acc8_3[sbj])),
                             np.mean(np.array(result_acc8_4[sbj])),
                             np.mean(np.array(result_acc8_5[sbj])),
                             np.mean(np.array(result_acc8_6[sbj])),
                             ]
        result_mean_all_subs8['effnetv2'] = result_mean_all_subs8['effnetv2'] + np.squeeze(np.array(result_mean8[sbj]))
        stds = [np.mean(np.std(result_acc_1[sbj])),
                np.std(np.array(result_acc_2[sbj])),
                np.std(np.array(result_acc_3[sbj]))]
        print(result_mean[sbj])
        print([stds])
        print('  ')
        ax2.plot(['1', '2', '3', '4', '5', '6'], result_mean[sbj], 'o-', color=COLOR[sbj])
        ax2.plot(['1', '2', '3', '4', '5', '6'], result_mean8[sbj], 's--', color=COLOR[sbj])
    result_mean_all_subs['effnetv2'] = result_mean_all_subs['effnetv2'] / 8
    result_mean_all_subs8['effnetv2'] = result_mean_all_subs8['effnetv2'] / 8
    ax2.legend(['SBJ01', 'SBJ01_8', 'SBJ02', 'SBJ02_8',
                'SBJ03', 'SBJ03_8', 'SBJ04', 'SBJ04_8',
                'SBJ05', 'SBJ05_8', 'SBJ06', 'SBJ06_8',
                'SBJ07', 'SBJ07_8', 'SBJ08', 'SBJ08_8'])
    ax3 = fig.add_subplot(154)
    ax3.set_title('CUSTOM')
    ax3.set_ylabel('AUC')
    ax3.set_xlabel('Repetition')
    ax3.set_ylim(y_lim)
    ax3.yaxis.grid(True)
    result_acc_1 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet2_epoch_1')
    result_acc_2 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet2_epoch_2')
    result_acc_3 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet2_epoch_3')
    result_acc_4 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet2_epoch_4')
    result_acc_5 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet2_epoch_5')
    result_acc_6 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet2_epoch_6')
    result_acc8_1 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet2_8ch_epoch_1')
    result_acc8_2 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet2_8ch_epoch_2')
    result_acc8_3 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet2_8ch_epoch_3')
    result_acc8_4 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet2_8ch_epoch_4')
    result_acc8_5 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet2_8ch_epoch_5')
    result_acc8_6 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet2_8ch_epoch_6')
    result_mean = {}
    result_mean_all_subs['custom'] = np.zeros(6)
    result_mean8 = {}
    result_mean_all_subs8['custom'] = np.zeros(6)
    for sbj in SBJ_PLOT:
        result_mean[sbj] = [np.mean(np.array(result_acc_1[sbj])),
                            np.mean(np.array(result_acc_2[sbj])),
                            np.mean(np.array(result_acc_3[sbj])),
                            np.mean(np.array(result_acc_4[sbj])),
                            np.mean(np.array(result_acc_5[sbj])),
                            np.mean(np.array(result_acc_6[sbj]))
                            ]
        result_mean_all_subs['custom'] = result_mean_all_subs['custom'] + np.squeeze(np.array(result_mean[sbj]))
        result_mean8[sbj] = [np.mean(np.array(result_acc8_1[sbj])),
                             np.mean(np.array(result_acc8_2[sbj])),
                             np.mean(np.array(result_acc8_3[sbj])),
                             np.mean(np.array(result_acc8_4[sbj])),
                             np.mean(np.array(result_acc8_5[sbj])),
                             np.mean(np.array(result_acc8_6[sbj])),
                             ]
        result_mean_all_subs8['custom'] = result_mean_all_subs8['custom'] + np.squeeze(np.array(result_mean8[sbj]))
        stds = [np.mean(np.std(result_acc_1[sbj])),
                np.std(np.array(result_acc_2[sbj])),
                np.std(np.array(result_acc_3[sbj]))]
        print(result_mean[sbj])
        print([stds])
        print('  ')
        ax3.plot(['1', '2', '3', '4', '5', '6'], result_mean[sbj], 'o-', color=COLOR[sbj])
        ax3.plot(['1', '2', '3', '4', '5', '6'], result_mean8[sbj], 's--', color=COLOR[sbj])
    result_mean_all_subs['custom'] = result_mean_all_subs['custom']/8
    result_mean_all_subs8['custom'] = result_mean_all_subs8['custom']/8
    ax3.legend(['SBJ01', 'SBJ01_8', 'SBJ02', 'SBJ02_8',
                'SBJ03', 'SBJ03_8', 'SBJ04', 'SBJ04_8',
                'SBJ05', 'SBJ05_8', 'SBJ06', 'SBJ06_8',
                'SBJ07', 'SBJ07_8', 'SBJ08', 'SBJ08_8'])

    ax = fig.add_subplot(155)
    ax.plot(['1', '2', '3', '4', '5', '6'], result_mean_all_subs['csplda'], 'o-', color=COLOR['01'])
    ax.plot(['1', '2', '3', '4', '5', '6'], result_mean_all_subs8['csplda'], 's--', color=COLOR['01'])
    ax.plot(['1', '2', '3', '4', '5', '6'], result_mean_all_subs['eegnet'], 'o-', color=COLOR['02'])
    ax.plot(['1', '2', '3', '4', '5', '6'], result_mean_all_subs8['eegnet'], 's--', color=COLOR['02'])
    ax.plot(['1', '2', '3', '4', '5', '6'], result_mean_all_subs['effnetv2'], 'o-', color=COLOR['03'])
    ax.plot(['1', '2', '3', '4', '5', '6'], result_mean_all_subs8['effnetv2'], 's--', color=COLOR['03'])
    ax.plot(['1', '2', '3', '4', '5', '6'], result_mean_all_subs['custom'], 'o-', color=COLOR['04'])
    ax.plot(['1', '2', '3', '4', '5', '6'], result_mean_all_subs8['custom'], 's--', color=COLOR['04'])

    ax.legend(['CSP-LDA', 'CSP-LDA_8', 'EEGNET', 'EEGNET_8', 'EFFNETv2', 'EFFNETV2_8', 'CUSTOM', 'CUSTOM_8'])
    ax.set_ylabel('AUC')
    ax.set_xlabel('Repetition')
    ax.set_ylim(y_lim)
    ax.yaxis.grid(True)
    # plt.ylim([0.8, 1.0])
    # plt.xlabel('Number of Samples Averaged')
    # plt.ylabel('AUC')
    # plt.grid(axis='y')
    plt.show()
    return


def _acc_plot():
    fig = plt.figure(1)
    acc_lines = [2, 7, 12, 17, 22, 27, 32, 37]
    auc_lines = [6, 11, 16, 11, 26, 21, 36, 41]
    X = ['1', '2', '3', '4', '5', '6']
    COLOR_1 = ['maroon', 'lightseagreen', 'navy']
    COLOR_2 = ['maroon', 'lightseagreen', 'navy']
    ax1 = fig.add_subplot(111)
    color_ind = 0
    for n_ch in [3, 8, 0]:
        filename = 'D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_' + str(n_ch) + 'ch.csv'
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            acc_results_csplda = []
            for row in csv_reader:
                line_count = line_count + 1
                if line_count in acc_lines:
                    acc_results_csplda.append((np.array(row[1:]).astype(dtype=float)))
        csv_file.close()
        acc_results_csplda = np.array(acc_results_csplda)
        acc_results_csplda_mean = np.mean(acc_results_csplda, axis=0)
        ax1.plot(X, acc_results_csplda_mean, 'o-.', color=COLOR_1[color_ind])
        color_ind = color_ind + 1
    color_ind = 0
    for n_ch in [3, 8, 0]:
        filename = 'D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_' + str(n_ch) + 'ch.csv'
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            acc_results_eegnet = []
            for row in csv_reader:
                line_count = line_count + 1
                if line_count in acc_lines:
                    acc_results_eegnet.append((np.array(row[1:]).astype(dtype=float)))
        csv_file.close()
        acc_results_eegnet = np.array(acc_results_eegnet)
        acc_results_eegnet_mean = np.mean(acc_results_eegnet, axis=0)
        ax1.plot(X, acc_results_eegnet_mean, 'o--', color=COLOR_1[color_ind])
        color_ind = color_ind + 1
    color_ind = 0
    for n_ch in [3, 8, 0]:
        filename = 'D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_' + str(n_ch) + 'ch.csv'
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            acc_results_effnet = []
            for row in csv_reader:
                line_count = line_count + 1
                if line_count in acc_lines:
                    acc_results_effnet.append((np.array(row[1:]).astype(dtype=float)))
        csv_file.close()
        acc_results_effnet = np.array(acc_results_effnet)
        acc_results_effnet_mean = np.mean(acc_results_effnet, axis=0)
        ax1.plot(X, acc_results_effnet_mean, 'v:', color=COLOR_1[color_ind])
        color_ind = color_ind + 1
    color_ind = 0
    for n_ch in [3, 8, 0]:
        filename = 'D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_' + str(n_ch) + 'ch.csv'
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            acc_results_custom = []
            for row in csv_reader:
                line_count = line_count + 1
                if line_count in acc_lines:
                    acc_results_custom.append((np.array(row[1:]).astype(dtype=float)))
        csv_file.close()
        acc_results_custom = np.array(acc_results_custom)
        acc_results__custom_mean = np.mean(acc_results_custom, axis=0)
        ax1.plot(X, acc_results__custom_mean, 's-', color=COLOR_2[color_ind])
        color_ind = color_ind + 1
    legends = ['CSPLDA 3 CH', 'CSPLDA 8 CH', 'CSPLDA 64 CH',
               'EEGNET 3 CH', 'EEGNET 8 CH', 'EEGNET 64 CH',
               'EFFNET 3 CH', 'EFFNET 8 CH', 'EFFNET 64 CH',
               'CUSTOM 3 CH', 'CUSTOM 8 CH', 'CUSTOM 64 CH']
    ax1.legend(legends)
    ax1.set_ylabel('ACC')
    ax1.set_xlabel('Number of trials')
    ax1.yaxis.grid(True)
    plt.show()
    print('a')
    return


def _acc_plot_2():
    fig = plt.figure(1)
    acc_lines = [2, 7, 12, 17, 22, 27, 32, 37]
    f1_lines = [5, 10, 15, 20, 25, 30, 35, 40]
    auc_lines = [6, 11, 16, 11, 26, 21, 36, 41]
    X = ['1', '2', '3', '4', '5', '6']
    STYLE = {
        'LDA': 's-',
        'EEGNET': 'mediumseagreen',
        'EFFNET': 'royalblue',
        'CUSTOM': 'darkred'
    }
    COLOR = {
        'LDA': 'lightcoral',
        'EEGNET': 'darkcyan',
        'EFFNET': 'royalblue',
        'VIT': 'mediumpurple',
        'CUSTOM': 'darkred'
    }
    FILE_NAMES = {
        'LDA': 'D:/Code/PycharmProjects/P300_detect/results/results_noICA_loss_csplda_',
        'EEGNET': 'D:/Code/PycharmProjects/P300_detect/results/results_noICA_loss_eegnet_',
        'EFFNET': 'D:/Code/PycharmProjects/P300_detect/results/results_noICA_loss_effnetv2_',
        'VIT': 'D:/Code/PycharmProjects/P300_detect/results/results_noICA_loss_vit_',
        'CUSTOM': 'D:/Code/PycharmProjects/P300_detect/results/results_noICA_loss_custom_'
    }
    TITLES = {
        3: '3 CH',
        8: '8 CH',
        0: 'All CH'
    }
    font_title = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=18)
    font = font_manager.FontProperties(family='Times New Roman', style='normal', size=18)
    ax_acc = {}
    ax_auc = {}
    ax_f1 = {}
    ACC = {}
    AUC = {}
    F1 = {}
    n_1 = 0
    for n_ch in [3, 8, 0]:
        if n_ch == 3:
            ax_acc[n_ch] = fig.add_subplot(int('33' + str(1 + n_1)))
            ax_auc[n_ch] = fig.add_subplot(int('33' + str(4 + n_1)))
            ax_f1[n_ch] = fig.add_subplot(int('33' + str(7 + n_1)))
        else:
            ax_acc[n_ch] = fig.add_subplot(int('33' + str(1 + n_1)), sharey=ax_acc[3])
            ax_auc[n_ch] = fig.add_subplot(int('33' + str(4 + n_1)), sharey=ax_auc[3])
            ax_f1[n_ch] = fig.add_subplot(int('33' + str(7 + n_1)), sharey=ax_f1[3])
        ax_acc[n_ch].set_xticks([])
        ax_auc[n_ch].set_xticks([])
        ax_acc[n_ch].set_ylim([0.7, 1.0])
        ax_auc[n_ch].set_ylim([0.65, 1.0])
        ax_f1[n_ch].set_ylim([0.75, 1.0])
        ax_acc[n_ch].set_title(TITLES[n_ch], fontsize=18, font='Times New Roman')
        ax_acc[n_ch].grid(axis='y')
        ax_auc[n_ch].grid(axis='y')
        ax_f1[n_ch].grid(axis='y')
        if n_ch == 8:
            ax_f1[n_ch].set_xlabel('Number of trials', fontsize=18, font='Times New Roman')
        if n_ch == 3:
            ax_acc[n_ch].set_ylabel('ACC', fontsize=18, font='Times New Roman')
            plt.yticks(fontname='Times New Roman', size=18)
            ax_auc[n_ch].set_ylabel('AUC', fontsize=18, font='Times New Roman')
            plt.yticks(fontname='Times New Roman', size=18)
            ax_f1[n_ch].set_ylabel('F1', fontsize=18, font='Times New Roman')
            plt.yticks(fontname='Times New Roman', size=18)
        else:
            ax_acc[n_ch].tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            ax_auc[n_ch].tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            ax_f1[n_ch].tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False)
        for key in FILE_NAMES.keys():
            with open(FILE_NAMES[key] + str(n_ch) + 'ch.csv') as csv_file:
                csv_reader = csv.reader(csv_file)
                line_count = 0
                acc_results = []
                auc_results = []
                f1_results = []
                for row in csv_reader:
                    line_count = line_count + 1
                    if line_count in acc_lines:
                        acc_results.append(np.array(row[1:]).astype(dtype=float))
                    elif line_count in auc_lines:
                        auc_results.append(np.array(row[1:]).astype(dtype=float))
                    elif line_count in f1_lines:
                        f1_results.append(np.array(row[1:]).astype(dtype=float))
                acc_results = np.array(acc_results)
                auc_results = np.array(auc_results)
                f1_results = np.array(f1_results)
                ACC[key] = np.mean(acc_results, axis=0)
                AUC[key] = np.mean(auc_results, axis=0)
                F1[key] = np.mean(f1_results, axis=0)
            ax_acc[n_ch].plot(X, ACC[key], 's-', color=COLOR[key])
            ax_auc[n_ch].plot(X, AUC[key], 's-', color=COLOR[key])
            ax_f1[n_ch].plot(X, F1[key], 's-', color=COLOR[key])
            plt.xticks(fontname='Times New Roman', size=18)
            # ax_acc.yaxis.grid(True)
            # ax_auc.yaxis.grid(True)
            # ax_f1.yaxis.grid(True)
        if n_ch == 8:
            ax_auc[n_ch].legend(FILE_NAMES.keys(), loc='upper center', bbox_to_anchor=(0.5, -1.4), prop=font, ncol=5)

        n_1 = n_1 + 1
    plt.tight_layout()
    plt.show()
    print('a')


def _signal_plot(SBJ_PLOT, CH=0):
    files = os.listdir(FOLDER_DATA)
    fig = plt.figure()
    k = 1
    time_axis = np.linspace(-0.2, 1.0, num=int(1.2*125), endpoint=False)
    CLASS_NAME = {
        '4': 'Electric-1',  # nt estim
        '8': 'Electric-2-Target',  # t estim
        '16': 'Audio-1',  # nt astim
        '32': 'Audio-2',  # t astim
        '64': 'Vibration-1',  # nt vstim
        '128': 'Vibration-2'  # t vstim
    }
    CLASS_COLORS = {
        '4': 'violet',  # nt estim
        '8': 'purple',  # t estim
        '16': 'yellow',  # nt astim
        '32': 'gold',  # t astim
        '64': 'peachpuff',  # nt vstim
        '128': 'chocolate'  # t vstim
    }
    font_title = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=18)
    font = font_manager.FontProperties(family='Times New Roman', style='normal', size=14)
    for sbj in SBJ_PLOT:
        TRAIN = []
        ax = fig.add_subplot(int('1' + str(len(SBJ_PLOT)) + str(k)))
        k = k + 1
        for file_name in files:
            if (file_name.split('_')[0] == sbj) and (file_name.endswith('.set')):
                TRAIN.append(file_name)
        LEGENDS = []
        X, Y, events = util_preprocessing._build_dataset_eeglab_plot(FOLDER=FOLDER_DATA, TRAIN=TRAIN, TEST=[], CLASS=CLASS)
        for key in CLASS.keys():
            signal_mean = np.mean(X[np.where(events == int(key))[0], :, :, :], axis=0)
            if key == '8':
                ax.plot(time_axis, signal_mean[CH, :], linewidth=5, color=CLASS_COLORS[key])
            else:
                ax.plot(time_axis, signal_mean[CH, :], linewidth=3, color=CLASS_COLORS[key])
            ax.set_xlabel('Time/s', font='Times New Roman', size=16)
            ax.set_ylabel('Amplitude/muV', font='Times New Roman', size=16)
            LEGENDS.append(CLASS_NAME[key])
        ax.axvline(x=0, color='black')
        ax.axvline(x=0.2, color='grey', linestyle='--')
        ax.axvline(x=0.8, color='grey', linestyle='--')
        # ax.set_ylim([-10, 25])
        ax.legend(LEGENDS, loc='lower right', prop=font)
        ax.grid(axis='y')
        ax.set_title('Subject ' + str(sbj), fontname='Times New Roman', size=16)

    plt.show()


def _signal_plot_2(SBJ_PLOT, CH=0, multiNT=False):
    files = os.listdir(FOLDER_DATA)
    fig = plt.figure()
    k = 1
    time_axis = np.linspace(-0.2, 1.0, num=int(1.2*125), endpoint=False)
    CLASS_NAME = {
        '4': 'Electric-1',  # nt estim
        '8': 'Electric-2-Target',  # t estim
        '16': 'Audio-1',  # nt astim
        '32': 'Audio-2',  # t astim
        '64': 'Vibration-1',  # nt vstim
        '128': 'Vibration-2'  # t vstim
    }
    CLASS_COLORS = {
        '4': ['tan', 'oldlace'],  # nt estim
        '8': ['darkred', 'lightcoral'],  # t estim
        '16': ['darkgreen', 'mediumseagreen'],  # nt astim
        '32': ['darkcyan', 'lightblue'],  # t astim
        '64': ['navy', 'lightsteelblue'],  # nt vstim
        '128': ['darkviolet', 'plum']  # t vstim
    }
    X_all = None
    events_all = None
    X_t = None
    X_nt = {
        '4': None,
        '16': None,
        '32': None,
        '64': None,
        '128': None,
        'nt': None
    }
    X_noise = None
    font_title = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=18)
    font = font_manager.FontProperties(family='Times New Roman', style='normal', size=14)
    for sbj in SBJ_PLOT:
        TRAIN = []
        for file_name in files:
            if (file_name.split('_')[0] == sbj) and (file_name.endswith('.set')):
                TRAIN.append(file_name)
        X, Y, events = util_preprocessing._build_dataset_eeglab_plot(FOLDER=FOLDER_DATA, TRAIN=TRAIN, TEST=[], CLASS=CLASS)
        if X_all is None:
            X_all = X
            events_all = events
        else:
            X_all = np.concatenate([X_all, X], axis=0)
            events_all = np.concatenate([events_all, events])
    for key in CLASS.keys():
        ind = np.where(events_all == int(key))[0]
        if key in ['8']:
            X_t = X_all[ind, :, :, :]
        else:
            if multiNT is True:
                if X_nt[key] is None:
                    X_nt[key] = X_all[ind, :, :, :]
                else:
                    X_nt[key] = np.concatenate([X_nt[key], X_all[ind, :, :, :]], axis=0)
            # else:
            elif key in ['4']:
                if X_nt['nt'] is None:
                    X_nt['nt'] = X_all[ind, :, :, :]
                else:
                    X_nt['nt'] = np.concatenate([X_nt['nt'], X_all[ind, :, :, :]], axis=0)
            elif key not in ['16', '32', '64', '128']:
                if X_noise is None:
                    X_noise = X_all[ind, :, :, :]
                else:
                    X_noise = np.concatenate([X_noise, X_all[ind, :, :, :]], axis=0)

    T_mean = np.squeeze(np.mean(X_t, axis=0)[CH, :, :])
    T_std = np.squeeze(np.std(X_t, axis=0)[CH, :, :])
    if multiNT is True:
        NT_mean = {
            '4': None,
            '16': None,
            '32': None,
            '64': None,
            '128': None,
        }
        NT_std = {
            '4': None,
            '16': None,
            '32': None,
            '64': None,
            '128': None,
        }
        for key in NT_mean.keys():
            NT_mean[key] = np.squeeze(np.mean(X_nt[key], axis=0)[CH, :, :])
            NT_std[key] = np.squeeze(np.std(X_nt[key], axis=0)[CH, :, :])
    else:
        NT_mean = np.squeeze(np.mean(X_nt['nt'], axis=0)[CH, :, :])
        NT_std = np.squeeze(np.std(X_nt['nt'], axis=0)[CH, :, :])
        NOISE_mean = np.squeeze(np.mean(X_noise, axis=0)[CH, :, :])
        NOISE_std = np.squeeze(np.std(X_noise, axis=0)[CH, :, :])
    ax = fig.add_subplot(111)
    if multiNT is False:
        ax.plot(time_axis, NT_mean, color='darkcyan')
        ax.fill_between(time_axis,
                        NT_mean - NT_std,
                        NT_mean + NT_std,
                        color='lightblue', alpha=0.75)
        ax.plot(time_axis, NOISE_mean, color='darkgoldenrod')
        ax.fill_between(time_axis,
                        NOISE_mean - NOISE_std,
                        NOISE_mean + NOISE_std,
                        color='oldlace', alpha=0.75)
        LEGENDS = ['Non-target Ave.', 'Non-target STD', 'Target Ave.', 'Target STD']

    else:
        LEGENDS = []
        for key in NT_mean.keys():
            ax.plot(time_axis, NT_mean[key], color=CLASS_COLORS[key][0])
            ax.fill_between(time_axis,
                            NT_mean[key] - NT_std[key],
                            NT_mean[key] + NT_std[key],
                            color=CLASS_COLORS[key][1], alpha=0.2)
            LEGENDS.append(CLASS_NAME[key] + 'AVE')
            LEGENDS.append(CLASS_NAME[key] + 'STD')
        LEGENDS.append(CLASS_NAME['8'] + 'AVE')
        LEGENDS.append(CLASS_NAME['8'] + 'STD')
    ax.plot(time_axis, T_mean, color='darkred')
    ax.fill_between(time_axis,
                        T_mean - T_std,
                        T_mean + T_std,
                        color='lightcoral', alpha=0.4)
    # ax.set_xlabel('Time/s', size=16, font='Times New Roman')
    ax.set_xlabel(' ', size=16, font='Times New Roman')
    # ax.set_ylabel('Amplitude/muV', size=16, font='Times New Roman')
    ax.set_ylabel(' ', size=16, font='Times New Roman')
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False)
    plt.xticks(fontname='Times New Roman', size=14)
    plt.yticks(fontname='Times New Roman', size=14)
    ax.axvline(x=0, color='black')
    ax.axvline(x=0.2, color='grey', linestyle='--')
    ax.axvline(x=0.8, color='grey', linestyle='--')
    ax.set_ylim([-30, 50])
    ax.set_xlim([-0.2, 1.0])
    # ax.legend(LEGENDS, loc='upper left', prop=font)
    ax.legend(['NT-mean', 'NT-std', 'N-mean', 'N-std', 'T-mean', 'T-std'], loc='upper left', prop=font)
    TITLE = 'SBJ09'
    ax.set_title(TITLE, size=18, font='Times New Roman')
    ax.grid(axis='y')
    plt.show()


def _write_csv():
    # result1 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_0ch_epoch_1/')
    # result2 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_0ch_epoch_2/')
    # result3 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_0ch_epoch_3/')
    # result4 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_0ch_epoch_4/')
    # result5 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_0ch_epoch_5/')
    # result6 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_0ch_epoch_6/')
    # result1 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_3ch_epoch_1/')
    # result2 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_3ch_epoch_2/')
    # result3 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_3ch_epoch_3/')
    # result4 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_3ch_epoch_4/')
    # result5 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_3ch_epoch_5/')
    # result6 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_3ch_epoch_6/')
    # result1 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_8ch_epoch_1/')
    # result2 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_8ch_epoch_2/')
    # result3 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_8ch_epoch_3/')
    # result4 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_8ch_epoch_4/')
    # result5 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_8ch_epoch_5/')
    # result6 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_csplda_8ch_epoch_6/')
    # result1 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_0ch_epoch_1/')
    # result2 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_0ch_epoch_2/')
    # result3 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_0ch_epoch_3/')
    # result4 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_0ch_epoch_4/')
    # result5 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_0ch_epoch_5/')
    # result6 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_0ch_epoch_6/')
    # result1 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_3ch_epoch_1/')
    # result2 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_3ch_epoch_2/')
    # result3 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_3ch_epoch_3/')
    # result4 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_3ch_epoch_4/')
    # result5 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_3ch_epoch_5/')
    # result6 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_3ch_epoch_6/')
    # result1 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_8ch_epoch_1/')
    # result2 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_8ch_epoch_2/')
    # result3 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_8ch_epoch_3/')
    # result4 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_8ch_epoch_4/')
    # result5 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_8ch_epoch_5/')
    # result6 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_8ch_epoch_6/')
    # result1 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_0ch_epoch_1/')
    # result2 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_0ch_epoch_2/')
    # result3 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_0ch_epoch_3/')
    # result4 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_0ch_epoch_4/')
    # result5 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_0ch_epoch_5/')
    # result6 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_0ch_epoch_6/')
    # result1 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_3ch_epoch_1/')
    # result2 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_3ch_epoch_2/')
    # result3 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_3ch_epoch_3/')
    # result4 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_3ch_epoch_4/')
    # result5 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_3ch_epoch_5/')
    # result6 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_3ch_epoch_6/')
    # result1 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_8ch_epoch_1/')
    # result2 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_8ch_epoch_2/')
    # result3 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_8ch_epoch_3/')
    # result4 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_8ch_epoch_4/')
    # result5 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_8ch_epoch_5/')
    # result6 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_8ch_epoch_6/')
    # result1 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_vit_0ch_epoch_1/')
    # result2 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_vit_0ch_epoch_2/')
    # result3 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_vit_0ch_epoch_3/')
    # result4 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_vit_0ch_epoch_4/')
    # result5 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_vit_0ch_epoch_5/')
    # result6 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_vit_0ch_epoch_6/')
    # result1 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_vit_3ch_epoch_1/')
    # result2 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_vit_3ch_epoch_2/')
    # result3 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_vit_3ch_epoch_3/')
    # result4 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_vit_3ch_epoch_4/')
    # result5 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_vit_3ch_epoch_5/')
    # result6 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_vit_3ch_epoch_6/')
    result1 = _read_acc('D:/Code/PycharmProjects/P300_detect/results/results_noICA_loss_vit_8ch_epoch_1/')
    result2 = _read_acc('D:/Code/PycharmProjects/P300_detect/results/results_noICA_loss_vit_8ch_epoch_2/')
    result3 = _read_acc('D:/Code/PycharmProjects/P300_detect/results/results_noICA_loss_vit_8ch_epoch_3/')
    result4 = _read_acc('D:/Code/PycharmProjects/P300_detect/results/results_noICA_loss_vit_8ch_epoch_4/')
    result5 = _read_acc('D:/Code/PycharmProjects/P300_detect/results/results_noICA_loss_vit_8ch_epoch_5/')
    result6 = _read_acc('D:/Code/PycharmProjects/P300_detect/results/results_noICA_loss_vit_8ch_epoch_6/')
    # result1 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_0ch_epoch_1/')
    # result2 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_0ch_epoch_2/')
    # result3 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_0ch_epoch_3/')
    # result4 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_0ch_epoch_4/')
    # result5 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_0ch_epoch_5/')
    # result6 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_0ch_epoch_6/')
    # result1 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_3ch_epoch_1/')
    # result2 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_3ch_epoch_2/')
    # result3 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_3ch_epoch_3/')
    # result4 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_3ch_epoch_4/')
    # result5 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_3ch_epoch_5/')
    # result6 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_3ch_epoch_6/')
    # result1 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_8ch_epoch_1/')
    # result2 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_8ch_epoch_2/')
    # result3 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_8ch_epoch_3/')
    # result4 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_8ch_epoch_4/')
    # result5 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_8ch_epoch_5/')
    # result6 = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_effnetv2_8ch_epoch_6/')
    # with open('results_noICA_loss_csplda_0ch.csv', 'w', encoding='UTF8', newline='') as f:
    # with open('results_noICA_loss_eegnet_0ch.csv', 'w', encoding='UTF8', newline='') as f:
    with open('results/results_noICA_loss_vit_8ch.csv', 'w', encoding='UTF8', newline='') as f:
    # with open('results_noICA_loss_custom_8ch.csv', 'w', encoding='UTF8', newline='') as f:
    # with open('results_noICA_loss_effnetv2_0ch.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epochs', '1', '2', '3', '4', '5', '6'])
        for i in range(8):
            row_acc = ['acc', result1[i][0], result2[i][0], result3[i][0],
                             result4[i][0], result5[i][0], result6[i][0]]
            row_precision = ['precision', result1[i][1], result2[i][1], result3[i][1],
                             result4[i][1], result5[i][1], result6[i][1]]
            row_recall = ['recall', result1[i][2], result2[i][2], result3[i][2],
                             result4[i][2], result5[i][2], result6[i][2]]
            row_f1 = ['f1', result1[i][3], result2[i][3], result3[i][3],
                             result4[i][3], result5[i][3], result6[i][3]]
            row_auc = ['auc', result1[i][4], result2[i][4], result3[i][4],
                      result4[i][4], result5[i][4], result6[i][4]]
            writer.writerow(row_acc)
            writer.writerow(row_precision)
            writer.writerow(row_recall)
            writer.writerow(row_f1)
            writer.writerow(row_auc)
    f.close()

    return


# _epoch_plot(SBJ_PLOT=['01', '02', '03', '04', '06', '07', '08', '09'])
# _epoch_plot_methods(SBJ_PLOT=['01', '02', '03', '04', '06', '07', '08', '09'])
# _plot_prediction(SBJ_PLOT=['02', '04'])
# _signal_plot(SBJ_PLOT=['01', '02', '03', '04', '06', '07', '08', '09'], CH=27)
# _signal_plot(SBJ_PLOT=['01'], CH=27)
# result = _read_acc('D:/Code/PycharmProjects/P300_detect/results_noICA_eegnet_0ch_epoch_1/')
_write_csv()
# with open('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_eegnet_0ch_epoch_1/01_01.json') as json_file:
# with open('D:/Code/PycharmProjects/P300_detect/results_noICA_loss_custom_0ch_epoch_4/01_01.json') as json_file:
#     data = json.load(json_file)
# plt.plot(data['Y_pred'], 'o')
# plt.plot(data['Y_true'], '-')
# plt.show()
# print('a')
# _acc_plot_2()
# _signal_plot_2(SBJ_PLOT=['01', '02', '03', '04', '06', '07', '08', '09'], CH=45, multiNT=False)
# _signal_plot_2(SBJ_PLOT=['01'], CH=29, multiNT=False)

# data_pkg = mne.read_epochs_eeglab(r'D:\Code\PycharmProjects\P300_detect\data\SEP BCI 125 0-20 no ICA\04_01.set')
# # montage = mne.channels.get_builtin_montages()
# data_pkg.ch_names
# data_pkg.set_montage('standard_1005', on_missing='warn')
#
# print('a')