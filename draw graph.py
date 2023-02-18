import numpy as np
import os
import json
import matplotlib.pyplot as plt
import util_preprocessing


FOLDER_1 = r'D:/Code/PycharmProjects/P300_detect/results_noICA_epoch_1'
FOLDER_2 = r'D:/Code/PycharmProjects/P300_detect/results_noICA_epoch_2'
FOLDER_3 = r'D:/Code/PycharmProjects/P300_detect/results_noICA_epoch_3'
# FOLDER_1 = r'D:/Code/PycharmProjects/P300_detect/results_ICA_epoch_1'
# FOLDER_2 = r'D:/Code/PycharmProjects/P300_detect/results_ICA_epoch_2'
# FOLDER_3 = r'D:/Code/PycharmProjects/P300_detect/results_ICA_epoch_3'
FOLDER_DATA = r'D:\Code\PycharmProjects\P300_detect\data\SEP BCI 125 0-20 no ICA'
# FOLDER_DATA = r'D:\Code\PycharmProjects\P300_detect\data\SEP BCI 125 0-20 ICA'

CLASS = {
        '4': [0], # nt estim
        '8': [1], # t estim
        '16': [0], # nt astim
        '32': [0], # t astim
        '64': [0], # nt vstim
        '128': [0] # t vstim
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
    y_lim = [0.5, 1.0]


    ax0 = fig.add_subplot(151)
    ax0.set_title('CSP-LDA')
    ax0.set_ylabel('AUC')
    ax0.set_xlabel('Repetition')
    ax0.set_ylim(y_lim)
    ax0.yaxis.grid(True)
    # result_acc_1 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_epoch_1')
    # result_acc_2 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_epoch_2')
    # result_acc_3 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_epoch_3')
    # result_acc_4 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_epoch_4')
    # result_acc_5 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_epoch_5')
    # result_acc_6 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_epoch_6')
    result_acc_1 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_8ch_epoch_1')
    result_acc_2 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_8ch_epoch_2')
    result_acc_3 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_8ch_epoch_3')
    result_acc_4 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_8ch_epoch_4')
    result_acc_5 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_8ch_epoch_5')
    result_acc_6 = _read_auc(r'D:/Code/PycharmProjects/P300_detect/results_noICA_csplda_8ch_epoch_6')
    result_mean = {}
    result_mean_all_subs['csplda'] = np.zeros(6)
    for sbj in SBJ_PLOT:
        result_mean[sbj] = [np.mean(np.array(result_acc_1[sbj])),
                            np.mean(np.array(result_acc_2[sbj])),
                            np.mean(np.array(result_acc_3[sbj])),
                            np.mean(np.array(result_acc_4[sbj])),
                            np.mean(np.array(result_acc_5[sbj])),
                            np.mean(np.array(result_acc_6[sbj]))]
        result_mean_all_subs['csplda'] = result_mean_all_subs['csplda'] + np.squeeze(np.array(result_mean[sbj]))
        stds = [np.mean(np.std(result_acc_1[sbj])),
                np.std(np.array(result_acc_2[sbj])),
                np.std(np.array(result_acc_3[sbj]))]
        print(result_mean[sbj])
        print([stds])
        print('  ')
        ax0.plot(['1', '2', '3', '4', '5', '6'], result_mean[sbj], 'o-')
    result_mean_all_subs['csplda'] = result_mean_all_subs['csplda'] / 8
    ax0.legend(['SBJ01', 'SBJ02', 'SBJ03', 'SBJ04', 'SBJ05', 'SBJ06', 'SBJ07', 'SBJ08'])

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
    result_mean = {}
    result_mean_all_subs['eegnet'] = np.zeros(6)
    for sbj in SBJ_PLOT:
        result_mean[sbj] = [np.mean(np.array(result_acc_1[sbj])),
                            np.mean(np.array(result_acc_2[sbj])),
                            np.mean(np.array(result_acc_3[sbj])),
                            np.mean(np.array(result_acc_4[sbj])),
                            np.mean(np.array(result_acc_5[sbj])),
                            np.mean(np.array(result_acc_6[sbj]))
                            ]
        result_mean_all_subs['eegnet'] = result_mean_all_subs['eegnet'] + np.squeeze(np.array(result_mean[sbj]))
        stds = [np.mean(np.std(result_acc_1[sbj])),
                np.std(np.array(result_acc_2[sbj])),
                np.std(np.array(result_acc_3[sbj]))]
        print(result_mean[sbj])
        print([stds])
        print('  ')
        ax1.plot(['1', '2', '3', '4', '5', '6'], result_mean[sbj], 'o-')
    result_mean_all_subs['eegnet'] = result_mean_all_subs['eegnet'] / 8
    ax1.legend(['SBJ01', 'SBJ02', 'SBJ03', 'SBJ04', 'SBJ05', 'SBJ06', 'SBJ07', 'SBJ08'])

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
    result_mean = {}
    result_mean_all_subs['effnetv2'] = np.zeros(6)
    for sbj in SBJ_PLOT:
        result_mean[sbj] = [np.mean(np.array(result_acc_1[sbj])),
                            np.mean(np.array(result_acc_2[sbj])),
                            np.mean(np.array(result_acc_3[sbj])),
                            np.mean(np.array(result_acc_4[sbj])),
                            np.mean(np.array(result_acc_5[sbj])),
                            np.mean(np.array(result_acc_6[sbj]))
                            ]
        result_mean_all_subs['effnetv2'] = result_mean_all_subs['effnetv2'] + np.squeeze(np.array(result_mean[sbj]))
        stds = [np.mean(np.std(result_acc_1[sbj])),
                np.std(np.array(result_acc_2[sbj])),
                np.std(np.array(result_acc_3[sbj]))]
        print(result_mean[sbj])
        print([stds])
        print('  ')
        ax2.plot(['1', '2', '3', '4', '5', '6'], result_mean[sbj], 'o-')
    result_mean_all_subs['effnetv2'] = result_mean_all_subs['effnetv2'] / 8
    ax2.legend(['SBJ01', 'SBJ02', 'SBJ03', 'SBJ04', 'SBJ05', 'SBJ06', 'SBJ07', 'SBJ08'])

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
    result_mean = {}
    result_mean_all_subs['custom'] = np.zeros(6)
    for sbj in SBJ_PLOT:
        result_mean[sbj] = [np.mean(np.array(result_acc_1[sbj])),
                            np.mean(np.array(result_acc_2[sbj])),
                            np.mean(np.array(result_acc_3[sbj])),
                            np.mean(np.array(result_acc_4[sbj])),
                            np.mean(np.array(result_acc_5[sbj])),
                            np.mean(np.array(result_acc_6[sbj]))
                            ]
        result_mean_all_subs['custom'] = result_mean_all_subs['custom'] + np.squeeze(np.array(result_mean[sbj]))
        stds = [np.mean(np.std(result_acc_1[sbj])),
                np.std(np.array(result_acc_2[sbj])),
                np.std(np.array(result_acc_3[sbj]))]
        print(result_mean[sbj])
        print([stds])
        print('  ')
        ax3.plot(['1', '2', '3', '4', '5', '6'], result_mean[sbj], 'o-')
    result_mean_all_subs['custom'] = result_mean_all_subs['custom']/8
    ax3.legend(['SBJ01', 'SBJ02', 'SBJ03', 'SBJ04', 'SBJ05', 'SBJ06', 'SBJ07', 'SBJ08'])

    ax = fig.add_subplot(155)
    ax.plot(['1', '2', '3', '4', '5', '6'], result_mean_all_subs['csplda'], 'o-')
    ax.plot(['1', '2', '3', '4', '5', '6'], result_mean_all_subs['eegnet'], 'o-')
    ax.plot(['1', '2', '3', '4', '5', '6'], result_mean_all_subs['effnetv2'], 'o-')
    ax.plot(['1', '2', '3', '4', '5', '6'], result_mean_all_subs['custom'], 'o-')
    ax.legend(['CSP-LDA', 'EEGNET', 'EFFNETv2', 'CUSTOM'])
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
    for sbj in SBJ_PLOT:
        TRAIN = []
        ax = fig.add_subplot(int('1' + str(len(SBJ_PLOT)) + str(k)))
        k = k + 1
        for file_name in files:
            if (file_name.split('_')[0] == sbj) and (file_name.endswith('.set')):
                TRAIN.append(file_name)
        LEGENDS = []
        # X_train, Y_train, _, _, _, events_train, _, _ = util_preprocessing._build_dataset_eeglab(
        #     FOLDER=FOLDER_DATA, TRAIN=TRAIN, TEST=[], CLASS=CLASS,
        #     ch_last=False, trainset_ave=1, testset_ave=1, for_plot=False)
        X, Y, events = util_preprocessing._build_dataset_eeglab_plot(FOLDER=FOLDER_DATA, TRAIN=TRAIN, TEST=[], CLASS=CLASS)
        for key in CLASS.keys():
            signal_mean = np.mean(X[np.where(events == int(key))[0], :, :, :], axis=0)
            if key == '8':
                ax.plot(time_axis, signal_mean[CH, :], linewidth=5, color=CLASS_COLORS[key], zorder=1)
            else:
                ax.plot(time_axis, signal_mean[CH, :], linewidth=3, color=CLASS_COLORS[key])
            ax.set_xlabel('Time/s')
            ax.set_ylabel('Amplitude/muV')
            LEGENDS.append(CLASS_NAME[key])
        ax.axvline(x=0, color='black')
        ax.axvline(x=0.2, color='grey', linestyle='--')
        ax.axvline(x=0.8, color='grey', linestyle='--')
        # ax.set_ylim([-10, 25])
        ax.legend(LEGENDS, loc='lower right')
        ax.grid(axis='y')
        ax.set_title('Subject ' + str(sbj))
    plt.show()


# _epoch_plot(SBJ_PLOT=['01', '02', '03', '04', '06', '07', '08', '09'])
_epoch_plot_methods(SBJ_PLOT=['01', '02', '03', '04', '06', '07', '08', '09'])
# _plot_prediction(SBJ_PLOT=['02', '04'])
# _signal_plot(SBJ_PLOT=['01', '02', '03', '04', '06', '07', '08', '09'], CH=29)
# _signal_plot(SBJ_PLOT=['01'], CH=29)