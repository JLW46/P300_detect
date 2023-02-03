import numpy as np
import os
import json
import matplotlib.pyplot as plt
import util_preprocessing


FOLDER_1 = r'D:/Code/PycharmProjects/P300_detect/results_noICA_epoch_1'
FOLDER_2 = r'D:/Code/PycharmProjects/P300_detect/results_noICA_epoch_2'
FOLDER_3 = r'D:/Code/PycharmProjects/P300_detect/results_noICA_epoch_3'

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
                result_acc[sbj] = [data['best_test_acc']]
            else:
                result_acc[sbj].append(data['best_test_acc'])
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
        label_to_box.append(str('N-T rep-1'))
        data_to_box.append(result_pred_1[sbj_plot + 'P'])
        label_to_box.append(str('T rep-1'))
        data_to_box.append(result_pred_2[sbj_plot + 'N'])
        label_to_box.append(str('N-T rep-2'))
        data_to_box.append(result_pred_2[sbj_plot + 'P'])
        label_to_box.append(str('T rep-2'))
        data_to_box.append(result_pred_3[sbj_plot + 'N'])
        label_to_box.append(str('N-T rep-3'))
        data_to_box.append(result_pred_3[sbj_plot + 'P'])
        label_to_box.append(str('T rep-3'))
    x_spacing = [1, 2, 4, 5, 7, 8]
    ax1 = fig.add_subplot(131)
    ax1.set_title(str('Subject ' + SBJ_PLOT[0]))
    ax1.set_ylabel('Prediction')
    box_dict_1 = ax1.boxplot(data_to_box[:6], labels=label_to_box[:6],
                             sym='+', positions=x_spacing, patch_artist=True, showmeans=True)
    ax1.yaxis.grid(True)
    ax2 = fig.add_subplot(132)
    ax2.set_title(str('Subject ' + SBJ_PLOT[1]))
    box_dict_2 = ax2.boxplot(data_to_box[6:12], labels=label_to_box[6:12],
                             sym='+', positions=x_spacing, patch_artist=True, showmeans=True)
    ax2.yaxis.grid(True)
    ax3 = fig.add_subplot(133)
    ax3.set_title(str('Subject ' + SBJ_PLOT[2]))
    box_dict_3 = ax3.boxplot(data_to_box[12:], labels=label_to_box[12:],
                             sym='+', positions=x_spacing, patch_artist=True, showmeans=True)
    ax3.yaxis.grid(True)
    COLOR = ['gold', 'mediumpurple', 'gold', 'mediumpurple', 'gold', 'mediumpurple']
    for i in range(6):
        box_dict_1.get('boxes')[i].set_facecolor(COLOR[i])
        box_dict_2.get('boxes')[i].set_facecolor(COLOR[i])
        box_dict_3.get('boxes')[i].set_facecolor(COLOR[i])
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
        plt.plot(['rep-1', 'rep-2', 'rep-3'], result_mean[sbj], 'o--')
    for k in range(len(SBJ_PLOT)):
        SBJ_PLOT[k] = str('Subject ' + SBJ_PLOT[k])
    plt.legend(SBJ_PLOT)
    plt.xlabel('repetition')
    plt.ylabel('accuracy (AUC)')
    plt.grid(axis='y')
    plt.show()
    return



# _epoch_plot(SBJ_PLOT=['01', '02', '03', '04', '06', '07', '08', '09'])
_plot_prediction(SBJ_PLOT=['02', '01', '04'])