import json
import keras.backend
import numpy as np
import os
import scipy
import mne
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import util_preprocessing
import util_tf
import util_torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
import torch
from sklearn.model_selection import KFold
from ModelD import ModelD


# FOLDER = r'D:/Code/PycharmProjects/P300_detect/data/SEP BCI 125 0-20 no ICA'
FOLDER = r'D:\Code\PycharmProjects\P300_detect\data\SEP BCI 125 0-20 with noise'
def _run_cnn_test():
    # out_len = 1, AUC
    TRAIN = []
    TEST = []
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
        '4': [0, 1],  # nt estim
        '8': [1, 0],  # t estim
        '16': [0, 1],  # nt astim
        '32': [1, 0],  # t astim
        '64': [0, 1],  # nt vstim
        '128': [1, 0]  # t vstim
    }
    # CLASS = {
    #     '4': [0],  # nt estim
    #     '8': [1],  # t estim
    #     '16': [0],  # nt astim
    #     '32': [0],  # t astim
    #     '64': [0],  # nt vstim
    #     '128': [0]  # t vstim
    # }
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
            strat = 2
            if strat == 1:
                X_train, Y_train, X_test, Y_test, class_weights, events_train, \
                sample_weights_train, sample_weights_test = util_preprocessing._build_dataset_eeglab(FOLDER=FOLDER, CLASS=CLASS,
                                                                                                     TRAIN=TRAIN, TEST=TEST,
                                                                                                     ch_last=False,
                                                                                                     trainset_ave=epochs,
                                                                                                     testset_ave=epochs,
                                                                                                     ch_select=CH_SELECT,
                                                                                                     rep=4)
            elif strat == 2:
                X_train, Y_train, X_test, Y_test = util_preprocessing._build_dataset_strat2(FOLDER, TRAIN, TEST, CLASS,
                                                                                            ch_select=CH_SELECT, rep_train=4,
                                                                                            rep_test=4,
                                                                                            mult=8,
                                                                                            from_rep0=False)
                # transpose to ch-last for tf2
                X_train = np.transpose(X_train, (0, 2, 3, 1))
                X_test = np.transpose(X_test, (0, 2, 3, 1))
            print(np.shape(X_train))
            print(np.shape(Y_train))
            print(np.shape(X_test))
            print(np.shape(Y_test))
            print(np.sum(Y_train, axis=0))
            print(np.sum(Y_test, axis=0))

            keras.backend.clear_session()
            callback_1 = util_tf.IterTracker(X_test=X_test, Y_test=Y_test)
            model = util_tf._eegnet(in_shape=np.shape(X_train)[-3:], out_shape=np.shape(Y_train)[-1])
            # model = util_tf._custom(in_shape=np.shape(X_train)[-3:], out_shape=np.shape(Y_train)[-1])
            # model = util_tf._effnetV2(in_shape=np.shape(X_train)[-3:], out_shape=np.shape(Y_train)[-1])
            # model = util_tf._vit(in_shape=np.shape(X_train)[-3:], out_shape=np.shape(Y_train)[-1])
            model.fit(x=X_train, y=Y_train, epochs=200, batch_size=16, callbacks=[callback_1])
            print('---------++++++++++++______________')
            print(callback_1.best_scores)
            print('DONE!')
            # SAVE_PATH = r'results_noICA_loss_eegnet_8ch_epoch_' + str(epochs) + '/'
            # SAVE_PATH = r'results_noICA_loss_custom_8ch_epoch_' + str(epochs) + '/'
            SAVE_PATH = r'results_noICA_effnetv2_0ch_epoch_' + str(epochs) + '/'
            # SAVE_PATH = r'results_noICA_loss_vit_0ch_epoch_' + str(epochs) + '/'
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

def _run_cnn_torch(epochs=1, flag1=True):
    CLASS = {
        '4': [0, 1],  # nt estim
        '8': [1, 0],  # t estim
        '16': [0, 1],  # nt astim
        '32': [0, 1],  # t astim
        '64': [0, 1],  # nt vstim
        '128': [0, 1]  # t vstim
    }
    # CH_SELECT = [9, 27, 45, 59, 43, 47, 50, 56]
    # CH_SELECT = [9, 27, 45]
    CH_SELECT = False
    if CH_SELECT is False:
        num_ch = 64
    else:
        num_ch = len(CH_SELECT)
    result_to_save = []
    for epochs in [1, 2, 3, 4, 5, 6]:
        for sbj in ['01', '02', '03', '04', '06', '07', '08', '09']:
        # for sbj in ['01']:
            # create TRAIN
            TRAIN = []
            for set in ['_01', '_02', '_03', '_04', '_05', '_06']:
                TRAIN.append(sbj + set + '.set')
            # run

            for item in TRAIN:
                TEST = [item]
            # if True:
            #     TEST = ['01_01.set']
                mults = [10, 15, 20, 25, 30, 35]
                b_sizes = [24, 32, 40, 48, 56, 64]
                # X_train, Y_train, X_test, Y_test = util_preprocessing._build_dataset_strat2(FOLDER, TRAIN, TEST, CLASS,
                #                                                          ch_select=CH_SELECT, rep_train=epochs, rep_test=epochs,
                #                                                                             mult=mults[epochs - 1], from_rep0=False)
                X_train, Y_train, X_test, Y_test, X_test_ext, Y_test_ext = util_preprocessing._build_dataset_strat2(FOLDER, TRAIN, TEST, CLASS,
                                                                                            ch_select=CH_SELECT,
                                                                                            rep_train=epochs,
                                                                                            rep_test=epochs,
                                                                                            mult=mults[epochs - 1],
                                                                                            from_rep0=False)
                # transpose to ch-first for torch
                # X_train = np.transpose(X_train, (0, 3, 1, 2))
                print(np.shape(X_train))
                print(np.shape(Y_train))
                # X_test = np.transpose(X_test, (0, 3, 1, 2))
                print(np.shape(X_test))
                print(np.shape(Y_test))
                print(np.sum(Y_train, axis=0))
                print(np.sum(Y_test, axis=0))

                model = util_torch.EEGNET(eeg_ch=num_ch)
                # model = util_torch.VIT(num_eegch=num_ch, num_heads=4, num_layers=2)
                util_torch._model_summary(model)
                data_set_train = util_torch.EegData(X_train, Y_train)
                data_set_test = util_torch.EegData(X_test, Y_test)
                train_set, val_set = torch.utils.data.random_split(data_set_train, [0.8, 0.2])

                data_lens = [len(train_set), len(val_set), len(data_set_test)]
                print(data_lens)
                b_size = b_sizes[epochs - 1]
                train_loader = torch.utils.data.DataLoader(train_set, batch_size=b_size, shuffle=True, num_workers=0)
                val_loader = torch.utils.data.DataLoader(val_set, batch_size=b_size, shuffle=False, num_workers=0)
                test_loader = torch.utils.data.DataLoader(data_set_test, batch_size=b_size, shuffle=False, num_workers=0)


                fitted_model, out = util_torch._fit(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

                result_to_save.append([TEST[0].split('.')[0], epochs,
                                       out['loss'], out['acc'], out['prec'], out['recall'], out['f1']])
    save_name = 'results/torch_eegnet_0ch.csv'
    # save_name = 'results/torch_vit_0ch.csv'
    with open(save_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['test', 'epochs', 'loss', 'acc', 'prec', 'recall', 'f1'])
        for row in result_to_save:
            writer.writerow(row)
    f.close()

    return


def _run_cnn_torch_strat3(trial_epochs=[1], from_npz=False, overwrite=True):
    # CH_SELECT = [9, 27, 45, 59, 43, 47, 50, 56]
    # CH_SELECT = [9, 27, 45, 59, 43, 47, 25, 29]
    # CH_SELECT = [9, 27, 25, 29]
    CH_SELECT = False
    if CH_SELECT is False:
        num_ch = 64
    else:
        num_ch = len(CH_SELECT)
    result_to_save = []
    save_name = 'results/torch_eegnetestim_0ch.csv'
    # save_name = 'results/torch_vit_0ch.csv'
    # save_name = 'results/torch_eegnetresC_0ch_improve1.csv'
    # save_name = 'results/torch_eegnetvitA_0ch_improve.csv'
    # save_name = 'results/torch_resnet_0ch_tbc.csv'
    if not os.path.isfile(save_name) or overwrite is True:
        with open(save_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['test', 'epochs',
                             'loss', 'acc', 'prec', 'recall', 'f1',
                             'ext_loss', 'ext_acc', 'ext_prec', 'ext_recall', 'ext_f1', 'fp_over_p'])
            f.close()
    for epochs in trial_epochs:
        for sbj in ['01', '02', '03', '04', '06', '07', '08', '09']:
        # for sbj in ['02']:
            # create TRAIN
            TRAIN = []
            for set in ['_01', '_02', '_03', '_04', '_05', '_06']:
                TRAIN.append(sbj + set + '.set')
            # run
            for item in TRAIN:
                TEST = [item]
            # if True:
            #     TEST = ['01_01.set']
                batch_size_scale = 64
                batch_size_schedule = [1 * batch_size_scale, 8 * batch_size_scale, 16 * batch_size_scale,
                                       32 * batch_size_scale, 32 * batch_size_scale, 32 * batch_size_scale]
                if from_npz:
                    load_name = os.path.join('D:/Data/SEP w noise/', load_name)
                    loaded = np.load(load_name)
                    X_train_ = loaded['x_train']
                    Y_train_ = loaded['y_train']
                    X_val_ = loaded['x_val']
                    Y_val_ = loaded['y_val']
                    X_test = loaded['x_test']
                    Y_test = loaded['y_test']
                    X_test_ext = loaded['x_test_ext']
                    Y_test_ext = loaded['y_test_ext']
                else:
                    FOLDER = r'data\SEP BCI 125 0-20 with noise'
                    random_seed = 4080
                    util_torch.seed_torch(seed=random_seed)
                    X_train, Y_train, X_test, Y_test, X_test_ext, Y_test_ext = util_preprocessing._build_dataset_strat3(FOLDER, TRAIN, TEST,
                                                                                                num_reps=epochs)
                    X_train_, Y_train_, X_val_, Y_val_ = util_torch._manual_val_split(X_train, Y_train, ratio=0.85)
                if CH_SELECT is not False:
                    X_train_ = X_train_[:, :, CH_SELECT, :]
                    X_val_ = X_val_[:, :, CH_SELECT, :]
                    X_test = X_test[:, :, CH_SELECT, :]
                    X_test_ext = X_test_ext[:, :, CH_SELECT, :]

                print('train')
                print(np.shape(X_train_))
                print(np.shape(Y_train_))
                print('test')
                print(np.shape(X_test))
                print(np.shape(Y_test))
                print('test_ext')
                print(np.shape(X_test_ext))
                print(np.shape(Y_test_ext))
                print('[target non_target]')
                print(np.sum(Y_train_, axis=0))
                print(np.sum(Y_test, axis=0))
                print(np.sum(Y_test_ext, axis=0))
                load_name = item.split('.')[0] + '_epoch11_' + str(epochs) + '.npz'
                print(load_name)

                model = util_torch.EEGNET(eeg_ch=num_ch)
                # model = util_torch.EEGNET_RES1(num_eegch=num_ch, num_res_module_1=1, num_res_module_2=1)
                # model = util_torch.VIT(num_eegch=num_ch, num_heads=4, num_layers=1)
                # model = util_torch.convVIT(num_eegch=num_ch, num_heads=4, num_layers=1)
                # model = util_torch.EEGNET_VIT_A(num_eegch=num_ch, num_heads=4, num_layers=1)
                constraints_1 = util_torch.weightConstraint(-1, 1)
                constraints_2 = util_torch.weightConstraint(-1, 1)
                model._modules['conv_spatial'].apply(constraints_1)
                model._modules['fc1'].apply(constraints_2)
                # model._modules['MLP_head']._modules['2'].apply(constraints_2)
                learning_rate = 0.005
                util_torch._model_summary(model)

                train_set = util_torch.EegData(X_train_, Y_train_)
                val_set = util_torch.EegData(X_val_, Y_val_)
                test_set = util_torch.EegData(X_test, Y_test)
                test_set_ext = util_torch.EegData(np.concatenate([X_test, X_test_ext], axis=0),
                                                       np.concatenate([Y_test, Y_test_ext], axis=0))
                sum_1 = np.sum(Y_train_, axis=0)
                class_weight = np.array([sum_1[1], sum_1[0]])/(sum_1[1] + sum_1[0])
                print(f'Class Weight: {class_weight}')
                class_weight = torch.from_numpy(class_weight).float()

                data_lens = [len(train_set), len(val_set), len(test_set), len(test_set_ext)]
                print(data_lens)
                batch_size = batch_size_schedule[epochs - 1]
                train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
                val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
                testext_loader = torch.utils.data.DataLoader(test_set_ext, batch_size=batch_size, shuffle=False, num_workers=0)

                fitted_model, out = util_torch._fit(model, train_loader=train_loader, val_loader=val_loader,
                                                    test_loader=test_loader, testext_loader=testext_loader,
                                                    class_weight=class_weight, lr=learning_rate)

                row = [TEST[0].split('.')[0], epochs,
                                       out['loss'], out['acc'], out['prec'], out['recall'], out['f1'],
                                       out['loss_ext'], out['acc_ext'], out['prec_ext'], out['recall_ext'],
                                       out['f1_ext'], out['fp_over_allp']]
                with open(save_name, 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                f.close()

    return


def _run_cnn_torch_kfold(trial_epochs=[1], overwrite=True):
    # CH_SELECT = [9, 27, 45, 59, 43, 47, 50, 56]
    # CH_SELECT = [9, 27, 45, 59, 43, 47, 25, 29]
    # CH_SELECT = [9, 27, 25, 29]
    CH_SELECT = False
    if CH_SELECT is False:
        num_ch = 64
    else:
        num_ch = len(CH_SELECT)
    model_save_name = 'modelDestim_0ch'
    save_name = 'results/' + model_save_name + '.csv'
    save_path = 'models/' + model_save_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save_name = 'results/torch_vit_0ch.csv'
    # save_name = 'results/torch_eegnetresC_0ch_improve1.csv'
    # save_name = 'results/torch_eegnetvitA_0ch_improve.csv'
    # save_name = 'results/torch_resnet_0ch_tbc.csv'
    if not os.path.isfile(save_name) or overwrite is True:
        with open(save_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['test', 'epochs',
                             'acc', 'prec', 'recall', 'f1',
                             'ext_acc', 'ext_prec', 'ext_recall', 'ext_f1', 'fp_over_p'])
            f.close()
    for epochs in trial_epochs:
        for sbj in ['01', '02', '03', '04', '06', '07', '08', '09']:
        # for sbj in ['02']:
            # create TRAIN
            DATA = []
            for set in ['_01', '_02', '_03', '_04', '_05', '_06']:
                DATA.append(sbj + set + '.set')
            # run
            for item in DATA:
                TEST = [item]
                TRAIN = DATA.copy()
                TRAIN.remove(item)
            # if True:
            #     TEST = ['01_01.set']
                batch_size_scale = 16
                batch_size_schedule = [1 * batch_size_scale, 8 * batch_size_scale, 16 * batch_size_scale,
                                       32 * batch_size_scale, 32 * batch_size_scale, 32 * batch_size_scale]
                FOLDER = r'data\SEP BCI 125 0-20 with noise'
                MODELS = []
                for val_num, val in enumerate(TRAIN):
                    X_train, Y_train, X_val, Y_val, X_test, Y_test, X_test_ext, Y_test_ext = util_preprocessing._build_dataset_kfold(FOLDER, TRAIN, [val], TEST, epochs)
                    if CH_SELECT is not False:
                        X_train = X_train[:, :, CH_SELECT, :]
                        X_val = X_val[:, :, CH_SELECT, :]
                        X_test = X_test[:, :, CH_SELECT, :]
                        X_test_ext = X_test_ext[:, :, CH_SELECT, :]

                    print('train: ' + str(np.shape(X_train)))
                    print('val: ' + str(np.shape(X_val)))
                    print('test: ' + str(np.shape(X_test)))
                    print('test_ext: ' + str(np.shape(X_test_ext)))
                    print('[target non_target]')
                    print(np.sum(Y_train, axis=0))
                    print(np.sum(Y_test, axis=0))
                    print(np.sum(Y_test_ext, axis=0))

                    # model = util_torch.EEGNET(eeg_ch=num_ch)
                    # model = util_torch.EEGNET_RES1(num_eegch=num_ch, num_res_module_1=1, num_res_module_2=1)
                    # model = util_torch.VIT(num_eegch=num_ch, num_heads=4, num_layers=1)
                    # model = util_torch.convVIT(num_eegch=num_ch, num_heads=4, num_layers=1)
                    # model = util_torch.EEGNET_VIT_A(num_eegch=num_ch, num_heads=4, num_layers=1)
                    model = ModelD(CLASSES_NUM=2, channel_len=num_ch, train_len=125)
                    constraints_1 = util_torch.weightConstraint(-1, 1)
                    constraints_2 = util_torch.weightConstraint(-1, 1)
                    model._modules['conv_spatial'].apply(constraints_1)
                    model._modules['fc1'].apply(constraints_2)
                    # model._modules['MLP_head']._modules['2'].apply(constraints_2)
                    learning_rate = 0.005
                    util_torch._model_summary(model)

                    train_set = util_torch.EegData(X_train, Y_train)
                    val_set = util_torch.EegData(X_val, Y_val)
                    test_set = util_torch.EegData(X_test, Y_test)
                    test_set_ext = util_torch.EegData(np.concatenate([X_test, X_test_ext], axis=0),
                                                           np.concatenate([Y_test, Y_test_ext], axis=0))
                    sum_1 = np.sum(Y_train, axis=0)
                    class_weight = np.array([sum_1[1], sum_1[0]])/(sum_1[1] + sum_1[0])
                    print(f'Class Weight: {class_weight}')
                    class_weight = torch.from_numpy(class_weight).float()

                    data_lens = [len(train_set), len(val_set), len(test_set), len(test_set_ext)]
                    print(data_lens)
                    batch_size = batch_size_schedule[epochs - 1]
                    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
                    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
                    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
                    testext_loader = torch.utils.data.DataLoader(test_set_ext, batch_size=batch_size, shuffle=False, num_workers=0)
                    model_save_path = save_path + item.split('.')[0] + '_' + str(epochs) + 'trial_' + str(val_num + 1) + '.pth'

                    fitted_model = util_torch._fit_kfold(model, train_loader=train_loader, val_loader=val_loader,
                                                        class_weight=class_weight, lr=learning_rate, path=model_save_path)
                    # save model
                    MODELS.append(fitted_model)
                # Evaluation with K-fold models
                out = util_torch._eval_kfold(MODELS, test_loader=test_loader, testext_loader=testext_loader, mode='mean')

                row = [TEST[0].split('.')[0], epochs,
                                       out['acc'], out['prec'], out['recall'], out['f1'],
                                       out['acc_ext'], out['prec_ext'], out['recall_ext'],
                                       out['f1_ext'], out['fp_over_allp']]
                with open(save_name, 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                f.close()

    return


def _run_csplda_strat3(trial_epochs=[1], overwrite=True, copy_balance=False):
    # CH_SELECT = [9, 27, 45, 59, 43, 47, 50, 56]
    # CH_SELECT = [9, 27, 45]
    CH_SELECT = False
    if CH_SELECT is False:
        num_ch = 64
    else:
        num_ch = len(CH_SELECT)
    result_to_save = []
    save_name = 'results/torch_csp_0ch.csv'
    if not os.path.isfile(save_name) or overwrite is True:
        with open(save_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['test', 'epochs',
                             'loss', 'acc', 'prec', 'recall', 'f1',
                             'ext_loss', 'ext_acc', 'ext_prec', 'ext_recall', 'ext_f1', 'fp_over_p'])
            f.close()

    for epochs in trial_epochs:
        for sbj in ['01', '02', '03', '04', '06', '07', '08', '09']:
        # for sbj in ['02']:
            # create TRAIN
            TRAIN = []
            for set in ['_01', '_02', '_03', '_04', '_05', '_06']:
                TRAIN.append(sbj + set + '.set')
            # run
            for item in TRAIN:
                TEST = [item]
            # if True:
            #     TEST = ['01_01.set']
                FOLDER = r'D:\Code\PycharmProjects\P300_detect\data\SEP BCI 125 0-20 with noise'
                X_train, Y_train, X_test, Y_test, X_test_ext, Y_test_ext = util_preprocessing._build_dataset_strat3(FOLDER, TRAIN, TEST,
                                                                                                num_reps=epochs)
                if CH_SELECT is not False:
                    X_train = X_train[:, :, CH_SELECT, :]
                    X_test = X_test[:, :, CH_SELECT, :]
                    X_test_ext = X_test_ext[:, :, CH_SELECT, :]
                X_train = np.squeeze(X_train)
                X_test = np.squeeze(X_test)
                X_test_ext = np.squeeze(X_test_ext)

                print('train:')
                print(np.shape(X_train))
                print(np.shape(Y_train))
                print('test:')
                print(np.shape(X_test))
                print(np.shape(Y_test))
                print('test_ext:')
                print(np.shape(X_test_ext))
                print(np.shape(Y_test_ext))
                print('[target non_target]')
                print(np.sum(Y_train, axis=0))
                print(np.sum(Y_test, axis=0))
                print(np.sum(Y_test_ext, axis=0))
                load_name = item.split('.')[0] + '_epoch11_' + str(epochs) + '.npz'
                print(load_name)

                class_weights = np.sum(Y_train, axis=0)
                mult = class_weights[1]//class_weights[0] - 1

                # target: [1, 0], non_target: [0, 1] --> target: [0], non_target: [1]
                Y_train = np.argmax(Y_train, axis=1)
                if copy_balance:
                    pos_ind = list(np.where(Y_train == 0)[0])
                    for i in range(mult):
                        X_train = np.concatenate([X_train, X_train[pos_ind]], axis=0)
                        Y_train = np.concatenate([Y_train, Y_train[pos_ind]])
                Y_test = np.argmax(Y_test, axis=1)
                Y_test_ext = np.argmax(Y_test_ext, axis=1)
                X_test_ext = np.concatenate([X_test, X_test_ext], axis=0)
                Y_test_ext = np.concatenate([Y_test, Y_test_ext], axis=0)

                lda = LinearDiscriminantAnalysis()
                csp = CSP(n_components=16, reg=None, log=True, norm_trace=False)
                Feature_train = csp.fit_transform(X_train, Y_train)
                lda.fit(Feature_train, Y_train)
                Feature_test = csp.transform(X_test)
                Feature_test_ext = csp.transform(X_test_ext)

                proba = lda.predict_proba(Feature_test)
                proba_ext = lda.predict_proba(Feature_test_ext)
                preds = np.argmax(proba, axis=1)
                preds_ext = np.argmax(proba_ext, axis=1)

                f1, balanced_acc, precision, recall, _ = util_torch._compute_matrics(preds, Y_test, print_tpr=False)
                f1_ext, balanced_acc_ext, precision_ext, recall_ext, fp_over_n = util_torch._compute_matrics(preds_ext, Y_test_ext,
                                                                                             print_tpr=True)

                row = [TEST[0].split('.')[0], epochs,
                                       0, balanced_acc, precision, recall, f1,
                                       0, balanced_acc_ext, precision_ext, recall_ext,
                                       f1_ext, fp_over_n]
                with open(save_name, 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                f.close()

    return


def _build_dataset(trial_epochs):
    CH_SELECT = False
    if CH_SELECT is False:
        num_ch = 64
    else:
        num_ch = len(CH_SELECT)
    result_to_save = []
    for epochs in trial_epochs:
        for sbj in ['01', '02', '03', '04', '06', '07', '08', '09']:
            # for sbj in ['01']:
            # create TRAIN
            TRAIN = []
            for set in ['_01', '_02', '_03', '_04', '_05', '_06']:
                TRAIN.append(sbj + set + '.set')
            # run
            for item in TRAIN:
                TEST = [item]
                # if True:
                #     TEST = ['01_01.set']
                batch_size_schedule = [24, 32, 40, 48, 56, 64]
                # batch_size_schedule = [24, 32, 32, 8, 32, 32]
                X_train, Y_train, X_test, Y_test, X_test_ext, Y_test_ext = util_preprocessing._build_dataset_strat3(
                    FOLDER, TRAIN, TEST,
                    ch_select=CH_SELECT,
                    num_reps=epochs)
                X_train_, Y_train_, X_val_, Y_val_ = util_torch._manual_val_split(X_train, Y_train, ratio=0.85)

                save_name = item.split('.')[0] + '_epoch11_' + str(epochs) + '.npz'
                np.savez_compressed(save_name,
                                    x_train=X_train_, y_train=Y_train_,
                                    x_val=X_val_, y_val=Y_val_,
                                    x_test=X_test, y_test=Y_test,
                                    x_test_ext=X_test_ext, y_test_ext=Y_test_ext)
                # save_name = item.split('.')[0] + '_epoch_' + str(epochs) + '.json'
                # data_save = {
                #     'X1': X_train_.tolist(),
                #     'Y1': Y_train_.tolist(),
                #     'X2': X_val_.tolist(),
                #     'Y2': Y_val_.tolist(),
                #     'X3': X_test.tolist(),
                #     'Y3': Y_test.tolist(),
                #     'X4': X_test_ext.tolist(),
                #     'Y4': Y_test_ext.tolist()
                # }
                # with open(save_name, 'w') as file:
                #     json.dump(data_save, file)


# for i in [1, 2, 3, 4, 5, 6]:
# for i in [5, 6]:
#     _run_csp_lda(display=False, epochs=i)
#     _run_cnn_test2(epochs=i)
#     _run_cnn_torch()
# _run_cnn_test2(epochs=6)
# trial_epochs=[1, 2, 3, 4, 5, 6]
trial_epochs = [1, 2, 3]
# _run_csplda_strat3(trial_epochs=trial_epochs, overwrite=True, copy_balance=False)
# _run_cnn_torch_strat3(trial_epochs=trial_epochs, from_npz=False, overwrite=True)
_run_cnn_torch_kfold(trial_epochs=trial_epochs, overwrite=True)
# _build_dataset(trial_epochs=trial_epochs)


# file = r'D:\Code\PycharmProjects\P300_detect\01_01_epoch11_1.npz'
# loaded = np.load(file)
# X1 = loaded['x_train']
# print('1')