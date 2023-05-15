import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt


# class EEGNET(nn.Module):
#       # len=75
#     def __init__(self, eeg_ch):
#         # in_shape = [C_ch, H_eegch, W_time] [1, 64, 75]
#         super(EEGNET, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=8,
#                                kernel_size=(1, 37), stride=(1, 1),
#                                padding='same')
#         self.bn1 = nn.BatchNorm2d(num_features=8)
#         self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,
#                                kernel_size=(eeg_ch, 1), stride=(1, 1),
#                                padding='valid', groups=8)
#         self.bn2 = nn.BatchNorm2d(num_features=16)
#         self.conv3 = nn.Conv2d(in_channels=16, out_channels=16,
#                                kernel_size=(1, 15), stride=(1, 1),
#                                padding='same', groups=16)
#         self.conv4 = nn.Conv2d(in_channels=16, out_channels=16,
#                                kernel_size=(1, 1), stride=(1, 1),
#                                padding='valid')
#         self.bn3 = nn.BatchNorm2d(num_features=16)
#         self.fc1 = nn.Linear(64, 2)
#
#     def forward(self, x):
#         # Block 1
#         x = self.bn1(self.conv1(x)) # [8, ch, 75]
#         x = self.bn2(self.conv2(x)) # [16, 1. 75]
#         x = func.dropout(input=(func.avg_pool2d(func.elu(x), (1, 4))), p=0.25) # [16, 1, 18]
#         # Block 2
#         x = self.conv3(x)
#         x = self.bn3(self.conv4(x)) # [16, 1, 18]
#         x = func.dropout(input=(func.avg_pool2d(func.elu(x), (1, 4))), p=0.5) # [16, 1, 4]
#         x = torch.flatten(input=x, start_dim=1) # [64]
#         x = func.softmax(self.fc1(x), dim=-1) # [2]
#         return x


class EEGNET(nn.Module):
    # len=125
    def __init__(self, eeg_ch):
        # in_shape = [C_ch, H_eegch, W_time] [1, 64, 125]
        super(EEGNET, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8,
                               kernel_size=(1, 37), stride=(1, 1),
                               padding='same')
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,
                               kernel_size=(eeg_ch, 1), stride=(1, 1),
                               padding='valid', groups=8)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16,
                               kernel_size=(1, 15), stride=(1, 1),
                               padding='same', groups=16)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16,
                               kernel_size=(1, 1), stride=(1, 1),
                               padding='valid')
        self.bn3 = nn.BatchNorm2d(num_features=16)
        self.fc1 = nn.Linear(80, 2)

    def forward(self, x):
        # Block 1
        x = self.bn1(self.conv1(x)) # [8, ch, 125]
        x = self.bn2(self.conv2(x)) # [16, 1. 125]
        x = func.dropout(input=(func.avg_pool2d(func.elu(x), (1, 5))), p=0.25) # [16, 1, 25]
        # Block 2
        x = self.conv3(x)
        x = self.bn3(self.conv4(x)) # [16, 1, 25]
        x = func.dropout(input=(func.avg_pool2d(func.elu(x), (1, 5))), p=0.5) # [16, 1, 5]
        x = torch.flatten(input=x, start_dim=1) # [80]
        x = func.softmax(self.fc1(x), dim=-1) # [2]
        return x


class VIT(nn.Module):
    def __init__(self, num_eegch, num_heads, num_layers):
        # in_shape = [C_ch, H_eegch, W_time] [1, 64, 125]
        super(VIT, self).__init__()
        self.num_projected_features = 64
        self.qkv_len = 16
        self.scale = self.qkv_len**(-0.5)
        self.h = num_eegch
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.p = 5
        self.projection = nn.Linear(self.h*25, self.num_projected_features)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.num_projected_features))
        self.pe = nn.Parameter(torch.randn(1, self.p + 1, self.num_projected_features))

        self.transformer_layers = nn.ModuleList([])
        for i in range(self.num_layers):
            self.transformer_layers.append(nn.ModuleList([
                nn.Linear(self.num_projected_features, 3 * self.qkv_len * self.num_heads), # make qkv
                nn.Softmax(dim=-1), # scaled dot product att softmax
                nn.Linear(self.num_heads * self.qkv_len, self.num_projected_features), # multi head out
                nn.LayerNorm(self.num_projected_features),
                nn.LayerNorm(self.num_projected_features),
                nn.Sequential(nn.Linear(self.num_projected_features, 2*self.num_projected_features),
                              nn.GELU(),
                              nn.Dropout(0.25),
                              nn.Linear(2*self.num_projected_features, self.num_projected_features),
                              nn.Dropout(0.25))
            ]))
        self.MLP_head = nn.Sequential(nn.LayerNorm(self.num_projected_features),
                                      nn.Dropout(0.5),
                                      nn.Linear(self.num_projected_features, 2),
                                      nn.Softmax(dim=-1))
        # self.make_qkv = nn.Linear(self.num_projected_features, 3 * self.qkv_len * self.num_heads),
        # self.softmax = nn.Softmax(dim=-1),
        # self.multi_head_linear = nn.Linear(self.num_heads * self.qkv_len, self.num_projected_features)

    def forward(self, x):
        b = x.size()[0]
        # Patching and positional embedding x[b, c, h, w=125] --> [b, p + 1, f]
        x = torch.transpose(torch.reshape(x, (-1, 1, self.h, self.p, 25)), 3, 4)
        x = torch.transpose(torch.reshape(x, (-1, self.h*25, self.p)), 1, 2) # [b, p, h*25]
        x = self.projection(x)
        cls_token = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pe # embedded_patches [b, p+1, projected_len]
        # transformer
        for make_qkv, softmax, multi_head_out, layer_norm_1, layer_norm_2, MLP in self.transformer_layers:
            ##### QKV #####
            qkv = make_qkv(layer_norm_1(x)).chunk(3, dim=-1) # [b, p+1, qkv_len*num_heads]*3
            # q: [b, p+1, qkv_len*num_heads] -> [b, p+1, num_heads, qkv_len] -> [b, num_heads, p+1, qkv_len]
            q = qkv[0].reshape((b, self.p + 1, self.num_heads, self.qkv_len)).transpose(1, 2)
            # k: [b, p+1, qkv_len*num_heads] -> [b, p+1, num_heads, qkv_len] -> [b, num_heads, p+1, qkv_len] -> [b, num_heads, qkv_len, p+1]
            k = qkv[1].reshape((b, self.p + 1, self.num_heads, self.qkv_len)).transpose(1, 2).transpose(2, 3)
            # v: [b, p+1, qkv_len*num_heads] -> [b, p+1, num_heads, qkv_len] -> [b, num_heads, p+1, qkv_len]
            v = qkv[2].reshape((b, self.p + 1, self.num_heads, self.qkv_len)).transpose(1, 2)
            ##### Scaled Dot-Product Att #####
            # [b, num_heads, p+1, qkv_len] X [b, num_heads, qkv_len, p+1] X [b, num_heads, p+1, qkv_len] -> [b, num_heads, p+1, qkv_len]
            scaled_dot_produc_att = torch.matmul(softmax(torch.matmul(q, k)*self.scale), v)
            # [b, num_heads, p+1, qkv_len] -> [b, p+1, num_heads*qkv_len] -> [b, p+1, projected_len]
            multi_head_att = multi_head_out(scaled_dot_produc_att.transpose(1, 2).reshape((b, self.p + 1, self.num_heads*self.qkv_len)))
            x = multi_head_att + x
            ##### MLP #####
            x = MLP(layer_norm_2(x)) + x
        out = self.MLP_head(x[:, 0, :])
        return out

# a = torch.randn(10, 3, 2, 4)
# print(a[0, 1, 1, :])
# a = a.transpose(1, 2) # [10, 2, 3, 4]
# print(a[0, 1, 1, :])
# a = a.reshape(10, 2, 12)
# print(a[0, 1, :])



class EegData(torch.utils.data.Dataset):
    def __init__(self, raw_x, raw_y):
        """
        raw: raw eeg samples in numpy [n_batch, c, ch_eeg, time]
        """
        self.raw_x = raw_x
        self.raw_y = raw_y

    def __len__(self):
        return np.shape(self.raw_x)[0]

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        if torch.is_tensor(self.raw_x):
            sample = {'x': self.raw_x[item, :, :, :],
                      'y': self.raw_y[item, :]}
        else:
            sample = {'x': torch.from_numpy(self.raw_x[item, :, :, :]).float(),
                      'y': torch.from_numpy(self.raw_y[item, :]).float()}

        return sample


def _manual_val_split(X, Y, ratio):
    pos_ind = np.where(Y[:, 0] == 1)[0]
    neg_ind = np.where(Y[:, 1] == 1)[0]
    random.shuffle(pos_ind)
    random.shuffle(neg_ind)
    pos_train = pos_ind[:int(len(pos_ind) * ratio)]
    pos_val = pos_ind[int(len(pos_ind) * ratio):]
    neg_train = neg_ind[:int(len(neg_ind) * ratio)]
    neg_val = neg_ind[int(len(neg_ind) * ratio):]
    X_train = np.concatenate([X[pos_train], X[neg_train]], axis=0)
    X_val = np.concatenate([X[pos_val], X[neg_val]], axis=0)
    Y_train = np.concatenate([Y[pos_train], Y[neg_train]], axis=0)
    Y_val = np.concatenate([Y[pos_val], Y[neg_val]], axis=0)
    return X_train, Y_train, X_val, Y_val


def _compute_matrics(preds, true, print_tpr=False):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(preds)):
        if true[i] == 0:
            if preds[i] == 0:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            if preds[i] == 1:
                TN = TN + 1
            else:
                FP = FP + 1
    tpr = TP/(TP + FN)
    tnr = TN/(TN + FP)
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1 = 2*(precision*recall)/(precision + recall)
    balanced_acc = (tpr + tnr)/2
    if print_tpr:
        print(f'TPR: {tpr:.4f}. TNR: {tnr:.4f}. Precision: {precision:.4f}. Recall: {recall:.4f}.')
    return f1, balanced_acc


def _fit(model, train_loader, val_loader, test_loader, testext_loader, class_weight):
    # optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.05)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.0)
    criterion.to(device)
    model.to(device)
    log_train_loss = []
    log_val_loss = []
    log_test_loss = []
    log_testext_loss = []
    log_train_f1 = []
    log_val_f1 = []
    log_test_f1 = []
    log_testext_f1 = []
    log_train_acc = []
    log_val_acc = []
    log_test_acc = []
    log_testext_acc = []
    for epoch in range(200):
        running_train_loss = 0
        running_val_loss = 0
        running_test_loss = 0
        running_testext_loss = 0
        train_outputs = None
        val_outputs = None
        test_outputs = None
        testext_outputs = None
        for i_batch, data_batch in enumerate(train_loader):
            x = data_batch['x'].to(device)
            y = data_batch['y'].to(device)
            optimizer.zero_grad()
            outputs = model(x)
            train_loss = criterion(outputs, y)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
            if train_outputs is None:
                train_outputs = outputs
                train_labels = y
            else:
                train_outputs = torch.cat((train_outputs, outputs), dim=0)
                train_labels = torch.cat((train_labels, y), dim=0)
        _, train_predicts = torch.max(train_outputs, dim=1)
        _, train_labels = torch.max(train_labels, dim=1)
        f1, ba_acc = _compute_matrics(train_predicts, train_labels)
        # total = 0
        # correct = 0
        # for i, predict in enumerate(train_predicts):
        #     total = total + 1
        #     if predict == train_labels[i]:
        #         correct = correct + 1
        # train_acc = correct/total
        log_train_f1.append(f1)
        log_train_acc.append(ba_acc)
        log_train_loss.append(running_train_loss/len(train_loader))
        if val_loader is not None:
            for data_batch in val_loader:
                x = data_batch['x'].to(device)
                y = data_batch['y'].to(device)
                batch_size = y.size(dim=0)
                outputs = model(x)
                val_loss = criterion(outputs, y)
                running_val_loss += val_loss.item()
                if val_outputs is None:
                    val_outputs = outputs
                    val_labels = y
                else:
                    val_outputs = torch.cat((val_outputs, outputs), dim=0)
                    val_labels = torch.cat((val_labels, y), dim=0)
            _, val_predicts = torch.max(val_outputs, dim=1)
            _, val_labels = torch.max(val_labels, dim=1)
            f1, ba_acc = _compute_matrics(val_predicts, val_labels)
            # total = 0
            # correct = 0
            # for i, predict in enumerate(val_predicts):
            #     total = total + 1
            #     if predict == val_labels[i]:
            #         correct = correct + 1
            # val_acc = correct / total
            log_val_f1.append(f1)
            log_val_acc.append(ba_acc)
            log_val_loss.append(running_val_loss/len(val_loader))
        if test_loader is not None:
            for data_batch in test_loader:
                x = data_batch['x'].to(device)
                y = data_batch['y'].to(device)
                outputs = model(x)
                test_loss = criterion(outputs, y)
                running_test_loss +=test_loss.item()
                if test_outputs is None:
                    test_outputs = outputs
                    test_labels = y
                else:
                    test_outputs = torch.cat((test_outputs, outputs), dim=0)
                    test_labels = torch.cat((test_labels, y), dim=0)
            _, test_predicts = torch.max(test_outputs, dim=1)
            _, test_labels = torch.max(test_labels, dim=1)
            f1, ba_acc = _compute_matrics(test_predicts, test_labels)
            # total = 0
            # correct = 0
            # for i, predict in enumerate(test_predicts):
            #     total = total + 1
            #     if predict == test_labels[i]:
            #         correct = correct + 1
            # test_acc = correct/total
            log_test_f1.append(f1)
            log_test_acc.append(ba_acc)
            log_test_loss.append(running_test_loss / len(test_loader))
        if testext_loader is not None:
            for data_batch in testext_loader:
                x = data_batch['x'].to(device)
                y = data_batch['y'].to(device)
                outputs = model(x)
                testext_loss = criterion(outputs, y)
                running_testext_loss +=testext_loss.item()
                if testext_outputs is None:
                    testext_outputs = outputs
                    testext_labels = y
                else:
                    testext_outputs = torch.cat((testext_outputs, outputs), dim=0)
                    testext_labels = torch.cat((testext_labels, y), dim=0)
            _, testext_predicts = torch.max(testext_outputs, dim=1)
            _, testext_labels = torch.max(testext_labels, dim=1)
            f1, ba_acc = _compute_matrics(testext_predicts, testext_labels, print_tpr=True)
            # total = 0
            # correct = 0
            # for i, predict in enumerate(test_predicts):
            #     total = total + 1
            #     if predict == test_labels[i]:
            #         correct = correct + 1
            # test_acc = correct/total
            log_testext_f1.append(f1)
            log_testext_acc.append(ba_acc)
            log_testext_loss.append(running_testext_loss / len(testext_loader))
        print(f'epoch: {epoch} '
              f'loss: [{running_train_loss/len(train_loader):.4f} {running_val_loss/len(val_loader):.4f} '
              f'{running_test_loss/len(test_loader):.4f} {running_testext_loss/len(testext_loader):.4f}] '
              f'ba_acc: [{log_train_acc[-1]:.4f} {log_val_acc[-1]:.4f} {log_test_acc[-1]:.4f} {log_testext_acc[-1]:.4f}]'
              f'f1: [{log_train_f1[-1]:.4f} {log_val_f1[-1]:.4f} {log_test_f1[-1]:.4f} {log_testext_f1[-1]:.4f}]')
        # Early stopping
        if len(log_val_loss) > 20:
            vals = np.array(log_val_loss[-20:])
            trains = np.array(log_train_loss[-20:])
            if ((np.amax(vals) - np.amin(vals)) < 0.0005) or ((np.amax(trains) - np.amin(trains)) < 0.0005):
                print('Triggered early stopping.')
                break
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(test_predicts)):
        if test_predicts[i] == 0:
            if test_labels[i] == 0:
                tn = tn + 1
            else:
                fn = fn + 1
        else:
            if test_labels[i] == 1:
                tp = tp + 1
            else:
                fp = fp + 1
    total = tp + tn + fp + fn
    ACC = (tp + tn) / total
    PRECISION = tp/(tp + fp)
    RECALL = tp/(tp + fn)
    F1 = 2*PRECISION*RECALL/(PRECISION + RECALL)
    print(f'Finished! ACC: {ACC:.4f} PRECISION: {PRECISION:.4f} RECALL: {RECALL:.4f} F1: {F1:.4f}')
    out = {
        'acc': ACC,
        'prec': PRECISION,
        'recall': RECALL,
        'f1': F1,
        'loss': log_test_loss[-1]
    }
    # ax_loss = fig.add_subplot(121, title="Loss")
    # ax_acc = fig.add_subplot(122, title="ACC")
    # ax_loss.set_xlim([0, 200])
    # ax_acc.set_xlim([0, 200])
    # ax_loss.set_ylim([0, 1.5])
    # ax_acc.set_ylim([0, 1])
    # ax_loss.grid()
    # ax_acc.grid()
    #
    # ax_loss.plot(log_train_loss, label='Train')
    # ax_loss.plot(log_val_loss, label='Val')
    # ax_loss.plot(log_test_loss, label='Test')
    #
    # ax_acc.plot(log_train_acc, label='Train')
    # ax_acc.plot(log_val_acc, label='Val')
    # ax_acc.plot(log_test_acc, label='Test')
    #
    # ax_loss.legend()
    # ax_acc.legend()
    #
    # plt.show()
    return model, out


def _model_summary(model):
    print("model_summary")
    print()
    print("Layer_name" + "\t" * 7 + "Number of Parameters")
    print("=" * 100)
    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    print("\t" * 10)
    for i in layer_name:
        print()
        param = 0
        try:
            bias = (i.bias is not None)
        except:
            bias = False
        if not bias:
            param = model_parameters[j].numel() + model_parameters[j + 1].numel()
            j = j + 2
        else:
            param = model_parameters[j].numel()
            j = j + 1
        print(str(i) + "\t" * 3 + str(param))
        total_params += param
    print("=" * 100)
    print(f"Total Params:{total_params}")
