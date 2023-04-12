import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
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


def _fit(model, train_loader, val_loader, test_loader):
    # optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.05)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    log_train_loss = []
    log_val_loss = []
    log_test_loss = []
    log_train_acc = []
    log_val_acc = []
    log_test_acc = []
    for epoch in range(200):
        running_train_loss = 0
        running_val_loss = 0
        running_test_loss = 0
        train_outputs = None
        val_outputs = None
        test_outputs = None
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
        total = 0
        correct = 0
        for i, predict in enumerate(train_predicts):
            total = total + 1
            if predict == train_labels[i]:
                correct = correct + 1
        train_acc = correct/total
        log_train_acc.append(train_acc)
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
            total = 0
            correct = 0
            for i, predict in enumerate(val_predicts):
                total = total + 1
                if predict == val_labels[i]:
                    correct = correct + 1
            val_acc = correct / total
            log_val_acc.append(val_acc)
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
            total = 0
            correct = 0
            for i, predict in enumerate(test_predicts):
                total = total + 1
                if predict == test_labels[i]:
                    correct = correct + 1
            test_acc = correct/total
            log_test_acc.append(test_acc)
            log_test_loss.append(running_test_loss/len(test_loader))
        print(f'epoch: {epoch} '
              f'loss: [{running_train_loss/len(train_loader):.4f} {running_val_loss/len(val_loader):.4f} {running_test_loss/len(test_loader):.4f}] '
              f'acc: [{train_acc:.4f} {val_acc:.4f} {test_acc:.4f}]')
        # Early stopping
        if len(log_val_loss) > 20:
            check = np.array(log_val_loss[-20:])
            if (np.amax(check) - np.amin(check)) < 0.001:
                print('Triggered early stopping.')
                break
    print(f'Finished! Test ACC: {correct / total:.3f}')
    fig = plt.figure()
    ax_loss = fig.add_subplot(121, title="Loss")
    ax_acc = fig.add_subplot(122, title="ACC")
    ax_loss.set_xlim([0, 200])
    ax_acc.set_xlim([0, 200])
    ax_loss.set_ylim([0, 1.5])
    ax_acc.set_ylim([0, 1])
    ax_loss.grid()
    ax_acc.grid()

    ax_loss.plot(log_train_loss, label='Train')
    ax_loss.plot(log_val_loss, label='Val')
    ax_loss.plot(log_test_loss, label='Test')

    ax_acc.plot(log_train_acc, label='Train')
    ax_acc.plot(log_val_acc, label='Val')
    ax_acc.plot(log_test_acc, label='Test')

    ax_loss.legend()
    ax_acc.legend()

    plt.show()
    return model


