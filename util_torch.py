import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np


class EEGNET(nn.Module):

    def __init__(self, eeg_ch):
        # in_shape = [C_ch, H_eegch, W_time] [1, 64, 75]
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
        self.fc1 = nn.Linear(64, 2)

    def forward(self, x):
        # Block 1
        x = self.bn1(self.conv1(x)) # [8, ch, 75]
        x = self.bn2(self.conv2(x)) # [16, 1. 75]
        x = func.dropout(input=(func.avg_pool2d(func.elu(x), (1, 4))), p=0.25) # [16, 1, 18]
        # Block 2
        x = self.conv3(x)
        x = self.bn3(self.conv4(x)) # [16, 1, 18]
        x = func.dropout(input=(func.avg_pool2d(func.elu(x), (1, 4))), p=0.5) # [16, 1, 4]
        x = torch.flatten(input=x, start_dim=1) # [64]
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
    optimizer = optim.SGD(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    for epoch in range(20):
        running_train_loss = 0
        running_val_loss = 0
        for i_batch, data_batch in enumerate(train_loader):
            x = data_batch['x'].to(device)
            y = data_batch['y'].to(device)
            optimizer.zero_grad()
            outputs = model(x)
            train_loss = criterion(outputs, y)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
        if val_loader is not None:
            for data_batch in val_loader:
                x = data_batch['x'].to(device)
                y = data_batch['y'].to(device)
                batch_size = y.size(dim=0)
                outputs = model(x)
                val_loss = criterion(outputs, y)
                running_val_loss += val_loss.item()
            print(f'{epoch + 1} train_loss: {running_train_loss/len(train_loader):.3f}. '
                  f'val_loss: {running_val_loss/len(val_loader):.3f}.')
        if test_loader is not None:
            test_outputs = None
            for data_batch in test_loader:
                x = data_batch['x'].to(device)
                y = data_batch['y'].to(device)
                outputs = model(x)
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
        running_train_loss = 0
        running_val_loss = 0
    print(f'Finished! Test ACC: {correct/total:.3f}')

    return model


