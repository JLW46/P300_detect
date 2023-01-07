import numpy as np
import os
import scipy
import matplotlib.pyplot as plt

PATH = r'data/bbci_3_2/Subject_A_Train.mat'
FREQ = 240 #Hz
WIN = int(0.8*FREQ)
CH = 64
TARGET = np.array([1, 0])
NONTARGET = np.array([0, 1])
# data_eeg = []
# file = open(PATH, 'r')
# lines = file.readlines()
# for line in lines:
#     data_eeg.append(np.squeeze(np.array(line.split(' '))))

data = scipy.io.loadmat(PATH)
Signal = data['Signal']
Flashing = data['Flashing']
StimType = data['StimulusType']
X_train = []
Y_train = []
cnt_target = 0
cnt_nontarget = 0
grand_ave_target = []
grand_ave_nontarget = []

for i in range(np.shape(Flashing)[0]):
    for j in range(np.shape(Flashing)[1] - 1):
        if Flashing[i, j + 1] == 1 and Flashing[i, j] == 0:
            x = np.reshape(Signal[i, j + 1:j + WIN + 1, :], (WIN, CH)).T
            x = (x - np.repeat(np.reshape(np.mean(x, axis=1), (CH, 1)), np.shape(x)[1], axis=1))/(np.repeat(np.reshape(np.std(x, axis=1), (CH, 1)), np.shape(x)[1], axis=1))
            X_train.append(x)
            if StimType[i, j + 1] == 1:
                grand_ave_target.append(X_train[-1])
                Y_train.append(TARGET)
                cnt_target = cnt_target + 1
            else:
                grand_ave_nontarget.append(X_train[-1])
                Y_train.append(NONTARGET)
                cnt_nontarget = cnt_nontarget + 1
X_train = np.array(X_train)
Y_train = np.array(Y_train)
grand_ave_target = np.array(grand_ave_target)
grand_ave_nontarget = np.array(grand_ave_nontarget)
grand_ave_target = np.mean(grand_ave_target, axis=0)
grand_ave_nontarget = np.mean(grand_ave_nontarget, axis=0)
print(np.shape(grand_ave_target))
print(np.shape(grand_ave_nontarget))

plt.plot(grand_ave_target[17, :])
plt.plot(grand_ave_nontarget[17, :])
plt.legend(['target', 'non_target'])
plt.grid()
plt.show()

print(cnt_target)
print(cnt_nontarget)

print(np.shape(X_train))
print(np.shape(Y_train))


print(len(data))

# print(np.shape(data_eeg[0]))
