import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
import util_preprocessing
import util_tf

PATH1 = r'data/BCI_Comp_III_Wads_2004/Subject_A_Train.mat'
PATH2 = r'data/BCI_Comp_III_Wads_2004/Subject_A_Test.mat'
# PATH1 = r'data/BCI_Comp_III_Wads_2004/Subject_B_Train.mat'
# PATH2 = r'data/BCI_Comp_III_Wads_2004/Subject_B_Test.mat'
TRUE_LABELS_A = 'WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU'
TRUE_LABELS_B = 'MERMIROOMUHJPXJOHUVLEORZP3GLOO7AUFDKEFTWEOOALZOP9ROCGZET1Y19EWX65QUYU7NAK_4YCJDVDNGQXODBEV2B5EFDIDNR'
FREQ = 240 #Hz
WIN = int(0.8*FREQ)
CH = 64

data_pkg_test = util_preprocessing._build_dataset_p300_test(PATH2, WIN, CH, TRUE_LABELS_A, epochs=1, ch_last=True)
data_pkg_train = util_preprocessing._build_dataset_p300(PATH1, WIN, CH, epochs=1, ch_last=True)


X_train = None
Y_train = None
X_test = None
Y_test = None
for i in range(np.shape(data_pkg_train['X_norm'])[0]):
    if X_train is None:
        X_train = data_pkg_train['X_norm'][0]
        Y_train = data_pkg_train['Y'][0]
    else:
        X_train = np.concatenate([X_train, data_pkg_train['X_norm'][i]], axis=0)
        Y_train = np.concatenate([Y_train, data_pkg_train['Y'][i]], axis=0)
for i in range(np.shape(data_pkg_test['X_norm'])[0]):
    if X_test is None:
        X_test = data_pkg_test['X_norm'][0]
        Y_test = data_pkg_test['Y'][0]
    else:
        X_test = np.concatenate([X_test, data_pkg_test['X_norm'][i]], axis=0)
        Y_test = np.concatenate([Y_test, data_pkg_test['Y'][i]], axis=0)

model = util_tf._cecotti_cnn1(in_shape=[64, 192, 1], out_shape=2)
# model = util_tf._eegnet(in_shape=[64, 192, 1], out_shape=2)
model.fit(x=X_train, y=Y_train, epochs=20, batch_size=16)
print('Training Complete')
model.evaluate(x=X_test, y=Y_test)
# plt.plot(data_pkg['grand_ave_target'][32, :])
# plt.plot(data_pkg['grand_ave_nontarget'][32, :])
# print(np.shape(data_pkg['X']))
# print(np.shape(data_pkg['Y']))
# plt.plot(data_pkg['X'][0, 32, :])
# plt.plot(data_pkg['X'][1, 32, :])
# plt.plot(data_pkg['X_norm'][0, 32, :])
# plt.plot(data_pkg['X_norm'][1, 32, :])
# plt.legend(['target', 'non_target'])
# plt.grid()
# plt.show()


# print(np.shape(data_eeg[0]))
