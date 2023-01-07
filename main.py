import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
import util_preprocessing

PATH = r'data/BCI_Comp_III_Wads_2004/Subject_A_Train.mat'
FREQ = 240 #Hz
WIN = int(0.8*FREQ)
CH = 64
data_pkg = util_preprocessing._build_dataset_p300(PATH, WIN, CH, epchs=4)





plt.plot(data_pkg['grand_ave_target'][32, :])
plt.plot(data_pkg['grand_ave_nontarget'][32, :])
plt.legend(['target', 'non_target'])
plt.grid()
plt.show()


# print(np.shape(data_eeg[0]))
