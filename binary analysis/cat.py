from __future__ import print_function
import pandas as pd
from keras.utils.np_utils import to_categorical
import numpy as np

print("Loading")


testlabel = pd.read_csv('train/classlabel.csv', header=None)
Y1 = testlabel.iloc[:,0]

y_test1 = np.array(Y1)

y_test= to_categorical(y_test1)

np.savetxt('train/classlabelonehot.csv', y_test, fmt='%01d')







