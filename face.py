import numpy as np
import pandas as pd
import sys
import os
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils


df = pd.read_csv('fer2013.csv')

#df.head()
print(df.info())
X_train,traim_y,X_test,test_y = [],[],[],[]
for index,row in df.iterrows():
    val=row['pixels'].splits("")
    try:
        if 'Training' in row['usage']:
            X_train.append(np.array(val,'float32'))
        elif
            