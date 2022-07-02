import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt
#import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

import tensorflow as tf

from keras.models import Sequential,save_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM

dataset=pd.read_excel('Dataset.xls')

dataset.columns = ["chek_acc","mon","credit_his","purpose","Credit_amo","saving_amo","Pre_employ","instalrate","p_status","guatan","pre_res","property","age","installment","Housing","existing_cards","job","no_people","telephn","for_work","status"]

X = list(dataset.columns)
X.remove("status")
#to make output value  binary represenations 0-good 1-bad
Y = dataset.status-1
#print(Y)
Y=np.array(Y).reshape(-1,1)

X= pd.get_dummies(dataset[X])

X=np.array(X)

train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.25,random_state=10)

scaler_train=MinMaxScaler(feature_range=(0,1))
train_X=scaler_train.fit_transform(train_X)

scaler_test=MinMaxScaler(feature_range=(0,1))
test_X=scaler_test.fit_transform(test_X)

joblib.dump(scaler_test, 'scaler.save') 

train_X=np.reshape(train_X,(train_X.shape[0],1,train_X.shape[1]))
test_X=np.reshape(test_X,(test_X.shape[0],1,test_X.shape[1]))
print(train_X.shape)
print(test_X.shape)

model=Sequential()

model.add(LSTM(128,input_shape=(train_X.shape[1],train_X.shape[2]),activation='relu',return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64,input_shape=(train_X.shape[1],train_X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32,input_shape=(train_X.shape[1],train_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(train_X,train_Y,batch_size=32,epochs=30,validation_data=(test_X,test_Y))

model.save(filepath="lstm.h5",save_format="h5")

print(model.predict(test_X))