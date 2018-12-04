# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:39:15 2018

@author: 呀呀呀呀呀可乐
"""
import matplotlib.pyplot as pyplot
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.datasets import mnist

#############数据准备
x_train=np.load('data_xtrain.npy')
x_test=np.load('data_xtest.npy')
#(10000,76,11)
y_tr=np.load('data_ytrain.npy')
#x_train=x_train.reshape(19000,44)
x_train=x_train.reshape(19000,80)
y_te=np.load('data_ytest.npy')
#x_test=x_test.reshape(19000,44)
x_test=x_test.reshape(19000,80)
y_tr=y_tr.reshape(19000,1)
y_train=np.array([[0 for i in range(2)] for j in range(19000)])
y_te=y_te.reshape(19000,1)
y_test=np.array([[0 for i in range(2)] for j in range(19000)])
for i in range(19000):
    if y_tr[i]==0:
        y_train[i][0]=1
        y_train[i][1]=0
    if y_tr[i]==1:
        y_train[i][0]=0
        y_train[i][1]=1
    if y_te[i]==0:
        y_test[i][0]=1
        y_test[i][1]=0
    if y_te[i]==1:
        y_test[i][0]=0
        y_test[i][1]=1


#####训练模型    
model=Sequential()
model.add(Dense(input_dim=80,units=100))
model.add(Activation('linear'))
#model.add(Dropout(0.5))
for i in range(5):
    model.add(Dense(units=100))
    model.add(Activation('tanh'))
    #model.add(Dropout(0.5))
    
model.add(Dense(units=2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adagrad',metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy',optimizer= SGD(lr=0.5),metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=10,epochs=20)

result=model.evaluate(x_test,y_test)
print('test acc:',result[1])