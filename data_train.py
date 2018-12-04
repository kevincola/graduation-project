# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:02:58 2018

@author: 呀呀呀呀呀可乐
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
import random
import pylab  #这个库中含有random函数
from scipy.stats import levy_stable

noise=1
alpha=0
beta=10
noise_test=1
alpha_test=0
beta_test=10

data_ini=[[0 for i in range(19)] for j in range(1000)]
data_ini_test=[[0 for i in range(19)] for j in range(1000)]
data_de=[[0 for i in range(20)] for j in range(1000)]
data_de_test=[[0 for i in range(20)] for j in range(1000)]

data_BPSK=[[0 for i in range(800)]for j in range(1000)]
data_BPSK_test=[[0 for i in range(800)]for j in range(1000)]

data_noise=[[0 for i in range(800)] for j in range(1000)]
data_noise_test=[[0 for i in range(800)] for j in range(1000)]

S=[[0 for i in range(869)] for j in range(1000)] 
S_test=[[0 for i in range(869)] for j in range(1000)]

x_train=[[0 for i in range(869)] for j in range(1000)]
x_train=np.array(x_train)
y_train=[[0 for i in range(20)] for j in range(1000)]
y_train=np.array(y_train)
x_test=[[0 for i in range(869)] for j in range(1000)]
x_test=np.array(x_train)
y_test=[[0 for i in range(20)] for j in range(1000)]
y_test=np.array(y_test)

def BPSK(Rb,fc,fs,data):           #Rb为码元速率、fc为载波频率、fs为采样频率
        k=len(data)                #码元总个数
        num_sam=int(fs/Rb)         #每个码元需要采样的个数
        num_sum=k*num_sam          #需要的总的采样数
        data_bpsk=[0 for i in range(num_sum)]
        for i in range(k):
            for j in range(num_sam):
                data_bpsk[i*num_sam+j]=np.cos(2*np.pi*fc*(i*num_sum+j)/fs+data[i]*np.pi)
        return data_bpsk
'''
def gauss(data,a,b):                #添加高斯噪声
    k=len(data)
    data_gauss=[0 for i in range(k)]
    for i in range(k):
        data_gauss[i]=data[i]+random.gauss(a,b)    
    return data_gauss
'''
def pulse(data,a,b):               #添加脉冲噪声
    k=len(data)
    data_pulse=[0 for i in range(k)]
    data_pulse=data+levy_stable.rvs(a,b,0,1,k)
    return data_pulse   

def gauss(data,a,b):                #添加窄带高斯噪声
    k=len(data)
    data_gauss=np.random.normal(a,b,k)
    fft=np.fft.fft(data_gauss)
    for i in range(k):      
        if i in range(0,50):
            fft[i]=0
        if i in range(150,650):
            fft[i]=0
        if i in range(750,800):
            fft[i]=0
    data_gauss=np.fft.ifft(fft)
    data_gauss=data+data_gauss
    return data_gauss

for i in range(1000):   
    data_ini[i]= np.random.randint(0,2,19)
    data_de[i][0]=0
    for j in range(1,20):
        if data_ini[i][j-1]==1:
            data_de[i][j]=1-data_de[i][j-1]
        else:
            data_de[i][j]=data_de[i][j-1]
    data_BPSK[i]=BPSK(40,200,1600,data_de[i])   
    #加上噪声
    if noise==0:
        data_noise=data_BPSK
    if noise==1:       
        data_noise[i]=gauss(data_BPSK[i],alpha,beta)
    if noise==2:
        data_noise[i]=pulse(data_BPSK[i],alpha,beta)  
    #for j in range(800):
     #   data_noise[i][j]=data_BPSK[i][j]+random.gauss(0,0.1)
    [S[i],F,T,P]=pylab.specgram(data_noise[i], NFFT=20, Fs=16000, noverlap=10)  #短时傅里叶变换后图像
    #pyplot.imshow(S[i])
      #将数据分成 NFFT长度段，并计算每个部分的频谱\noverlap表示各段之间重叠的采样点数

######test数据
    data_ini_test[i]= np.random.randint(0,2,19)
    data_de_test[i][0]=0
    for j in range(1,20):
        if data_ini_test[i][j-1]==1:
            data_de_test[i][j]=1-data_de_test[i][j-1]
        else:
            data_de_test[i][j]=data_de_test[i][j-1]
    #print(data_ini[i])
    #print(data_de[i])
    data_BPSK_test[i]=BPSK(40,200,1600,data_de_test[i]) ##调制后 
    #加上噪声    
    if noise_test==0:
        data_noise_test=data_BPSK_test
    if noise_test==1:
        data_noise_test[i]=gauss(data_BPSK_test[i],alpha_test,beta_test)
    if noise_test==2:
        data_noise_test[i]=pulse(data_BPSK_test[i],alpha_test,beta_test)
    [S_test[i],F_test,T_test,P_test]=pylab.specgram(data_noise_test[i], NFFT=20, Fs=16000, noverlap=10)  #短时傅里叶变换后图像
    #pyplot.imshow(S[i])
      #将数据分成 NFFT长度段，并计算每个部分的频谱\noverlap表示各段之间重叠的采样点数
s=np.delete(S,0,2)
s=np.delete(s,0,2)
s=np.delete(s,0,2)
s_test=np.delete(S_test,0,2)
s_test=np.delete(s_test,0,2)
s_test=np.delete(s_test,0,2)

x_train=s  
y_train=data_ini
ls=[[0 for i in range(836)]for j in range(1000)]
for i in range(1000):
    ls[i] = [[k[j] for k in x_train[i]] for j in range(len(x_train[i][0]))]
np.save('data_xtrain.npy',ls)
np.save('data_ytrain.npy',y_train)

x_test=s_test 
y_test=data_ini_test
ls_test=[[0 for i in range(836)]for j in range(1000)]
for i in range(1000):
    ls_test[i] = [[k[j] for k in x_test[i]] for j in range(len(x_test[i][0]))]
np.save('data_xtest.npy',ls_test)
np.save('data_ytest.npy',y_test)