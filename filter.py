# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 19:58:12 2018

@author: 呀呀呀呀呀可乐
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
from scipy.stats import levy_stable

Rb=40
fc=200
fs=16000
##可以得到fft频谱图峰值处k=fc*N/fs=fc*num_sum/fs   此处应等于10000
##每个码元抽样的数据的个数为n=num_sum/len(data)  此处应等于400
##高斯噪声的功率为Ni=sigma^2   输入信号的功率为Si=sqrt(2)/2
noise=1      #noise=0 选择脉冲噪声   noise=1选择高斯噪声
mean=0
variance=5  ###高斯噪声的均值和方差


def BPSK(Rb,fc,fs,data):           #Rb为码元速率、fc为载波频率、fs为采样频率
        k=len(data)       #码元总个数
        num_sam=int(fs/Rb)     #每个码元需要采样的个数
        num_sum=k*num_sam      #需要的总的采样数
        data_bpsk=[0 for i in range(num_sum)]
        for i in range(k):
            for j in range(num_sam):
                data_bpsk[i*num_sam+j]=np.cos(2*np.pi*fc*(i*num_sum+j)/fs+data[i]*np.pi)
        return data_bpsk

def limiter(a):           #限幅器
    if a>1:
        return 1
    if a<-1:
        return -1
    else:
        return a
    
def judge(a):           #判断是0还是1
    if a>0.5:
        return 1
    if a<=0.5:
        return 0
'''
def gauss(data,a,b):            #添加高斯噪声
    k=len(data)
    data_gauss=[0 for i in range(k)]
    for i in range(k):
        data_gauss[i]=data[i]+random.gauss(a,b)    
    return data_gauss
'''
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


def pulse(data,a,b):
    k=len(data)
    data_pulse=[0 for i in range(k)]
    for i in range(k):
        data_pulse[i]=np.float32(data[i]+levy_stable.rvs(a,b,0,1,1))
    return data_pulse

data_ini= np.random.randint(0,2,20000)
k=len(data_ini)
num_sum=k*int(fs/Rb)

data_noise=[0 for i in range(num_sum)]
car_sig=[0 for i in range(num_sum)]
sig_demodule=[0 for i in range(num_sum)]
data_BPSK=BPSK(Rb,fc,fs,data_ini) 
result=np.array([0 for i in range(k)])

if noise==1:
    data_noise=gauss(data_BPSK,mean,variance)             #加上高斯噪声
    
if noise==0:
    data_noise=pulse(data_BPSK,mean,variance)             #加上脉冲噪声



for j in range(num_sum):
        car_sig[j]=np.cos(2*np.pi*fc*j/fs+np.pi)                  #同频载波
        sig_demodule[j]=limiter(data_noise[j]*car_sig[j])         #乘以同频载波
       

BPSK_fft=np.fft.fft(data_BPSK)
noise_fft=np.fft.fft(data_noise)
demodule_fft=np.fft.fft(sig_demodule)

for i in range(120000,7880000):     #低通滤波器
    demodule_fft[i]=0
    
demodule_result=np.fft.ifft(demodule_fft)
demodule_result=demodule_result+0.5

for i in range(20000):                     #       取平均 做滤波
    sum=0
    for j in range(1,4):
        sum=sum+demodule_result[i*400+j*100]  #  抽样判决的点数
    result[i]=judge(sum/3)
num=0

for i in range(20000):                #结果检验
    if result[i]==data_ini[i]:
        num=num+1
        
acc=num/20000
print(acc)       

'''
plt.subplot(411)
plt.plot(data_BPSK)

plt.subplot(412)
plt.plot(demodule_fft)

plt.subplot(413)
plt.plot(demodule_result)

plt.subplot(414)
plt.plot(sig_demodule)
'''
