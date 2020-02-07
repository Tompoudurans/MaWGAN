# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:24:13 2019

@author: user
"""
import seaborn as sns
import pandas as pd
import numpy as np
import random as rd
import torch as tc
import matplotlib.pyplot as mp

def simplesplit(x,y,fac=10):
    size=len(x)
    z = np.split(rd.sample(range(size),size),[int(size*(1-fac/100))])
    return [x[z[0]],y[z[0]],x[z[1]],y[z[1]]]

def xref(bas,h):
    datrain = []
    datest = []
    htrain = []
    htest = []
    size = len(bas)
    samp = rd.sample(range(size),size)
    x = np.split(samp,np.arange(int(size*0.10),size,int(size*0.10)))
    for i in range(10):
        train = []
        test = []
        for j in range(10):
            if i == j:
                test = x[i]
            else:
                train = np.concatenate((train,x[j]),axis=None)
        xtr = []
        for k in range(len(train)):
            xtr.append(int(train[k]))# for some resson i need to turn the idex to interger again
        datrain.append(bas[xtr])
        datest.append(bas[test])
        htrain.append(h[xtr])
        htest.append(h[test])
        print(i)
    return [datrain,htrain,datest,htest]

def testmodel(mod,actual):
    count = 0
    for i in range(len(actual)):
        p=np.argmax(mod[i])
        q=actual[i]
        if p == q:
            print(i,')',p,q,'v')
            count = count + 1
        else:
            print(i,')',p,q,'x')
    print(count,'/',len(actual))

def acctest(mod,actual):
    count = 0
    for i in range(len(actual)):
        p=tc.argmax(mod[i])
        q=actual[i]
        if p == q:
            count = count + 1
    l = len(actual)
    return str(count) + '/'+ str(l)

def plotting(tenso,other):
    tenso = tenso.transpose(1,0)
    other = other.transpose(1,0)
    for x in range(len(tenso)):
        for y in range(len(tenso)):
            if x < y:
                print(x,y)
                mp.scatter(tenso[x],tenso[y])
                mp.scatter(other[x],other[y])
                mp.show()

def dagpolt(x,y):
    fake = pd.DataFrame(x)
    real = pd.DataFrame(y)
    fake['dataset'] = ['fake']*len(x)
    real['dataset'] = ['real']*len(y)
    result = pd.concat([real, fake])
    sns.pairplot(result,hue='dataset')
    mp.show()

def show_loss_progress(loss_discriminator,loss_generator):
        print('discriminator')
        mp.plot(loss_discriminator)
        mp.show()
        print('generator')
        mp.plot(loss_generator)
        mp.show()
