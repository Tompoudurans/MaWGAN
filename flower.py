# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:27:10 2019

@author: user
"""
import random as rd
from time import time
import numpy as np
import tensorflow as tf
from sklearn import datasets

iris = datasets.load_iris()
idat=iris.data
itar=iris.target


def simplesplit(x,y):
    size=len(x)
    z = np.split(rd.sample(range(size),size),[int(size*(1-10/100))])
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


def make_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(33,activation = tf.nn.relu))#,kernel_regularizer=reg.l2(0.1)))#adds one hiden layer with 99 nerons with relu activation fx\n",
    model.add(tf.keras.layers.Dense(33,activation = tf.nn.relu))#,kernel_regularizer=reg.l2(0.1)))
    model.add(tf.keras.layers.Dense(33,activation = tf.nn.relu))#,kernel_regularizer=reg.l2(0.1)))
    model.add(tf.keras.layers.Dense(3,activation = tf.nn.softmax))#,activity_regularizer=reg.l2(0.1)))#adds ouput layer with 10 nerons with softmax activation fx
    model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

def simplefit(train,tar,mod,ep=3):
    mod.fit(train, tar, epochs=ep,callbacks=[tensorboard])
    return mod

def testmod(test,actual,model):
    res=model.predict(test)
    model.evaluate(test,actual)
    for i in range(len(actual)):
        p=np.argmax(res[i])
        q=actual[i]
        if p == q:
            print(i,')',p,q,'v')
        else:
            print(i,')',p,q,'x')

def complexfit(mod,batch):
    for i in range(10):
        #mod.train_on_batch(batch[0][i],batch[1][i])
        mod.fit(batch[0][i], batch[1][i], epochs=3)
        print(mod.test_on_batch(batch[2][i],batch[3][i]))
    return mod
