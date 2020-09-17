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

def make_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(33,activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(33,activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(33,activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(3,activation = tf.nn.softmax))
    return model

def simplefit(train,tar,mod,ep=3):
    mod.fit(train, tar, epochs=ep)
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


sets =  simplesplit(idat,itar)
print('seperating done')
mods = make_model()
print('model done')
truemod = simplefit(sets[0],sets[1],mods,15)
print('fiting done')
truemod.summary()
testmod(sets[2],sets[3],truemod)

#mod = tf.keras.models.load_model('name.HDF5')
#mod.save('name.HDF5')
