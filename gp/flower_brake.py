# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:27:10 2019

@author: user
"""

import random as rd
import numpy as np
import tensorflow as tf
from sklearn import datasets

iris = datasets.load_iris()
idat=iris.data
itar=iris.target

optimizer= tf.keras.optimizers.SGD(learning_rate=0.01)
loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)
  return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


def simplesplit(x,y):
    size=len(x)
    z = np.split(rd.sample(range(size),size),[int(size*(1-10/100))])
    return [x[z[0]],y[z[0]],x[z[1]],y[z[1]]]

def make_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(20,activation = tf.nn.relu,input_shape=(4,)))
    model.add(tf.keras.layers.Dense(20,activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(20,activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(3,activation = tf.nn.softmax))
    return model

def simplefit(train,tar,model,ep=3):
    ## Note: Rerunning this cell uses the same model variable
    
    for epoch in range(ep):
      epoch_loss_avg = tf.keras.metrics.Mean()
      epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
      loss_value, grads = grad(model, train, tar)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      # Track progress
      epoch_loss_avg.update_state(loss_value)  # Add current batch loss
      # Compare predicted label to actual label
      # training=True is needed only if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      epoch_accuracy.update_state(tar, model(train, training=True))

    
      if epoch % 10 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

def testmod(test,actual,model):
    res=model(test)
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
mods.summary()
simplefit(sets[0],sets[1],mods,1000)
print('fiting done')
testmod(sets[2],sets[3],mods)

#mod = tf.keras.models.load_model('name.HDF5')
#mod.save('name.HDF5')
