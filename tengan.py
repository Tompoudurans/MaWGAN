# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:32:32 2019

@author: Thomas
"""
from tensorflow.keras.models import Model#, Sequential
#from keras import backend as K
import tensorflow.keras.layers as tkl
import tensorflow as tf
import numpy as np
import random as rd

class dataGAN():
    def __init__(self,optimiser,z_dim,data_dim,net_dim):
        #discriminator_learning_rate,
        #generator_learning_rate,
        #self.discriminator_learning_rate = discriminator_learning_rate
        #self.generator_learning_rate = generator_learning_rate
        #self.inputs = inputs
        self.net_dim = net_dim
        self.data_dim =data_dim
        self.d_losses = []
        self.g_losses = []
        self.epoch = 0
        self.optimiser = optimiser
        self.z_dim = z_dim
        self.make_discriminator()
        self.make_generator()
        self.build_adversarial()

    #create the genartaeer network
    def make_generator(self):
        gen_input = tkl.Input(shape=(self.z_dim,),name='gen_input')
        a =tkl.Dense(self.net_dim,activation = 'tanh')(gen_input)
        b =tkl.Dense(self.net_dim,activation = 'tanh')(a)
        c =tkl.Dense(self.net_dim,activation = 'tanh')(b)
        d =tkl.Dense(self.data_dim,activation = 'linear')(c)#,kernel_regularizer=reg.l2(0.1)))
        self.generator = tf.keras.models.Model(inputs=gen_input,outputs=d)
        #self.generator.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])


    #create the dicrinator network
    def make_discriminator(self):
        dis_input = tkl.Input(shape=(self.data_dim,),name='dis_input')
        a =tkl.Dense(self.net_dim,activation = tf.nn.relu)(dis_input)
        b =tkl.Dense(self.net_dim,activation = tf.nn.relu)(a)
        c =tkl.Dense(self.net_dim,activation = tf.nn.relu)(b)
        d =tkl.Dense(1,activation = 'sigmoid')(b)
        self.discriminator = tf.keras.models.Model(inputs=dis_input,outputs=d)
        #,activity_regularizer=reg.l2(0.0001)))#adds ouput layer with 10 nerons with softmax activation fx
        #model.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])

    def noise_vec(self,vec):
        noise = []
        for i in range(vec):
            noise.append(rd.random())
        noise = np.array(noise)
        noise = noise.reshape(1,100)
        return noise

  #  def makedat(gen,real):
        #nos = noise_vec(len(real))
        #gen.fit(nos, real, epochs=3)
  #      x = gen.predict(nos)
   #     return [gen, x]
# my old code:
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# foster code:
#    def get_opti(self, lr):
#        if self.optimiser == 'adam':
#            opti = Adam(lr=lr, beta_1=0.5)
#        elif self.optimiser == 'rmsprop':
#            opti = RMSprop(lr=lr)
 #       else:
 #           opti = Adam(lr=lr)


    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def build_adversarial(self):

        ### COMPILE DISCRIMINATOR

        self.discriminator.compile(
        optimizer=self.optimiser
        , loss = 'binary_crossentropy'
        ,  metrics = ['accuracy']
        )

        ### COMPILE THE FULL GAN

        self.set_trainable(self.discriminator, False)
        #-------------------------------------------------
        model_input = tkl.Input(shape=(self.z_dim,), name='model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = Model(model_input, model_output)
        self.model.compile(optimizer=self.optimiser , loss='binary_crossentropy', metrics=['accuracy'])
        self.set_trainable(self.discriminator, True)




    def train_discriminator(self, x_train, batch_size, using_generator):

        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

    #---------------------------------------------
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
    #-----------------------------------------------
        gen_imgs = self.generator.predict(noise)
        d_loss_real, d_acc_real =   self.discriminator.train_on_batch(true_imgs, valid)
        d_loss_fake, d_acc_fake =   self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss =  0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]

    def train_generator(self, batch_size):
        valid = np.ones((batch_size,1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)


    def train(self, x_train, batch_size, epochs
    , print_every_n_batches = 50
    , using_generator = False):

#-----------
        for epoch in range(epochs):
#------------
            d = self.train_discriminator(x_train, batch_size, using_generator)
            g = self.train_generator(batch_size)

            if epoch % print_every_n_batches == 0:
                print ("%d [D loss: (%.3f)(R %.3f, F %.3f)] [D acc: (%.3f)(%.3f, %.3f)] [G loss: %.3f] [G acc: %.3f]" % (epoch, d[0], d[1], d[2], d[3], d[4], d[5], g[0], g[1]))

            self.d_losses.append(d)
            self.g_losses.append(g)

            self.epoch += 1
