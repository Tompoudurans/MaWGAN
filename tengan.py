# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:32:32 2019

@author: Thomas
"""
import tensorflow.keras.layers as tkl
import tensorflow as tf
import numpy as np
import random as rd

class dataGAN():
    #create the genartaeer network
    def make_generator():
        self.generator = tf.keras.models.Sequential()
        self.generator.add(tkl.Dense(100,activation = 'linear'))#,kernel_regularizer=reg.l2(0.1)))#adds one hiden layer with 99 nerons with relu activation fx\n",
        self.generator.add(tkl.Dense(100,activation = 'linear'))#,kernel_regularizer=reg.l2(0.1)))
        self.generator.add(tkl.Dense(100,activation = 'linear'))#,kernel_regularizer=reg.l2(0.1)))
        self.generator.add(tkl.Dense(4,activation = 'softmax'))
        #model.compile(optimizer='adam',loss = 'mse',metrics=['accuracy'])


    #create the dicrinator network
    def make_discriminator():
        self.discriminator = tf.keras.models.Sequential()
        self.discriminator.add(tf.keras.layers.Dense(100,activation = tf.nn.relu))#adds one hiden layer with 128 nerons with relu activation fx\n",
        self.discriminator.add(tf.keras.layers.Dense(100,activation = tf.nn.relu))#adds one hiden layer with 128 nerons with relu activation fx\n",
        self.discriminator.add(tf.keras.layers.Dense(100,activation = tf.nn.relu))
        self.discriminator.add(tf.keras.layers.Dense(2,activation = 'sigmoid'))#,activity_regularizer=reg.l2(0.0001)))#adds ouput layer with 10 nerons with softmax activation fx
        #model.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])

    def noise_vec(vec):
        noise = []
        for i in range(vec):
            small =[]
            for j in range(4):
                small.append(rd.random())
            #noise.append(rd.random())
            noise.append(small)
        return np.array(noise)

    def makedat(gen,real):
        nos = noise(len(real))
        #gen.fit(nos, real, epochs=3)
        x = gen.predict(nos)
        return [gen, x]
# my old code:
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# foster code:
    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def build_adversarial(self):

        ### COMPILE DISCRIMINATOR

        self.discriminator.compile(
        optimizer=self.get_opti(self.discriminator_learning_rate)
        , loss = 'binary_crossentropy'
        ,  metrics = ['accuracy']
        )

        ### COMPILE THE FULL GAN

        self.set_trainable(self.discriminator, False)

        model_input = Input(shape=(self.z_dim,), name='model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = Model(model_input, model_output)

        self.model.compile(optimizer=self.get_opti(self.generator_learning_rate) , loss='binary_crossentropy', metrics=['accuracy'])

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
        noise = self.noise_vec(100) #changed
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

            print ("%d [D loss: (%.3f)(R %.3f, F %.3f)] [D acc: (%.3f)(%.3f, %.3f)] [G loss: %.3f] [G acc: %.3f]" % (epoch, d[0], d[1], d[2], d[3], d[4], d[5], g[0], g[1]))

            self.d_losses.append(d)
            self.g_losses.append(g)

            self.epoch += 1
