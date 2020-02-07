# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:32:32 2019

@author: Thomas
"""
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow.keras.layers as tkl
import tensorflow as tf
import numpy as np
import random as rd

class dataGAN():
    def __init__(self,optimiser,z_dim,data_dim,net_dim):
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


    def wasserstein(self, y_true, y_pred):
        return -K.mean(y_true * y_pred)

    #create the genartaeer network
    def make_generator(self):
        gen_input = tkl.Input(shape=(self.z_dim,),name='gen_input')
        a =tkl.Dense(self.net_dim,activation = 'tanh')(gen_input)
        b =tkl.Dense(self.net_dim,activation = 'tanh')(a)
        c =tkl.Dense(self.net_dim,activation = 'tanh')(b)
        d =tkl.Dense(self.data_dim,activation = 'linear')(c)#,kernel_regularizer=reg.l2(0.1)))
        self.generator = tf.keras.models.Model(inputs=gen_input,outputs=d)

    #create the dicrinator network
    def make_discriminator(self):
        dis_input = tkl.Input(shape=(self.data_dim,),name='dis_input')
        a =tkl.Dense(self.net_dim,activation = tf.nn.relu)(dis_input)
        b =tkl.Dense(self.net_dim,activation = tf.nn.relu)(a)
        c =tkl.Dense(self.net_dim,activation = tf.nn.relu)(b)
        d =tkl.Dense(1,activation = None)(b)
        self.discriminator = tf.keras.models.Model(inputs=dis_input,outputs=d)

    def gradient_penalty_loss(self, y_true, y_pred, interpolated_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, interpolated_samples)[0]

        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def build_adversarial(self):

        ### COMPILE DISCRIMINATOR

        self.discriminator.compile(
        optimizer=self.optimiser
        , loss =self.wasserstein
        )

        ### COMPILE THE FULL GAN

        self.set_trainable(self.discriminator, False)#temp freeze the dis wight so it does not efect the dis network
        #-------------------------------------------------
        model_input = tkl.Input(shape=(self.z_dim,), name='model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = Model(model_input, model_output)

        self.model.compile(optimizer=self.optimiser, loss=self.wasserstein)
        self.set_trainable(self.discriminator, True)

    def train_discriminator(self, x_train, batch_size, using_generator):

        clip_threshold = 0.1
        valid = np.ones((batch_size,1))
        fake = -np.ones((batch_size,1))

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
        d_loss_real =   self.discriminator.train_on_batch(true_imgs, valid)
        d_loss_fake =   self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss =  0.5 * (d_loss_real + d_loss_fake)

        for l in self.discriminator.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -clip_threshold, clip_threshold) for w in weights]
            l.set_weights(weights)


        return [d_loss, d_loss_real, d_loss_fake]

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
            for n in range(5):
                d = self.train_discriminator(x_train, batch_size, using_generator)
            g = self.train_generator(batch_size)

            if epoch % print_every_n_batches == 0:
                print ("%d [D loss: (%.3f)(R %.3f, F %.3f)] [G loss: %.3f]" % (epoch, d[0], d[1], d[2], g))
                self.d_losses.append(d)
                self.g_losses.append(g)

            self.epoch += 1

    def save_model(self):
        self.model.save('Wgan_model.h5')
        self.discriminator.save('Wgan_discriminator.h5')
        self.generator.save('Wgan_generator.h5')
