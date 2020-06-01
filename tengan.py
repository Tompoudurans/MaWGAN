# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:32:32 2019

@author: Thomas
"""
from tensorflow.keras.models import Model
import tensorflow.keras.layers as tkl
import tensorflow as tf
import numpy as np
import random as rd

class dataGAN():
    def __init__(self,optimiser,z_dim,data_dim,net_dim,number_of_layers):
        """
        This bulid the gan model so it can be trained, the differrent varible are
        net_dim is number of nuerons per layer,
        data_dim is the number of flied in the database,
        z_dim is the length of noise vector that is used to gentrate data
        and optimiser is the optimiser used for the whole gan.
        """
        self.net_dim = net_dim
        self.data_dim = data_dim
        self.d_losses = []
        self.g_losses = []
        self.epoch = 0
        self.optimiser = optimiser
        self.z_dim = z_dim
        self.make_discriminator(number_of_layers)
        self.make_generator(number_of_layers)
        self.build_adversarial()

    #create the genartaeer network
    def make_generator(self,no_of_layer):
        """
        makes a generator network with 'no_of_layer' of layers and 'net_dim' of nodes per layer
        it takes in a vector of 'z_dim' lenght and output a vector of data that is 'data_dim' long
        """
        gen_input = tkl.Input(shape=(self.z_dim,),name='gen_input')
        gen_layer =tkl.Dense(self.net_dim,activation = 'tanh')(gen_input)
        no_of_layer -= 2
        while no_of_layer > 1:
            gen_layer =tkl.Dense(self.net_dim,activation = 'tanh')(gen_layer)
            no_of_layer -= 1
        final_layer =tkl.Dense(self.data_dim,activation = 'linear')(gen_layer)#,kernel_regularizer=reg.l2(0.1)))
        self.generator = tf.keras.models.Model(inputs=gen_input,outputs=final_layer)

    #create the dicrinator network
    def make_discriminator(self,no_of_layer):
        """
        makes a discriminator network with 'no_of_layer' of layers and 'net_dim' of nodes per layer
        it takes in a vector of data that is 'data_dim' long and output a probality of the data being real or fake
        """
        dis_input = tkl.Input(shape=(self.data_dim,),name='dis_input')
        dis_layer =tkl.Dense(self.net_dim,activation = tf.nn.relu)(dis_input)
        no_of_layer -= 2
        while no_of_layer > 1:
            dis_layer =tkl.Dense(self.net_dim,activation = 'tanh')(dis_layer)
            no_of_layer -= 1
        final_layer =tkl.Dense(1,activation = 'sigmoid')(dis_layer)
        self.discriminator = tf.keras.models.Model(inputs=dis_input,outputs=final_layer)

    def set_trainable(self, m, val):
        """
        freeze and unfrezes weights
        """
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def build_adversarial(self):
        """complie the discriminator and
        then compile a GAN model that is used that is used for tainning the generator
        it conciste of the generator directly outputed into a frozen discriminator. """

        self.discriminator.compile(
        optimizer=self.optimiser
        , loss = 'binary_crossentropy'
        ,  metrics = ['accuracy']
        )

        self.set_trainable(self.discriminator, False)#temp freeze the dis wight so it does not efect the dis network
        #-------------------------------------------------
        model_input = tkl.Input(shape=(self.z_dim,), name='model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = Model(model_input, model_output)
        self.model.compile(optimizer=self.optimiser , loss='binary_crossentropy', metrics=['accuracy'])
        self.set_trainable(self.discriminator, True)

    def train_discriminator(self, true_imgs, batch_size):
        """this train discriminator once """
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))
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
            d = self.train_discriminator(x_train, batch_size)
            g = self.train_generator(batch_size)

            if epoch % print_every_n_batches == 0:
                print ("%d [D loss: (%.3f)(R %.3f, F %.3f)] [D acc: (%.3f)(%.3f, %.3f)] [G loss: %.3f] [G acc: %.3f]" % (epoch, d[0], d[1], d[2], d[3], d[4], d[5], g[0], g[1]))
                self.d_losses.append(d)
                self.g_losses.append(g)
                #self.save_model()

            self.epoch += 1

    def save_model(self,filepath):
        self.model.save(filepath + 'model.h5')
        self.discriminator.save(filepath + 'discriminator.h5')
        self.generator.save(filepath + 'generator.h5')

    def load_weights(self, filepath):
        self.model.load_weights(filepath + 'model.h5')
        self.discriminator.load_weights(filepath + 'discriminator.h5')
        self.generator.load_weights(filepath + 'generator.h5')
