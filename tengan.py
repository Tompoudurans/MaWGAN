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
    def __init__(self,optimiser,noise_dim,data_dim,net_dim,number_of_layers):
        """
        This builds the GAN model so it can be trained. The different variables are:
        'optimiser' which is the optimiser used for the whole GAN
        noise_dim which is the length of the noise vector that is used to generate data,
        data_dim which is the number of fields in the database,
        net_dim which is the number of neurons per layer.
        """
        self.optimiser = optimiser
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        self.net_dim = net_dim
        self.d_losses = []
        self.g_losses = []
        self.epoch = 0
        self.make_discriminator(number_of_layers)
        self.make_generator(number_of_layers)
        self.build_adversarial()

    def make_generator(self,number_of_layers):
        """
        This makes a generator network with 'number_of_layers' layers and 'net_dim' of nodes per layer.
        It takes in a vector of 'noise_dim' length and outputs a vector of data that is 'data_dim' long.
        """
        gen_input = tkl.Input(shape=(self.noise_dim,),name='gen_input')
        gen_layer =tkl.Dense(self.net_dim,activation = 'tanh')(gen_input)
        number_of_layers -= 2
        while number_of_layers > 1:
            gen_layer =tkl.Dense(self.net_dim,activation = 'tanh')(gen_layer)
            number_of_layers -= 1
        final_layer =tkl.Dense(self.data_dim,activation = 'linear')(gen_layer)
        self.generator = tf.keras.models.Model(inputs=gen_input,outputs=final_layer)

    def make_discriminator(self,number_of_layers):
        """
        This makes a discriminator network with 'number_of_layers' layers and 'net_dim' of nodes per layer.
        It takes in a vector of data that is 'data_dim' long and outputs a probability of the data being real or fake.
        """
        dis_input = tkl.Input(shape=(self.data_dim,),name='dis_input')
        dis_layer =tkl.Dense(self.net_dim,activation = tf.nn.relu)(dis_input)
        number_of_layers -= 2
        while number_of_layers > 1:
            dis_layer =tkl.Dense(self.net_dim,activation = 'tanh')(dis_layer)
            number_of_layers -= 1
        final_layer =tkl.Dense(1,activation = 'sigmoid')(dis_layer)
        self.discriminator = tf.keras.models.Model(inputs=dis_input,outputs=final_layer)

    def set_trainable(self, m, val):
        """
        This freezes and unfreezes weights depending on the value of 'val'
        """
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def build_adversarial(self):
        """
        This compiles the discriminator and
        then compiles a GAN model that is used for training the generator
        it consists of the generator directly outputed into a frozen discriminator
        """

        self.discriminator.compile(optimizer=self.optimiser,
                                   loss = 'binary_crossentropy',
                                   metrics = ['accuracy'])
        #temporarily freezes the discriminator weight so it does not affect the discriminator network
        self.set_trainable(self.discriminator, False)
        #creating the GAN model
        model_input = tkl.Input(shape=(self.noise_dim,), name='model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = Model(model_input, model_output)
        self.model.compile(optimizer=self.optimiser , loss='binary_crossentropy', metrics=['accuracy'])
        #Unfreezes the weights
        self.set_trainable(self.discriminator, True)

    def train_discriminator(self, real_data, batch_size):
        """
        This trains the discriminator once by creating a set of fake_data and
        traning them aganist the real_data
        """
        # create the labels
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))
        #create noise vector z
        noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
        fake_data = self.generator.predict(noise)
        #calculate value function for real
        d_loss_real, d_acc_real =   self.discriminator.train_on_batch(real_data, valid)
        #calculate value function for fake
        d_loss_fake, d_acc_fake =   self.discriminator.train_on_batch(fake_data, fake)
        d_loss =  0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)
        return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]

    def train_generator(self, batch_size):
        """
        This trains the generator once by creating a set of fake data and
        uses the dicrimator score to train on
        """
        valid = np.ones((batch_size,1))
        noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
        return self.model.train_on_batch(noise, valid)


    def train(self,
              x_train,
              batch_size,
              epochs,
              every_n_batches=50):
        """
        This trains the GAN by alternating between training the discriminator and training the generator once
        in each epoch on the dataset x_train which has a length of batch_size.
        It will print and record the loss of the generator and discriminator every_n_batches.
        """
        for epoch in range(epochs):
            d = self.train_discriminator(x_train, batch_size)
            g = self.train_generator(batch_size)
            if epoch % every_n_batches == 0:
                print ("%d [D loss: (%.3f)(R %.3f, F %.3f)] [D acc: (%.3f)(%.3f, %.3f)] [G loss: %.3f] [G acc: %.3f]" % (epoch, d[0], d[1], d[2], d[3], d[4], d[5], g[0], g[1]))
                self.d_losses.append(d)
                self.g_losses.append(g)
            self.epoch += 1

    def save_model(self,filepath):
        """
        This saves the weights of the three models that are used in the GAN on the 'filepath'.
        """
        self.model.save(filepath + 'model.h5')
        self.discriminator.save(filepath + 'discriminator.h5')
        self.generator.save(filepath + 'generator.h5')

    def load_weights(self, filepath):
        """
        This loads the weights of the three models that are used in the GAN on the 'filepath'.
        """
        self.model.load_weights(filepath + 'model.h5')
        self.discriminator.load_weights(filepath + 'discriminator.h5')
        self.generator.load_weights(filepath + 'generator.h5')
