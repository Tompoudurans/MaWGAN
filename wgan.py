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

class wGAN():
    def __init__(self,optimiser,z_dim,data_dim,net_dim,number_of_layers,weight_cliping):
        """
        This builds the GAN model so it can be trained. The different variables are:
        'optimiser' which is the optimiser used for the whole GAN
        z_dim which is the length of the noise vector that is used to generate data,
        data_dim which is the number of fields in the database,
        net_dim which is the number of neurons per layer.
        """
        self.net_dim = net_dim
        self.data_dim =data_dim
        self.d_losses = []
        self.g_losses = []
        self.epoch = 0
        self.clip = float(weight_cliping)
        self.optimiser = optimiser
        self.z_dim = z_dim
        self.make_critc(number_of_layers)
        self.make_generator(number_of_layers)
        self.build_adversarial()

    def wasserstein(self, y_true, y_pred):
        """
        calcuate half the wasserstein distance
        """
        muti = y_true * y_pred
        s = K.sum(muti)
        return (s/self.z_dim)
        #return -K.mean(y_true * y_pred)

    def wasserstein_critic(self, fake, real):
        return K.mean(fake)-K.mean(real)


    def make_generator(self,number_of_layers):
        """
        This makes a generator network with 'number_of_layers' layers and 'net_dim' of nodes per layer.
        It takes in a vector of 'z_dim' length and outputs a vector of data that is 'data_dim' long.
        """
        gen_input = tkl.Input(shape=(self.z_dim,),name='gen_input')
        gen_layer =tkl.Dense(self.net_dim,activation = 'tanh')(gen_input)
        number_of_layers -= 2
        while number_of_layers > 1:
            gen_layer =tkl.Dense(self.net_dim,activation = 'tanh')(gen_layer)
            number_of_layers -= 1
        final_layer =tkl.Dense(self.data_dim,activation = 'linear')(gen_layer)
        self.generator = tf.keras.models.Model(inputs=gen_input,outputs=final_layer)


    def make_critc(self,number_of_layers):
        """
        This makes a critic network with 'number_of_layers' layers and 'net_dim' of nodes per layer.
        It takes in a vector of data that is 'data_dim' long and outputs a probability of the data being real or fake.
        """
        dis_input = tkl.Input(shape=(self.data_dim,),name='dis_input')
        dis_layer =tkl.Dense(self.net_dim,activation = tf.nn.relu)(dis_input)
        number_of_layers -= 2
        while number_of_layers > 1:
            dis_layer =tkl.Dense(self.net_dim,activation = 'tanh')(dis_layer)
            number_of_layers -= 1
        final_layer =tkl.Dense(1,activation = 'sigmoid')(dis_layer)
        self.critic = tf.keras.models.Model(inputs=dis_input,outputs=final_layer)

    def set_trainable(self, m, val):
        """
        This freezes and unfreezes weights depending on the value of 'val'
        """
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def build_adversarial(self):
        """
        This compiles the critic and
        then compiles a GAN model that is used for training the generator
        it consists of the generator directly outputed into a frozen critic
        """
        self.critic.compile(
        optimizer=self.optimiser
        , loss =self.wasserstein
        )

        #temporarily freezes the critic weight so it does not affect the critic network
        self.set_trainable(self.critic, False)
        #creating the GAN model
        model_input = tkl.Input(shape=(self.z_dim,), name='model_input')
        model_output = self.critic(self.generator(model_input))
        self.model = Model(model_input, model_output)
        #Unfreezes the weights
        self.model.compile(optimizer=self.optimiser, loss=self.wasserstein)
        self.set_trainable(self.critic, True)

    def train_critic(self, x_train, batch_size):
        """
        This trains the critc once by creating a set of fake_data and
        traning them aganist the real_data, then the wieght are cliped
        """
        clip_threshold = self.clip
        # create the labels
        valid = np.ones((batch_size,1))
        fake = -np.ones((batch_size,1))
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        true_imgs = x_train[idx]
        #create noise vector z
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)
        d_loss_real =   self.critic.train_on_batch(true_imgs, valid)
        d_loss_fake =   self.critic.train_on_batch(gen_imgs, fake)
        d_loss =  0.5 * (d_loss_real + d_loss_fake)
        # clip the weights
        for l in self.critic.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -clip_threshold, clip_threshold) for w in weights]
            l.set_weights(weights)

        return [d_loss, d_loss_real, d_loss_fake]

    def train_generator(self, batch_size):
        """
        This trains the generator once by creating a set of fake data and
        uses the critic score to train on
        """
        valid = np.ones((batch_size,1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)


    def train(self, x_train, batch_size, epochs, print_every_n_batches = 50,critic_round = 5):
        """
        This trains the GAN by alternating between training the critic 'critic_round' times
        and training the generator once in each epoch on
        the dataset x_train which has a length of batch_size.
        It will print and record the loss of the generator and critic every_n_batches.
        """
        for epoch in range(epochs):
            for n in range(critic_round):
                d = self.train_critic(x_train, batch_size)
            g = self.train_generator(batch_size)

            if epoch % print_every_n_batches == 0:
                print ("%d [D loss: (%.3f)(R %.3f, F %.3f)] [G loss: %.3f]" % (epoch, d[0], d[1], d[2], g))
                self.d_losses.append(d)
                self.g_losses.append(g)
            self.epoch += 1

    def create_fake(self, batch_size):
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        fake_data = self.generator.predict(noise)
        return fake_data

    def save_model(self,f):
        """
        This saves the weights of the three models that are used in the GAN on the 'filepath'.
        """
        self.model.save(f + '_model.h5')
        self.critic.save(f + '_critic.h5')
        self.generator.save(f + '_generator.h5')

    def load_weights(self, filepath):
        """
        This loads the weights of the three models that are used in the GAN on the 'filepath'.
        """
        self.model.load_weights(filepath + "_model.h5")
        self.critic.load_weights(filepath + "_critic.h5")
        self.generator.load_weights(filepath + "_generator.h5")
