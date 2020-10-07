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


class wGANgp:
    def __init__(self, optimiser, z_dim, data_dim, net_dim, number_of_layers, lamabda):
        """
        This builds the GAN model so it can be trained. The different variables are:
        'optimiser' which is the optimiser used for the whole GAN
        z_dim which is the length of the noise vector that is used to generate data,
        data_dim which is the number of fields in the database,
        net_dim which is the number of neurons per layer.
        """
        self.net_dim = net_dim
        self.data_dim = data_dim
        self.d_losses = []
        self.g_losses = []
        self.epoch = 0
        self.optimiser = optimiser
        self.z_dim = z_dim
        self.make_critc(number_of_layers)
        self.make_generator(number_of_layers)
        self.lamabda = float(lamabda)
        self.bypass = np.array([[[2] * self.data_dim], [[1] * self.data_dim]])
        self.build_adversarial()

    def wasserstein_critic(self, fake, real):
        wasserstein = K.mean(fake) - K.mean(real) + self.gradient_penalty()
        return wasserstein

    def generator_loss(self, fake, ones):
        # = self.critic(fake) - self.critic(fake)
        predict = K.mean(-self.critic(fake)) + K.mean(ones) - K.mean(ones)
        return predict

    def gradient_penalty(self):
        eta = np.random.rand()
        interprated_data = self.bypass[0] * eta + self.bypass[1] * (1 - eta)
        zero = np.zeros((self.z_dim, 1), dtype=np.float32)
        gradients = tf.gradients(
            zero, self.critic(interprated_data), unconnected_gradients="zero"
        )
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        # summing over the rows
        gradients_sqr_sum = K.sum(
            gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape))
        )
        # and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = self.lamabda * K.square(1 - gradient_l2_norm)
        return gradient_penalty

    def make_generator(self, number_of_layers):
        """
        This makes a generator network with 'number_of_layers' layers and 'net_dim' of nodes per layer.
        It takes in a vector of 'z_dim' length and outputs a vector of data that is 'data_dim' long.
        """
        gen_input = tkl.Input(shape=(self.z_dim,), name="gen_input")
        gen_layer = tkl.Dense(self.net_dim, activation="tanh")(gen_input)
        number_of_layers -= 2
        while number_of_layers > 1:
            gen_layer = tkl.Dense(self.net_dim, activation="tanh")(gen_layer)
            number_of_layers -= 1
        final_layer = tkl.Dense(self.data_dim, activation="linear")(gen_layer)
        self.generator = tf.keras.models.Model(inputs=gen_input, outputs=final_layer)

    def make_critc(self, number_of_layers):
        """
        This makes a critic network with 'number_of_layers' layers and 'net_dim' of nodes per layer.
        It takes in a vector of data that is 'data_dim' long and outputs a probability of the data being real or fake.
        """
        dis_input = tkl.Input(shape=(self.data_dim,), name="dis_input")
        dis_layer = tkl.Dense(self.net_dim, activation=tf.nn.relu)(dis_input)
        number_of_layers -= 2
        while number_of_layers > 1:
            dis_layer = tkl.Dense(self.net_dim, activation="tanh")(dis_layer)
            number_of_layers -= 1
        final_layer = tkl.Dense(1, activation="sigmoid")(dis_layer)
        self.critic = tf.keras.models.Model(inputs=dis_input, outputs=final_layer)

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
        self.critic.compile(optimizer=self.optimiser, loss=self.wasserstein_critic)
        self.generator.compile(optimizer=self.optimiser, loss=self.generator_loss)

    def train_critic(self, x_train, batch_size):
        """
        This trains the critc once by creating a set of fake_data and
        traning them aganist the real_data, then the wieght are cliped
        """
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        true_imgs = x_train[idx]
        # create noise vector z
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)
        self.bypass = [gen_imgs, true_imgs]
        predict_true = self.critic.predict(true_imgs)
        d_loss = self.critic.train_on_batch(gen_imgs, predict_true)
        return d_loss

    def train_generator(self, batch_size):
        """
        This trains the generator once by creating a set of fake data and
        uses the critic score to train on
        """
        # idx = np.random.randint(0, x_train.shape[0], batch_size)
        # true_imgs = x_train[idx]
        lable = np.ones((batch_size, self.data_dim), dtype=np.float32)
        # create noise vector z
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.generator.train_on_batch(noise, lable)

    def train(
        self, x_train, batch_size, epochs, print_every_n_batches=50, critic_round=5
    ):
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
                print("%d [D loss: %.3f] [G loss: %.3f]" % (epoch, d, g))
                self.d_losses.append(d)
                self.g_losses.append(g)
            self.epoch += 1

    def create_fake(self, batch_size):
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        fake_data = self.generator.predict(noise)
        return fake_data

    def save_model(self, f):
        """
        This saves the weights of the three models that are used in the GAN on the 'filepath'.
        """
        self.critic.save(f + "_critic.h5")
        self.generator.save(f + "_generator.h5")

    def load_weights(self, filepath):
        """
        This loads the weights of the three models that are used in the GAN on the 'filepath'.
        """
        self.critic.load_weights(filepath + "_critic.h5")
        self.generator.load_weights(filepath + "_generator.h5")
