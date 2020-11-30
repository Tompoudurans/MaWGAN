from keras.layers import *

from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras.initializers import RandomNormal
from functools import partial
import numpy as np
import json
import os
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

import logging


class RandomWeightedAverage(tf.keras.layers.Layer):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        alpha = tf.random.uniform((self.batch_size, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class wGANgp:
    def __init__(
        self,
        optimiser,
        input_dim,
        noise_size,
        batch_size,
        number_of_layers,
        lambdas,
        learning_rate,
    ):
        # tf.keras.backend.set_floatx('float32')
        self.name = "gan"
        self.input_dim = input_dim
        self.critic_learning_rate = learning_rate

        self.net_dim = noise_size
        self.generator_learning_rate = learning_rate
        self.lambdas = lambdas
        self.optimiser = optimiser
        self.batch_size = batch_size

        self.d_losses = []
        self.g_losses = []
        self.epoch = 0

        self.make_critc(number_of_layers)
        self.make_generator(number_of_layers)
        self.d_optimizer = self.get_opti(learning_rate)
        self.g_optimizer = self.get_opti(learning_rate)
        #self._build_adversarial()
        
    def critic_loss(self,real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss


    # Define the loss functions to be used for generator
    def generator_loss(self,fake_img):
        return -tf.reduce_mean(fake_img)


    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.critic(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def wasserstein(self, y_true, y_pred):
        """
        calcuate half the wasserstein distance
        """
        return -K.mean(y_true * y_pred)

    def make_generator(self, number_of_layers):
        """
        This makes a generator network with 'number_of_layers' layers and 'net_dim' of nodes per layer.
        It takes in a vector of 'batch_size' length and outputs a vector of data that is 'data_dim' long.
        """
        gen_input = Input(shape=(self.batch_size,), name="gen_input")
        gen_layer = Dense(self.net_dim, activation="tanh")(gen_input)
        number_of_layers -= 2
        while number_of_layers > 1:
            gen_layer = Dense(self.net_dim, activation="tanh")(gen_layer)
            number_of_layers -= 1
        final_layer = Dense(self.input_dim, activation="linear")(gen_layer)
        self.generator = Model(inputs=gen_input, outputs=final_layer)

    def make_critc(self, number_of_layers):
        """
        This makes a critic network with 'number_of_layers' layers and 'net_dim' of nodes per layer.
        It takes in a vector of data that is 'data_dim' long and outputs a probability of the data being real or fake.
        """
        dis_input = Input(shape=(self.input_dim,), name="dis_input")
        dis_layer = Dense(self.net_dim, activation="tanh")(dis_input)
        number_of_layers -= 2
        while number_of_layers > 1:
            dis_layer = Dense(self.net_dim, activation="tanh")(dis_layer)
            number_of_layers -= 1
        final_layer = Dense(1, activation="sigmoid")(dis_layer)
        self.critic = Model(inputs=dis_input, outputs=final_layer)

    def get_opti(self, lr):
        """
        sets up optimisers to the network
        """
        if self.optimiser == "adam":
            opti = Adam(lr=lr, beta_1=0.5)
        elif self.optimiser == "rmsprop":
            opti = RMSprop(lr=lr)
        else:
            raise ValueError("Unknown optimizer")
        return opti

    def set_trainable(self, m, val):
        """
        This freezes and unfreezes weights depending on the value of 'val'
        """
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def create_fake(self, batch_size):
        """
        this creates a batch of fake data
        """
        noise = np.random.normal(0, 1, (batch_size, self.batch_size))
        fake_data = self.generator.predict(noise)
        return fake_data

    def train_step(self, real_images, d_steps, batch_size):
        #if isinstance(real_images, tuple):
        #    real_images = real_images[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper.
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add gradient penalty to the discriminator loss
        # 6. Return generator and discriminator losses as a loss dictionary.

        # Train discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(d_steps):
            # Get the latent vector
            random_latent_vectors = np.random.normal(0, 1, (batch_size, self.batch_size))
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.critic(fake_images, training=True)
                # Get the logits for real images
                real_logits = self.critic(real_images, training=True)
                # Calculate discriminator loss using fake and real logits
                d_cost = self.critic_loss(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.lambdas
            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.critic.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.critic.trainable_variables)
            )

        # Train the generator now.
        # Get the latent vector
        random_latent_vectors = np.random.normal(0, 1, (batch_size, self.batch_size))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.critic(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.generator_loss(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return (d_loss, g_loss)

    def train(
        self,
        data,
        batch_size,
        epochs,
        print_every_n_batches=10,
        n_critic=10,
        using_generator=False,
    ):
        """
        This trains the GAN by alternating between training the critic 'critic_round' times
        and training the generator once in each epoch on
        the dataset x_train which has a length of batch_size.
        It will print and record the loss of the generator and critic every_n_batches.
        """

        for epoch in range(self.epoch, self.epoch + epochs):
                loss = self.train_step(data,n_critic,batch_size)
                print(
                    "%d [D loss: %.1f][G loss: %.1f]"
                    % (
                        epoch,
                        loss[0],
                        loss[1]
                    )
                )
                self.epoch += 1

    def save_model(self, run_folder):
        """
        This saves the weights of the three models that are used in the GAN on the 'filepath'.
        """
        self.model.save(run_folder + "_model.h5")
        self.critic.save(run_folder + "_critic.h5")
        self.generator.save(run_folder + "_generator.h5")

    def load_weights(self, filepath):
        """
        This loads the weights of the three models that are used in the GAN on the 'filepath'.
        """
        self.model.load_weights(filepath + "_model.h5")
        self.critic.load_weights(filepath + "_critic.h5")
        self.generator.load_weights(filepath + "_generator.h5")
