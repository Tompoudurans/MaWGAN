from keras.layers import *  # Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D

from keras.layers.merge import _Merge
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


class RandomWeightedAverage(_Merge):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    """Provides a (random) weighted average between real and generated image samples"""

    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


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

        self.name = "gan"
        self.input_dim = input_dim
        self.critic_learning_rate = learning_rate

        self.net_dim = noise_size
        self.generator_learning_rate = learning_rate
        self.lambdas = lambdas
        self.optimiser = optimiser

        self.weight_init = RandomNormal(
            mean=0.0, stddev=0.02
        )  # 'he_normal' #RandomNormal(mean=0., stddev=0.02)
        self.grad_weight = 1
        self.batch_size = batch_size

        self.d_losses = []
        self.g_losses = []
        self.epoch = 0

        self.make_critc(number_of_layers)
        self.make_generator(number_of_layers)

        self._build_adversarial()

    def gradient_penalty_loss(self, y_true, y_pred, interpolated_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, interpolated_samples)[0]

        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(
            gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape))
        )
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = self.lambdas * K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein(self, y_true, y_pred):
        """
        calcuate half the wasserstein distance
        """
        return -K.mean(y_true * y_pred)

    def RandomWeightedAv(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

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
        dis_layer = Dense(self.net_dim, activation="relu")(dis_input)
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

    def _build_adversarial(self):
        """
        This compiles the critic and
        then compiles a GAN model that is used for training the generator
        it consists of the generator directly outputed into a frozen critic
        """
        # -------------------------------
        # Construct Computational Graph
        #       for the Critic
        # -------------------------------

        # Freeze generator's layers while training critic
        self.set_trainable(self.generator, False)

        # Image input (real sample)
        real_img = Input(shape=(self.input_dim,))

        # Fake image
        z_disc = Input(shape=(self.net_dim,))
        fake_img = self.generator(z_disc)

        # critic determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage(self.batch_size)([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'interpolated_samples' argument
        partial_gp_loss = partial(
            self.gradient_penalty_loss, interpolated_samples=interpolated_img
        )
        partial_gp_loss.__name__ = "gradient_penalty"  # Keras requires function names

        self.critic_model = Model(
            inputs=[real_img, z_disc], outputs=[valid, fake, validity_interpolated]
        )

        self.critic_model.compile(
            loss=[self.wasserstein, self.wasserstein, partial_gp_loss],
            optimizer=self.get_opti(self.critic_learning_rate),
            loss_weights=[1, 1, self.grad_weight],
        )

        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        self.set_trainable(self.critic, False)
        self.set_trainable(self.generator, True)

        # Sampled noise for input to generator
        model_input = Input(shape=(self.net_dim,))
        # Generate images based of noise
        img = self.generator(model_input)
        # Discriminator determines validity
        model_output = self.critic(img)
        # Defines generator model
        self.model = Model(model_input, model_output)

        self.model.compile(
            optimizer=self.get_opti(self.generator_learning_rate), loss=self.wasserstein
        )

        self.set_trainable(self.critic, True)

    def train_critic(self, x_train, batch_size, using_generator=False):
        """
        This trains the critc once by creating a set of fake_data and
        traning them aganist the real_data, then the wieght are cliped
        """
        valid = np.ones((batch_size, 1), dtype=np.float32)
        fake = -np.ones((batch_size, 1), dtype=np.float32)
        dummy = np.zeros(
            (batch_size, 1), dtype=np.float32
        )  # Dummy gt for gradient penalty

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.net_dim))

        # print([true_imgs, noise], [valid, fake, dummy])
        d_loss = self.critic_model.train_on_batch(
            [true_imgs, noise], [valid, fake, dummy]
        )
        return d_loss

    def train_generator(self, batch_size):
        """
        This trains the generator once by creating a set of fake data and
        uses the critic score to train on
        """
        valid = np.ones((batch_size, 1), dtype=np.float32)
        noise = np.random.normal(0, 1, (batch_size, self.net_dim))
        return self.model.train_on_batch(noise, valid)

    def create_fake(self, batch_size):
        """
        this creates a batch of fake data
        """
        noise = np.random.normal(0, 1, (batch_size, self.batch_size))
        fake_data = self.generator.predict(noise)
        return fake_data

    def train(
        self,
        x_train,
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
            critic_loops = n_critic
            for _ in range(critic_loops):
                d_loss = self.train_critic(x_train, batch_size, using_generator)
            g_loss = self.train_generator(batch_size)
            if epoch % print_every_n_batches == 0:
                self.d_losses.append(d_loss)
                self.g_losses.append(g_loss)
                assert not np.isnan(g_loss)
                print(
                    "%d (%d) [D loss: (%.1f)(R %.1f, F %.1f, G %.1f)] [G loss: %.1f]"
                    % (
                        epoch,
                        critic_loops,
                        d_loss[0],
                        d_loss[1],
                        d_loss[2],
                        d_loss[3],
                        g_loss,
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
