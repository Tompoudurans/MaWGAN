import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
from .utlility import copy_format
import logging
plt.switch_backend("agg")
import os


class wGANgp(object):
    def __init__(
        self,
        optimiser,
        input_dim,
        noise_size,
        number_of_layers,
        lambdas,
        learning_rate,
        network,
        #b1=0.5,
        #b2=0.999
    ):
        if network == "wgangp":
            self.network = "linear"
            print("old model")
            # backward comblilty with old models
        else:
            self.network = network.lower()
        self.net_dim = noise_size
        self.data_dim = input_dim
        self.Make_Generator(number_of_layers)
        self.Make_Critic(number_of_layers)
        # WGAN values from paper
        self.learning_rate = learning_rate
        #self.b1 = b1
        #self.b2 = b2
        self.lambda_term = lambdas
        self.make_optimize(optimiser.lower())

    def make_optimize(self,opt):
        if opt == "adam":
            self.d_optimizer = optim.Adam(
                self.Critic.parameters(), lr=self.learning_rate)#, betas=(self.b1, self.b2)
            )
            self.g_optimizer = optim.Adam(
                self.Generator.parameters(), lr=self.learning_rate)#, betas=(self.b1, self.b2)
            )
        if opt == "adadelta":
            self.d_optimizer = optim.Adadelta(self.Critic.parameters(), lr=self.learning_rate)
            self.g_optimizer = optim.Adadelta(self.Generator.parameters(), lr=self.learning_rate)
        if opt == "adagrad":
            self.d_optimizer = optim.Adagrad(self.Critic.parameters(), lr=self.learning_rate)
            self.g_optimizer = optim.Adagrad(self.Generator.parameters(), lr=self.learning_rate)
        if opt == "rmsprop":
            self.d_optimizer = optim.RMSprop(self.Critic.parameters(), lr=self.learning_rate)
            self.g_optimizer = optim.RMSprop(self.Generator.parameters(), lr=self.learning_rate)


    def Make_Generator(self, number_of_layers):
        """
        This makes a generator network with 'number_of_layers' layers and 'net_dim' of nodes per layer.
        It takes in a vector of 'batch_size' length and outputs a vector of data that is 'data_dim' long.
        """
        self.Generator = nn.Sequential()
        self.Generator.add_module(
            str(number_of_layers) + "Glayer", nn.Linear(self.data_dim, self.net_dim)
        )
        self.Generator.add_module(str(number_of_layers) + "active", nn.Tanh())
        number_of_layers -= 1
        while number_of_layers > 1:
            if self.network == "linear":
                self.Generator.add_module(
                    str(number_of_layers) + "Glayer",
                    nn.Linear(self.net_dim, self.net_dim),
                )
                self.Generator.add_module(str(number_of_layers) + "active", nn.Tanh())
            elif self.network == "rnn":
                self.Generator.add_module(
                    str(number_of_layers) + "Glayer",
                    nn.RNNCell(self.net_dim, self.net_dim),
                )
            elif self.network == "lstm":
                self.Generator.add_module(
                    str(number_of_layers) + "Glayer",
                    nn.LSTMCell(self.net_dim, self.net_dim),
                )
            elif self.network == "gru":
                self.Generator.add_module(
                    str(number_of_layers) + "Glayer",
                    nn.GRUCell(self.net_dim, self.net_dim),
                )
            else:
                raise ValueError("network type not found")
            number_of_layers -= 1
        self.Generator.add_module(
            str(number_of_layers) + "Glayer", nn.Linear(self.net_dim, self.data_dim)
        )

    def Make_Critic(self, number_of_layers):
        """
        This makes a critic network with 'number_of_layers' layers and 'net_dim' of nodes per layer.
        It takes in a vector of data that is 'data_dim' long and outputs a probability of the data being real or fake.
        """
        self.Critic = nn.Sequential()
        self.Critic.add_module(
            str(number_of_layers) + "Clayer", nn.Linear(self.data_dim, self.net_dim)
        )
        self.Critic.add_module(str(number_of_layers) + "active", nn.Tanh())
        number_of_layers -= 1
        while number_of_layers > 1:
            if self.network == "linear":
                self.Critic.add_module(
                    str(number_of_layers) + "Clayer",
                    nn.Linear(self.net_dim, self.net_dim),
                )
                self.Critic.add_module(str(number_of_layers) + "active", nn.Tanh())
            elif self.network == "rnn":
                self.Critic.add_module(
                    str(number_of_layers) + "Clayer",
                    nn.RNNCell(self.net_dim, self.net_dim),
                )
            elif self.network == "lstm":
                self.Critic.add_module(
                    str(number_of_layers) + "Clayer",
                    nn.LSTMCell(self.net_dim, self.net_dim),
                )
            elif self.network == "gru":
                self.Critic.add_module(
                    str(number_of_layers) + "Clayer",
                    nn.GRUCell(self.net_dim, self.net_dim),
                )
            else:
                raise ValueError("network type not found")
            number_of_layers -= 1
        self.Critic.add_module(
            str(number_of_layers) + "Clayer", nn.Linear(self.net_dim, 1)
        )

    def create_fake(self, batch_size):
        """
        this creates a batch of fake data
        """
        z = torch.randn(batch_size, self.data_dim)
        fake_images = self.Generator(z)
        return fake_images.detach().numpy()

    def linear_sample(self, data):
        "select samples that are linearly dependent"
        sizes = len(data) - self.batch_size
        start_loc = torch.randint(0, sizes, (1,))
        index = range(start_loc, start_loc + self.batch_size)
        return data[index]

    def sample_type(self, data):
        if self.network == "linear":
            sample = self.pick_sample(data)
        else:
            sample = self.linear_sample(data)
        return sample

    def train(
        self,
        data,
        batch_size,
        epochs,
        hasmissing=False,
        print_every_n_batches=10,
        n_critic=5,
        usegpu=False
    ):
        """
        This trains the GAN by alternating between training the critic 'critic_round' times
        and training the generator once in each epoch on
        the dataset x_train which has a length of batch_size.
        It will print and record the loss of the generator and critic every_n_batches.
        """
        self.usegpu = usegpu
        if self.usegpu:
            self.Critic = self.Critic.cuda()
            self.Generator = self.Generator.cuda()
        if hasmissing:
            print("missing data mode on")
        self.batch_size = batch_size
        logging.info(self.batch_size,batch_size)
        data_tensor = torch.Tensor(data)
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        for g_iter in range(epochs):
            # Requires grad, Generator requires_grad = False
            for p in self.Critic.parameters():
                p.requires_grad = True
            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0
            # Train Dicriminator forward-loss-backward-update n_critic times while 1 Generator forward-loss-backward-update
            for d_iter in range(n_critic):
                self.Critic.zero_grad()
                sample = self.sample_type(data_tensor)
                images = Variable(sample)
                # Train discriminator
                z = Variable(torch.randn(self.batch_size, self.data_dim))
                if self.usegpu:
                    fake_images = self.Generator(z.cuda())
                else:
                    fake_images = self.Generator(z)
                if hasmissing:
                    images, fake_images = copy_format(images, fake_images,self.usegpu)
                if self.usegpu:
                    images = images.cuda()
                # Train with real images
                d_loss_real = self.Critic(images)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images

                d_loss_fake = self.Critic(fake_images)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)
                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(
                    images.data, fake_images.data
                )
                gradient_penalty.backward()

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
            # Generator update
            for p in self.Critic.parameters():
                p.requires_grad = False  # to avoid computation

            self.Generator.zero_grad()
            # train generator
            # compute loss with fake images
            z = Variable(torch.randn(self.batch_size, self.data_dim))
            if self.usegpu:
                fake_images = self.Generator(z.cuda())
            else:
                fake_images = self.Generator(z)
            g_loss = self.Critic(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()
            if g_iter % print_every_n_batches == 0:
                logging.info(
                    f"iteration: {g_iter}/{epochs}, g_loss: {g_loss:.2f}, loss_fake: {d_loss_fake:.2f}, loss_real: {d_loss_real:.2f}"
                )
            assert g_loss > 0 or g_loss < 0
        self.Critic = self.Critic.cpu()
        self.Generator = self.Generator.cpu()
        # Saving model and sampling images every 1000th generator iterations

    def calculate_gradient_penalty(self, real_images, fake_images):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        eta = torch.FloatTensor(self.batch_size, 1).uniform_(0, 1)
        if self.usegpu:
            eta = eta.cuda()
        eta = eta.expand(self.batch_size, real_images.size(1))
        interpolated = eta * real_images + ((1 - eta) * fake_images)
        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)
        # calculate probability of interpolated examples
        prob_interpolated = self.Critic(interpolated)
        # calculate gradients of probabilities with respect to examples
        if self.usegpu:
            gradients = autograd.grad(
                outputs=prob_interpolated,
                inputs=interpolated,
                grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                create_graph=True,
                retain_graph=True,
            )[0]
        else:
            gradients = autograd.grad(
                outputs=prob_interpolated,
                inputs=interpolated,
                grad_outputs=torch.ones(prob_interpolated.size()),
                create_graph=True,
                retain_graph=True,
            )[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def summary(self):
        """
        prints the composition of the gan
        """
        print(self.Critic)
        print(self.Generator)

    def pick_sample(self, data):
        """
        pick a smaple of the data of size of the batch
        """
        perm = torch.randperm(len(data))
        index = perm[: self.batch_size]
        return data[index]

    def save_model(self, filepath):
        """
        This saves the weights of the two networks that are used in the GAN on the 'filepath'.
        """
        torch.save(self.Generator.state_dict(), filepath + "_generator.pkl")
        torch.save(self.Critic.state_dict(), filepath + "_critic.pkl")
        print("Models saved ")

    def load_model(self, filepath):
        """
        This loads the weights of the two networks that are used in the GAN on the 'filepath'.
        """
        G_model_filename = filepath + "_generator.pkl"
        D_model_filename = filepath + "_critic.pkl"
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.Critic.load_state_dict(torch.load(D_model_path))
        self.Generator.load_state_dict(torch.load(G_model_path))
        print("Generator model loaded from {}.".format(G_model_path))
        print("Critic model loaded from {}-".format(D_model_path))
