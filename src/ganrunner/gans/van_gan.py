import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
from .utlility import copy_format
import os
import logging
import pandas


def normalize(dataset, mean, std):
    """
    Normalises the dataset by mean and standard deviation
    """
    mid = dataset - mean
    new_data = mid / std
    return new_data


def unnormalize(dataset, mean, std):
    """
    Reverts the normalised dataset to original format
    """
    df = pandas.DataFrame(dataset)
    mid = df * std
    original = mid + mean
    return original


def get_norm(data):
    """
    Provides the mean and standard deviation for the dataset so it can be normalised.
    """
    mean = data.mean()
    std = data.std()
    data = normalize(data, mean, std)
    return data.to_numpy("float"), mean.to_numpy("float"), std.to_numpy("float")


class vgan(object):
    def __init__(
        self,
        optimiser,
        input_dim,
        noise_size,
        number_of_layers,
        lambdas,
        learning_rate,
        network,
    ):
        self.network = network.lower()
        self.net_dim = noise_size
        self.data_dim = input_dim
        self.Make_Generator(number_of_layers)
        self.Make_Critic(number_of_layers)
        self.adversarial_loss = torch.nn.BCELoss()
        # WGAN values from paper
        self.learning_rate = learning_rate
        self.b1 = 0.5
        self.b2 = 0.999

        # WGAN_gradient penalty uses ADAM ------------------------ do somthing here
        self.d_optimizer = optim.Adam(
            self.Critic.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2)
        )
        self.g_optimizer = optim.Adam(
            self.Generator.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2)
        )
        self.lambda_term = lambdas

    def Make_Generator(self, number_of_layers):
        """
        This makes a generator network with 'number_of_layers' layers and 'net_dim' of nodes per layer.
        It takes in a vector of 'batch_size' length and outputs a vector of data that is 'data_dim' long.
        """
        self.Generator = nn.Sequential()
        self.Generator.add_module(
            str(number_of_layers) + "Glayer", nn.Linear(self.data_dim, self.net_dim)
        )
        number_of_layers -= 1
        while number_of_layers > 1:
            if self.network == "linear":
                self.Generator.add_module(
                    str(number_of_layers) + "Glayer",
                    nn.Linear(self.net_dim, self.net_dim),
                )
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
                raise ValueError("network type not found", self.network)
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
        self.Critic.add_module(str(number_of_layers) + "Clayer", nn.Sigmoid())

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

    def exctract(self, timedata, alpha=0.3):
        timedata = pandas.DataFrame(timedata)
        trend = timedata.ewm(alpha=alpha, adjust=False).mean()
        trendless = trend[::5]
        noise = timedata / trend
        return trendless, noise  # ,trendless

    def train(
        self,
        data,
        batch_size,
        epochs,
        hasmissing=False,
        print_every_n_batches=10,
        n_critic=10,
        usegpu=False,
    ):
        """
        This trains the GAN by alternating between training the critic 'critic_round' times
        and training the generator once in each epoch on
        the dataset x_train which has a length of batch_size.
        It will print and record the loss of the generator and critic every_n_batches.
        """
        print("critc train =", n_critic)
        self.summary()
        self.usegpu = usegpu
        if self.usegpu:
            self.Critic = self.Critic.cuda()
            self.Generator = self.Generator.cuda()
        if hasmissing:
            print("missing data mode on")
        self.batch_size = batch_size
        trend, noise = self.exctract(data, 0.3)
        signal, mean, std = get_norm(trend)
        data_tensor = torch.Tensor(signal)
        valid = Variable(
            torch.Tensor(batch_size, self.net_dim).fill_(1.0), requires_grad=False
        )
        fake = Variable(
            torch.Tensor(batch_size, self.net_dim).fill_(0.0), requires_grad=False
        )
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
                # Train self.Critic
                z = Variable(torch.randn(self.batch_size, self.data_dim))
                if self.usegpu:
                    fake_images = self.Generator(z.cuda())
                else:
                    fake_images = self.Generator(z)
                if hasmissing:
                    images, fake_images = copy_format(images, fake_images, self.usegpu)
                if self.usegpu:
                    images = images.cuda()
                # Train with real images
                real_loss = self.adversarial_loss(self.Critic(images), valid)
                fake_loss = self.adversarial_loss(
                    self.Critic(fake_images.detach()), fake
                )
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.d_optimizer.step()
                # Generator update
            self.Generator.zero_grad()
            # train generator
            # compute loss with fake images
            z = Variable(torch.randn(self.batch_size, self.data_dim))
            if self.usegpu:
                fake_images = self.Generator(z.cuda())
            else:
                fake_images = self.Generator(z)
            g_loss = self.adversarial_loss(self.Critic(fake_images), valid)
            g_loss = g_loss.mean()
            g_loss.backward()
            self.g_optimizer.step()
            if g_iter % print_every_n_batches == 0:
                logging.info(
                    f"iteration: {g_iter}/{epochs}, g_loss: {g_loss:.2f}, d_loss: {d_loss:.2f}"
                )
            assert g_loss > 0 or g_loss < 0
        # self.Critic = self.Critic.cpu()
        # self.Generator = self.Generator.cpu()
        return unnormalize(trend, mean, std)
        # Saving model and sampling images every 1000th generator iterations

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
