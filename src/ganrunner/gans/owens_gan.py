import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
from .utlility import copy_format, make_mask
import os
import logging
import numpy


class OwGAN(object):
    def __init__(
        self,
        optimiser,
        input_dim,
        noise_size,
        number_of_layers,
        lambdas,
        learning_rate,
        network,
        k_size,
    ):
        if network == "wgangp":
            self.network = "linear"
            print("old model")
            # backward comblilty with old models
        else:
            self.network = network.lower()
        self.net_dim = noise_size
        self.input_dim = input_dim
        self.data_dim = input_dim * k_size
        print("building", number_of_layers, self.net_dim, self.data_dim)
        self.Make_Generator(number_of_layers)
        print("Generator")
        self.Make_Critic(number_of_layers)
        print("Critic")
        self.Make_forcast(number_of_layers, input_dim)
        print("forcaster")
        # WGAN values from paper
        self.learning_rate = learning_rate

        # WGAN_gradient penalty uses ADAM ------------------------ do somthing here
        self.d_optimizer = optim.Adam(
            self.Critic.parameters(), lr=self.learning_rate#, betas=(self.b1, self.b2)
        )
        self.g_optimizer = optim.Adam(
            self.Generator.parameters(), lr=self.learning_rate#, betas=(self.b1, self.b2)
        )
        self.f_optimizer = optim.Adam(
            self.forcaster.parameters(), lr=self.learning_rate#, betas=(self.b1, self.b2)
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
        self.Generator.add_module(str(number_of_layers) + "active", nn.Tanh())
        number_of_layers -= 1
        while number_of_layers > 1:
            self.Generator.add_module(
                str(number_of_layers) + "Glayer", nn.Linear(self.net_dim, self.net_dim),
            )
            self.Generator.add_module(str(number_of_layers) + "active", nn.Tanh())
            number_of_layers -= 1
            print(number_of_layers)
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
            self.Critic.add_module(
                str(number_of_layers) + "Clayer", nn.Linear(self.net_dim, self.net_dim),
            )
            self.Critic.add_module(str(number_of_layers) + "active", nn.Tanh())
            number_of_layers -= 1
        self.Critic.add_module(
            str(number_of_layers) + "Clayer", nn.Linear(self.net_dim, 1)
        )

    def Make_forcast(self, number_of_layers, input_dim):
        """
        input sinthetic
        """
        self.forcaster = nn.Sequential()
        self.forcaster.add_module(
            str(number_of_layers) + "Flayer", nn.Linear(self.data_dim, self.net_dim)
        )
        self.forcaster.add_module(str(number_of_layers) + "active", nn.Tanh())
        number_of_layers -= 1
        while number_of_layers > 1:
            self.forcaster.add_module(
                str(number_of_layers) + "Flayer", nn.Linear(self.net_dim, self.net_dim),
            )
            self.forcaster.add_module(str(number_of_layers) + "active", nn.Tanh())
            number_of_layers -= 1
        self.forcaster.add_module(
            str(number_of_layers) + "Flayer", nn.Linear(self.net_dim, input_dim)
        )

    def create_fake(self, batch_size):
        """
        this creates a batch of fake data
        """
        z = torch.randn(batch_size, self.data_dim)
        fake_images = self.Generator(z)
        return fake_images  # .detach().numpy()

    def create_series(self,length_of_series):
        batch_size = 1
        w = self.create_fake(batch_size)[0]
        end = self.data_dim - self.input_dim
        wfac = w[:end]
        series = [w[:self.input_dim].detach().numpy()]
        for i in range(length_of_series):
            b = torch.randn(self.input_dim)
            to_forcat = torch.cat((wfac, b))
            predic = self.forcaster(to_forcat)
            middle = wfac[self.input_dim:]
            wfac = torch.cat((middle, predic))
            series.append(middle[:self.input_dim].detach().numpy())
        series.append(predic.detach().numpy())
        return series

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

    def fcast(self, w):
        b = torch.randn(self.batch_size, self.input_dim)
        end = self.data_dim - self.input_dim
        fac = torch.tensor([[1]*end + [0]*self.input_dim]*self.batch_size)
        invfac = torch.tensor([[0]*end]*self.batch_size)
        wfac = w*fac
        to_forcat = wfac + torch.cat((invfac, b), dim=1)
        predic = self.forcaster(to_forcat)
        together = wfac + torch.cat((invfac, predic), dim=1)
        return together

    def psudo_training(self, data, tf, tsf, tc, mc, print_every_n_batches):
        """
        data_dim is the number
        """
        # torch.autograd.set_detect_anomaly(True)
        d_loss_real, d_loss_fake = [0,0]
        data = torch.Tensor(data)
        nummask,bimask = make_mask(data)
        for s in range(tf):
            if s >= tsf:
                flag = True
                for p in self.Critic.parameters():
                    p.requires_grad = True
                for t in range(tc):
                    self.Critic.zero_grad()
                    ubar = Variable(self.sample_type(data))
                    # for i in range(mc):
                    ubar[bimask[self.index]] = 0
                    initw = self.create_fake(self.batch_size)
                    w = self.fcast(initw)
                    wbar = w * nummask[self.index]
                    d_loss_real, d_loss_fake = self.critic_update(
                        ubar, wbar
                    )  # include calculate_gradient_penalty,loss calculation, critic weights opti
                # end for
            else:
                flag = False
            # end if
            # for i in range(mf):
            for p in self.Critic.parameters():
                p.requires_grad = False
            self.forcaster.zero_grad()
            intw = self.create_fake(self.batch_size)
            w = self.fcast(intw)
            floss2 = self.Critic(w)  # wk is not used
            # end = self.data_dim - self.input_dim
            # floss2 = floss[:,end:]
            floss2 = floss2.mean()
            if s % print_every_n_batches == 0:
                logging.info(
                    f"iteration: {s}/{tf}({flag}), f_loss: {floss2:.2f}, loss_fake: {d_loss_fake:.2f}, loss_real: {d_loss_real:.2f}"
                )
            floss2.backward(
                self.mone
            )  # this funtions needs more info which i am going to add later
            self.f_optimizer.step()
        print("fc")

    def train_critcgen(
        self, data, epochs, hasmissing=False, print_every_n_batches=10, n_critic=5,
    ):
        """
        This trains the GAN by alternating between training the critic 'critic_round' times
        and training the generator once in each epoch on
        the dataset x_train which has a length of batch_size.
        It will print and record the loss of the generator and critic every_n_batches.
        """
        print("critc train =", n_critic)
        if self.usegpu:
            self.Critic = self.Critic.cuda()
            self.Generator = self.Generator.cuda()
        if hasmissing:
            print("missing data mode on")
        data_tensor = torch.Tensor(data)
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
                    images, fake_images = copy_format(images, fake_images, self.usegpu)
                if self.usegpu:
                    images = images.cuda()
                # Train with real images
                d_loss_real, d_loss_fake = self.critic_update(images, fake_images)
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
            g_loss.backward(self.mone)
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

    def critic_update(self, images, fake_images):
        # Train with real images
        d_loss_real = self.Critic(images)
        d_loss_real = d_loss_real.mean()
        d_loss_real.backward(self.mone)

        # Train with fake images

        d_loss_fake = self.Critic(fake_images)
        d_loss_fake = d_loss_fake.mean()
        d_loss_fake.backward(self.one)
        # Train with gradient penalty
        gradient_penalty = self.calculate_gradient_penalty(
            images.data, fake_images.data
        )
        gradient_penalty.backward()

        d_loss = d_loss_fake - d_loss_real + gradient_penalty
        Wasserstein_D = d_loss_real - d_loss_fake
        self.d_optimizer.step()
        return d_loss_real, d_loss_fake

    def calculate_gradient_penalty(self, real_images, fake_images):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        size = real_images.shape[0]
        eta = torch.FloatTensor(size, 1).uniform_(0, 1)
        if self.usegpu:
            eta = eta.cuda()
        eta = eta.expand(size, real_images.size(1))
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

    def full_train(
        self,
        data,
        batch_size,
        print_every_n_batches,
        n_gen,
        n_critic,
        n_forcast,
        tsf,
        usegpu,
        lag
    ):
        u_matrix = numpy.array(self.extend_data(lag, data, len(data)))
        self.one = torch.tensor(1, dtype=torch.float)
        self.mone = self.one * -1
        self.batch_size = batch_size
        print("gpu =", usegpu)
        self.usegpu = usegpu
        print("u", u_matrix.shape, "d", data.shape)
        self.train_critcgen(u_matrix, n_gen, True, print_every_n_batches, n_critic)
        print("gen training complet")
        self.psudo_training(
            u_matrix,
            n_forcast,
            tsf,
            n_critic,
            batch_size,
            print_every_n_batches,
        )

    def extend_data(self, k, x, n):
        U = []
        for i in range(n - k + 1):
            temp = numpy.array([])
            for j in range(k):
                temp = numpy.append(temp, x[i + j,])
            U.append(temp)
        return U

    def summary(self):
        """
        prints the composition of the gan
        """
        print(self.Critic)
        print(self.Generator)
        print(self.forcaster)

    def pick_sample(self, data):
        """
        pick a smaple of the data of size of the batch
        """
        perm = torch.randperm(len(data))
        index = perm[: self.batch_size]
        self.index = index
        return data[index]

    def save_model(self, filepath):
        """
        This saves the weights of the two networks that are used in the GAN on the 'filepath'.
        """
        torch.save(self.Generator.state_dict(), filepath + "_generator.pkl")
        torch.save(self.Critic.state_dict(), filepath + "_critic.pkl")
        torch.save(self.forcaster.state_dict(), filepath + "_forcat.pkl")
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
