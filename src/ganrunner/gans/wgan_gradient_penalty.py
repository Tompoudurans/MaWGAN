import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os

class wGANgp(object):
    def __init__(
            self,
            optimiser,
            input_dim,
            noise_size,
            batch_size,
            number_of_layers,
            lambdas,
            learning_rate
        ):
        self.net_dim = noise_size
        self.data_dim = noise_size
        self.z_dim = input_dim
        self.Make_Generator(number_of_layers)
        self.Make_Critic(number_of_layers)
        # WGAN values from paper
        self.learning_rate = learning_rate
        self.b1 = 0.5
        self.b2 = 0.999

        # WGAN_gradient penalty uses ADAM ------------------------ do somthing here
        self.d_optimizer = optim.Adam(self.Critic.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.Generator.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.lambda_term = lambdas

    def Make_Generator(self,number_of_layers):
        self.Generator = nn.Sequential()
        self.Generator.add_module(str(number_of_layers) + "layer",nn.Linear(self.net_dim, self.net_dim))
        self.Generator.add_module(str(number_of_layers) + "active",nn.Tanh())
        number_of_layers -= 1
        while number_of_layers > 1:
            self.Generator.add_module(str(number_of_layers) + "layer",nn.Linear(self.net_dim, self.net_dim))
            self.Generator.add_module(str(number_of_layers) + "active",nn.Tanh())
            number_of_layers -= 1
        self.Generator.add_module(str(number_of_layers) + "layer",nn.Linear(self.net_dim,self.z_dim))
        #------------------------------------------------------------------------------------------------------------------
        #nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
        #nn.BatchNorm2d(num_features=1024),
        #nn.ReLU(True),

    def Make_Critic(self,number_of_layers):
        self.Critic = nn.Sequential()
        self.Critic.add_module(str(number_of_layers) + "layer",nn.Linear(self.z_dim, self.net_dim))
        self.Critic.add_module(str(number_of_layers) + "active",nn.Tanh())
        number_of_layers -= 1
        while number_of_layers > 1:
            self.Critic.add_module(str(number_of_layers) + "layer",nn.Linear(self.net_dim, self.net_dim))
            self.Critic.add_module(str(number_of_layers) + "active",nn.Tanh())
            number_of_layers -= 1
        self.Critic.add_module(str(number_of_layers) + "layer",nn.Linear(self.net_dim,1))

    def create_fake(self, batch_size):
        z = torch.randn(batch_size, self.data_dim)
        fake_images = self.Generator(z)
        return fake_images.detach().numpy()

    def train(self,data,batch_size,epochs,print_every_n_batches=10,n_critic=5):
        self.batch_size = batch_size
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
                sample = self.pick_sample(data_tensor)
                images = Variable(sample)
                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.Critic(images)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = Variable(torch.randn(self.batch_size, self.data_dim))

                fake_images = self.Generator(z)
                d_loss_fake = self.Critic(fake_images)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
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
            fake_images = self.Generator(z)
            g_loss = self.Critic(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()
            if g_iter % print_every_n_batches == 0:
                print(f'iteration: {g_iter}/{epochs}, g_loss: {g_loss}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')
            # Saving model and sampling images every 1000th generator iterations

    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1))
        interpolated = eta * real_images + ((1 - eta) * fake_images)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.Critic(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs= torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def summary(self):
        print(self.Critic)
        print(self.Generator)
        
    def pick_sample(self,data):
        perm = torch.randperm(len(data))
        index = perm[:self.batch_size]
        return data[index]

    def save_model(self,filepath):
        torch.save(self.Generator.state_dict(),filepath + '_generator.pkl')
        torch.save(self.Critic.state_dict(),filepath + '_critic.pkl')
        print('Models saved ')

    def load_model(self, filepath):
        G_model_filename = filepath + '_generator.pkl'
        D_model_filename = filepath + '_critic.pkl'
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.Critic.load_state_dict(torch.load(D_model_path))
        self.Generator.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Critic model loaded from {}-'.format(D_model_path))
