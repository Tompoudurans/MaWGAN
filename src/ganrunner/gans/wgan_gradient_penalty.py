import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os

class WGAN_GP(object):
    def __init__(self):
        print("WGAN_GradientPenalty init model.")
        self.net_dim = 150
        self.data_dim = 150
        self.z_dim = 4
        self.Generator(5)
        self.Discriminator(5)
        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = 150

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        self.generator_iters = 2000 # epochs
        self.critic_iter = 5
        self.lambda_term = 10

    def Generator(self,number_of_layers):
        self.G = nn.Sequential()
        self.G.add_module(str(number_of_layers) + "layer",nn.Linear(self.net_dim, self.net_dim))
        self.G.add_module(str(number_of_layers) + "active",nn.Tanh())
        number_of_layers -= 1
        while number_of_layers > 1:
            self.G.add_module(str(number_of_layers) + "layer",nn.Linear(self.net_dim, self.net_dim))
            self.G.add_module(str(number_of_layers) + "active",nn.Tanh())
            number_of_layers -= 1
        self.G.add_module(str(number_of_layers) + "layer",nn.Linear(self.net_dim,self.z_dim))
        #------------------------------------------------------------------------------------------------------------------
        #nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
        #nn.BatchNorm2d(num_features=1024),
        #nn.ReLU(True),

    def Discriminator(self,number_of_layers):
        self.D = nn.Sequential()
        self.D.add_module(str(number_of_layers) + "layer",nn.Linear(self.z_dim, self.net_dim))
        self.D.add_module(str(number_of_layers) + "active",nn.Tanh())
        number_of_layers -= 1
        while number_of_layers > 1:
            self.D.add_module(str(number_of_layers) + "layer",nn.Linear(self.net_dim, self.net_dim))
            self.D.add_module(str(number_of_layers) + "active",nn.Tanh())
            number_of_layers -= 1
        self.D.add_module(str(number_of_layers) + "layer",nn.Linear(self.net_dim,1))


    def get_torch_variable(self, arg):
            return Variable(arg)

    def train(self,data):
        self.data = torch.Tensor(data)
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        for g_iter in range(self.generator_iters):
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0
            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                images = self.get_torch_variable(self.data)

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = self.get_torch_variable(torch.randn(self.batch_size, 150))

                fake_images = self.G(z)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
                gradient_penalty.backward()


                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()
            # train generator
            # compute loss with fake images
            z = self.get_torch_variable(torch.randn(self.batch_size, 150))
            fake_images = self.G(z)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')
            # Saving model and sampling images every 1000th generator iterations

    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1))
        interpolated = eta * real_images + ((1 - eta) * fake_images)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs= torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images
