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

class supGenerator(torch.nn.Module):
    def __init__(self,number_of_layers,net_dim,data_dim,emb_dim):
        """
        This makes a generator network with 'number_of_layers' layers and 'net_dim' of nodes per layer.
        It takes in a vector of 'batch_size' length and outputs a vector of data that is 'data_dim' long.
        """
        super(supGenerator,self).__init__()
        self.net_dim = net_dim
        self.data_dim = data_dim
        self.emb_dim = emb_dim
        self.emb()
        self.center(number_of_layers)


    def emb(self):
        self.emb_output = nn.Sequential(
            nn.Embedding(self.emb_dim, self.net_dim), nn.Linear(self.net_dim, self.data_dim)
        )

    def center(self,number_of_layers):
        self.model = nn.Sequential()
        self.model.add_module(
            str(number_of_layers) + "Glayer", nn.Linear(self.data_dim*2, self.net_dim)
        )
        self.model.add_module(str(number_of_layers) + "active", nn.Tanh())
        number_of_layers -= 1
        while number_of_layers > 1:
            self.model.add_module(
                str(number_of_layers) + "rec_Glayer",
                nn.GRUCell(self.net_dim, self.net_dim),
            )
            number_of_layers -= 1
        self.model.add_module(
            str(number_of_layers) + "Glayer",nn.Linear(self.net_dim, self.data_dim),
        )

    def forward(self,data,label):
        concat = torch.cat((data, self.emb_output(label)), dim=1)
        end = concat.float()
        return self.model(end)



class supCritic(torch.nn.Module):
    def __init__(self,number_of_layers,net_dim,data_dim,emb_dim):
        """
        This makes a generator network with 'number_of_layers' layers and 'net_dim' of nodes per layer.
        It takes in a vector of 'batch_size' length and outputs a vector of data that is 'data_dim' long.
        """
        super(supCritic,self).__init__()
        self.net_dim = net_dim
        self.data_dim = data_dim
        self.emb(emb_dim)
        self.center(number_of_layers)

    def emb(self,emb_dim):
        self.emb_output = nn.Sequential(
            nn.Embedding(emb_dim, self.net_dim), nn.Linear(self.net_dim, self.data_dim)
        )

    def center(self,number_of_layers):
        self.model = nn.Sequential()
        self.model.add_module(
            str(number_of_layers) + "Clayer", nn.Linear(self.data_dim*2, self.net_dim)
        )
        self.model.add_module(str(number_of_layers) + "active", nn.Tanh())
        number_of_layers -= 1
        while number_of_layers > 1:
            self.model.add_module(
                str(number_of_layers) + "rec_Clayer",
                nn.GRUCell(self.net_dim, self.net_dim)
            )
            number_of_layers -= 1
        self.model.add_module(
            str(number_of_layers) + "Clayer", nn.Linear(self.net_dim, 1)
        )
        #self.model.add_module(str(number_of_layers) + "active", nn.Tanh())

    def forward(self,data,label):
        try:
            concat = torch.cat((data, self.emb_output(label)), dim=1)
            end = concat.float()
            return self.model(end)
        except Exception as e:
            print("critic")
            print(data)#.shape)
            print(label.shape)
            print(concat.shape)
            raise RuntimeError(e)

class decompGAN(object):
    def __init__(
        self,
        optimiser,
        input_dim,
        noise_size,
        number_of_layers,
        lambdas,
        learning_rate,
        network,
        emb_dim
    ):
        self.network = network.lower()
        self.data_dim = input_dim
        self.Generator = supGenerator(number_of_layers,noise_size,input_dim,emb_dim)
        self.Critic = supCritic(number_of_layers,noise_size,input_dim,emb_dim)
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


    def create_fake(self, batch_size, label):
        """
        this creates a batch of fake data
        """
        labels_tensor = torch.tensor(label)
        z = torch.randn(batch_size, self.data_dim)
        fake_datas = self.Generator(z,labels_tensor)
        return fake_datas.detach().numpy()

    def linear_sample(self, data, label):
        """
        select samples that are linearly dependent
        """
        sizes = len(data) - self.batch_size
        start_loc = torch.randint(0, sizes, (1,))
        index = range(start_loc, start_loc + self.batch_size)
        return data[index], label[index]



    def sample_type(self, data, label):
        sample, self.tar = self.linear_sample(data,label)

    def train(
        self,
        data,
        label,
        batch_size,
        epochs,
        hasmissing=False,
        print_every_n_batches=10,
        n_critic=5,
        usegpu=False,
    ):
        """
        This trains the GAN by alternating between training the critic 'critic_round' times
        and training the generator once in each epoch on
        the dataset x_train which has a length of batch_size.
        It will print and record the loss of the generator and critic every_n_batches.
        """
        print("critc train =", n_critic)
        print("gpu =", usegpu)
        self.usegpu = usegpu
        if self.usegpu:
            self.Critic = self.Critic.cuda()
            self.Generator = self.Generator.cuda()
        if hasmissing:
            print("missing data mode on")
        self.batch_size = batch_size
        data_tensor = torch.Tensor(data)
        labels_tensor = torch.LongTensor(label)
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
                sample,self.tar = self.sample_type(data_tensor,labels_tensor)
                self.Critic.zero_grad()
                datas = Variable(sample)
                # Train discriminator
                z = Variable(torch.randn(self.batch_size, self.data_dim))
                if self.usegpu:
                    fake_datas = self.Generator(z.cuda())
                else:
                    fake_datas = self.Generator(z,self.tar)
                if hasmissing:
                    datas, fake_datas = copy_format(datas, fake_datas, self.usegpu)
                if self.usegpu:
                    datas = datas.cuda()
                # Train with real datas
                d_loss_real = self.Critic(datas,self.tar)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)
                # if torch.cuda.device_count() > 1:
                #  model = nn.DataParallel(model)
                # Train with fake datas

                d_loss_fake = self.Critic(fake_datas,self.tar)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)
                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(
                    datas.data, fake_datas.data
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
            # compute loss with fake datas
            z = Variable(torch.randn(self.batch_size, self.data_dim))
            if self.usegpu:
                fake_datas = self.Generator(z.cuda())
            else:
                fake_datas = self.Generator(z,self.tar)
            g_loss = self.Critic(fake_datas,self.tar)
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
        # Saving model and sampling datas every 1000th generator iterations

    def calculate_gradient_penalty(self, real_datas, fake_datas):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        eta = torch.FloatTensor(self.batch_size, 1).uniform_(0, 1)
        if self.usegpu:
            eta = eta.cuda()
        eta = eta.expand(self.batch_size, real_datas.size(1))
        interpolated = eta * real_datas + ((1 - eta) * fake_datas)
        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)
        # calculate probability of interpolated examples
        prob_interpolated = self.Critic(interpolated,self.tar)
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
