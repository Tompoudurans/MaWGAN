import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
from .masker import copy_format
import logging
import os

#___#---------------------------------------------------------------------------
#___#
#___#**Project:**         Masked Wasterstin generative adviersal network\
#___#**Script:**          wgan_gradient_penalty.py\
#___#**Author:**          Thomas Poudevigne\
#___#**Date Created:**    ‎21 ‎July ‎2021\
#___#**Reviewer:**        TBC\
#___#**Devops Feature:**  #[don't know]\
#___#**Devops Backlog:**  #[don't know]\
#___#**Devops Task:**     #[don't know]\
#___#**Devops Repo:**     ganrunner\gans\
#___#**MARS:**            "S:\..."
#___#
#___#
#____#Description
#____#This script is the main part of the GAN network code. It builds, trains
#____#and creates sythetic data
#____#This is done by using a class object
#___#
#___#---------------------------------------------------------------------------


class wGANgp(object):
#-------------------------------------------------------------------------------
#_#
#__#1.Creates the GAN according to the parameters given
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function creates the class object
#_#
#_#Reviewer Notes\
#_#
#_#

    def __init__(
        self,
        optimiser,
        number_of_variables,
        number_of_nodes,
        number_of_layers,
        lambdas,
        learning_rate,
        network,
    ):
    #_#Steps\
    #_# Save the number of nodes in a layer to the class object
        self.net_dim = number_of_nodes
    #_# Save the number of variables to the class object
        self.data_dim = number_of_variables
    #_# Create the generator and save it in the class object
        self.Make_Generator(number_of_layers)
    #_# Create the critic and save it in the class object
        self.Make_Critic(number_of_layers)
    #_# Save the lambda_term to the class object
        self.lambda_term = lambdas
    #_# Save the learning rate to the class object
        self.learning_rate = learning_rate
    #_# Create the optimzers used for training the networks
        self.make_optimize(optimiser.lower())

#-------------------------------------------------------------------------------
#_#
#__#2. Create the optimzers used for training the networks
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function creates the optimers for the critic and the generator
#_#that are used for training the networks. There is a choice of five optimsers
#_#(adam, adadelta, adagrad, rmsprop).
#_#Reviewer Notes\
#_#
#_#

    def make_optimize(self,opt):
        #_#Steps\
        #_# Check if the optimser 'adam' has been selected
        if opt == "adam":
        #_# Assign the optimser 'adam' to the Critic parameters using the saved learning rate
            self.d_optimizer = optim.Adam(self.Critic.parameters(), lr=self.learning_rate))
        #_# Assign the optimser 'adam' to the Generator parameters using the saved learning rate
            self.g_optimizer = optim.Adam(self.Generator.parameters(), lr=self.learning_rate))
        #_# Check if the optimser 'adadelta' has been selected
        if opt == "adadelta":
        #_# Assign the optimser 'adelta' to the Critic parameters using the saved learning rate
            self.d_optimizer = optim.Adadelta(self.Critic.parameters(), lr=self.learning_rate)
        #_# Assign the optimser 'adam' to the Generator parameters using the saved learning rate
            self.g_optimizer = optim.Adadelta(self.Generator.parameters(), lr=self.learning_rate)
        #_# Check if the optimser 'adagrad' has been selected
        if opt == "adagrad":
        #_# Assign the optimser 'adagrad' to the Critic parameters using the saved learning rate
            self.d_optimizer = optim.Adagrad(self.Critic.parameters(), lr=self.learning_rate)
                #_# Assign the optimser 'adagrad' to the Generator parameters using the saved learning rate
            self.g_optimizer = optim.Adagrad(self.Generator.parameters(), lr=self.learning_rate)
        #_# Check if the optimser 'rmsprop' has been selected
        if opt == "rmsprop":
        #_# Assign the optimser 'rmsprop' to the Critic parameters using the saved learning rate
            self.d_optimizer = optim.RMSprop(self.Critic.parameters(), lr=self.learning_rate)
        #_# Assign the optimser 'rmsprop' to the Generator parameters using the saved learning rate
            self.g_optimizer = optim.RMSprop(self.Generator.parameters(), lr=self.learning_rate)
        #_# Check if the optimser 'SGD' has been selected
        if opt == "sgd":
        #_# Assign the optimser 'sgd' to the Critic parameters using the saved learning rate
            self.d_optimizer = optim.SGD(self.Critic.parameters(), lr=self.learning_rate)
        #_# Assign the optimser 'sgd' to the Generator parameters using the saved learning rate
            self.g_optimizer = optim.SGD(self.Generator.parameters(), lr=self.learning_rate)

#-------------------------------------------------------------------------------
#_#
#__#3. Bulid generator
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#
#_#Reviewer Notes\
#_#
#_#
    def Make_Generator(self, number_of_layers):
        """
        This makes a generator network with 'number_of_layers' layers and 'net_dim' of nodes per layer.
        It takes in a vector of 'batch_size' length and outputs a vector of data that is 'data_dim' long.
        """
        #_#steps/
        #_#create empty neural network object
        self.Generator = nn.Sequential()
        #_# add the input layers
        self.Generator.add_module(
            str(number_of_layers) + "Glayer", nn.Linear(self.data_dim, self.net_dim)
        )
        #_# adds an activation function
        self.Generator.add_module(str(number_of_layers) + "active", nn.Tanh())
        #_# reduce the number of layer counter by 1
        number_of_layers -= 1
        #_# loops until number layers counter = 0
        while number_of_layers > 1:
        #_# creates a hidden layer
            self.Generator.add_module(
                str(number_of_layers) + "Glayer",
                nn.Linear(self.net_dim, self.net_dim),
            )
            #_# adds an activation function
            self.Generator.add_module(str(number_of_layers) + "active", nn.Tanh())
            #_# reduce the number of layer counter by 1
            number_of_layers -= 1
        #_# creates the ouput layer
        self.Generator.add_module(
            str(number_of_layers) + "Glayer", nn.Linear(self.net_dim, self.data_dim)
        )
#-------------------------------------------------------------------------------
#_#
#__#4. Bulid the critic
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#
#_#Reviewer Notes\
#_#
#_#
    def Make_Critic(self, number_of_layers):
        """
        This makes a critic network with 'number_of_layers' layers and 'net_dim' of nodes per layer.
        It takes in a vector of data that is 'data_dim' long and outputs a probability of the data being real or synthetic.
        """
        #_#steps/
        #_#create empty neural network object
        self.Critic = nn.Sequential()
        #_# add the input layers
        self.Critic.add_module(
            str(number_of_layers) + "Clayer", nn.Linear(self.data_dim, self.net_dim)
        )
        #_# adds an activation function
        self.Critic.add_module(str(number_of_layers) + "active", nn.Tanh())
        #_# reduce the number of layer counter by 1
        number_of_layers -= 1
        #_# loops until number layers counter = 0
        while number_of_layers > 1:
        #_# creates a hidden layer
            self.Critic.add_module(
                str(number_of_layers) + "Clayer",
                nn.Linear(self.net_dim, self.net_dim),
            )
            #_# adds an activation function
            self.Critic.add_module(str(number_of_layers) + "active", nn.Tanh())
            number_of_layers -= 1
        #_# creates the ouput layer
        self.Critic.add_module(
            str(number_of_layers) + "Clayer", nn.Linear(self.net_dim, 1)
        )
#-------------------------------------------------------------------------------
#_#
#__#5. Create sythetic data
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#this creates a batch of synthetic data use outside training
#_#Reviewer Notes\
#_#
#_#

    def create_synthetic(self, batch_size):
        """
        this creates a batch of synthetic data
        """
        #_#steps\
        #_#creates a random matrix with dimtions batch_size * data_dim
        z = torch.randn(batch_size, self.data_dim)
        #_# feed the random matix into the gentrator
        synthetic_data = self.Generator(z)
        #_# outputs the synthetic dataset without the gradient metadata
        return synthetic_data.detach().numpy()
#-------------------------------------------------------------------------------
#_#
#__#6. training the gan
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This trains the GAN by alternating between training the critic 'critic_round' times
#_#and training the generator once in each epoch on
#_#the dataset x_train which has a length of batch_size.
#_#It will print and record the loss of the generator and critic every_n_batches.
#_#Reviewer Notes\
#_#
#_#

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
        #_#Steps\
        #_#save usegpu flag to the class object
        self.usegpu = usegpu
        #_#if usegpu flag is true then move to the gan to the gpu
        if self.usegpu:
            self.Critic = self.Critic.cuda()
            self.Generator = self.Generator.cuda()
        #if hasmissing:
        #    print("missing data mode on")
        #_#save the batch size to the class object
        self.batch_size = batch_size
        #logging.info(self.batch_size,batch_size)
        #_# convert data format so it can be porcessed (from numpy.array to torch.tensor)
        data_tensor = torch.Tensor(data)
        #_# create a matrix of ones
        one = torch.tensor(1, dtype=torch.float)
        #_# create a matrix of minus ones
        mone = one * -1
        #_#main traing loop
        for g_iter in range(epochs):
            #_# allows the critic to be trained
            for p in self.Critic.parameters():
                p.requires_grad = True
            #d_loss_real = 0
            #d_loss_synthetic = 0
            for d_iter in range(n_critic):
                #_# reset Critic gardent
                self.Critic.zero_grad()
                #_# sample dataset
                sample = self.pick_sample(data_tensor)
                org_data = Variable(sample)
                #_#creates a random matrix with dimtions batch_size * data_dim
                z = Variable(torch.randn(self.batch_size, self.data_dim))
                #_# feed the random matix into the gentrator, tranfer to gpu if needed
                if self.usegpu:
                    synthetic_data = self.Generator(z.cuda())
                else:
                    synthetic_data = self.Generator(z)
                #_# mask the data
                if hasmissing:
                    org_data, synthetic_data = copy_format(org_data, synthetic_data,self.usegpu)
                #_# tranfer the mask data to gpu if needed
                if self.usegpu:
                    org_data = org_data.cuda()
                #_# feed the original data to the critic
                d_loss_real = self.Critic(org_data)
                #_# calculate the mean of the outputs of the critic
                d_loss_real = d_loss_real.mean()
                #_# calculate negative gradient
                d_loss_real.backward(mone)
                #_# feed the synthetic data to the critic
                d_loss_synthetic = self.Critic(synthetic_data)
                #_# calculate the mean of the outputs of the critic
                d_loss_synthetic = d_loss_synthetic.mean()
                #_# calculate positive gradient
                d_loss_synthetic.backward(one)
                #_#calculate the gradent plenalty
                gradient_penalty = self.calculate_gradient_penalty(
                    org_data.data, synthetic_data.data
                )
                #_# calculate the gradent of the gradent plenalty
                gradient_penalty.backward()
                #_# add all the loss
                d_loss = d_loss_synthetic - d_loss_real + gradient_penalty
                #_#adjust the weight of the crtic
                self.d_optimizer.step()
            # Generator update
            for p in self.Critic.parameters():
                p.requires_grad = False  # to avoid computation

            self.Generator.zero_grad()
            # train generator
            # compute loss with synthetic data
            z = Variable(torch.randn(self.batch_size, self.data_dim))
            #_# feed the random matix into the gentrator, tranfer to gpu if needed
            if self.usegpu:
                synthetic_data = self.Generator(z.cuda())
            else:
                synthetic_data = self.Generator(z)
            #_# feed the synthetic data to the critic
            g_loss = self.Critic(synthetic_data)
            #_# calculate the mean of the outputs of the critic
            g_loss = g_loss.mean()
            #_# calculate negative gradient
            g_loss.backward(mone)
            #_#adjust the weight of the gentrator
            self.g_optimizer.step()
            #_# send progess to the log file
            if g_iter % print_every_n_batches == 0:
                logging.info(
                    f"iteration: {g_iter}/{epochs}, g_loss: {g_loss:.2f}, loss_synthetic: {d_loss_synthetic:.2f}, loss_real: {d_loss_real:.2f}"
                )
        self.Critic = self.Critic.cpu()
        self.Generator = self.Generator.cpu()

    def calculate_gradient_penalty(self, real_data, synthetic_data):
        """
        Computes gradient penalty based on prediction and weighted real / synthetic samples
        """
        eta = torch.FloatTensor(self.batch_size, 1).uniform_(0, 1)
        if self.usegpu:
            eta = eta.cuda()
        eta = eta.expand(self.batch_size, real_data.size(1))
        interpolated = eta * real_data + ((1 - eta) * synthetic_data)
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
