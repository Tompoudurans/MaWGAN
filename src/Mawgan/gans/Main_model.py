import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
from .masker import make_mask
import logging

#___#---------------------------------------------------------------------------
#___#
#___#**Project:**         Masked Wasterstin generative adviersal network\
#___#**Script:**          wgan_gradient_penalty.py\
#___#**Author:**          Thomas Poudevigne\
#___#**Date Created:**    21 July 2021\
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


class MaWGAN(object):
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
        learning_rate
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
            self.d_optimizer = optim.Adam(self.Critic.parameters(), lr=self.learning_rate)
        #_# Assign the optimser 'adam' to the Generator parameters using the saved learning rate
            self.g_optimizer = optim.Adam(self.Generator.parameters(), lr=self.learning_rate)
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
#__#3. Build generator
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This makes a Generator network with a given number of layers (number_of_layers) and
#_# a given number of nodes per layer (net_dim).
#_#The input to the generator is a random matrix where the dimensions are the batch size and
#_#the number of variables (data_dim) . The outputs are synthetic data with the same size as the input matrix.
#_#Reviewer Notes\
#_#
#_#
    def Make_Generator(self, number_of_layers):
        """
        This makes a generator network with 'number_of_layers' layers and 'net_dim' of nodes per layer.
        It takes in a vector of 'batch_size' length and outputs a vector of data that is 'data_dim' long.
        """
        #_#Steps/
        #_#Create empty neural network object
        self.Generator = nn.Sequential()
        #_# Add the input layers
        self.Generator.add_module(
            str(number_of_layers) + "Glayer", nn.Linear(self.data_dim, self.net_dim)
        )
        #_# Add an activation function
        self.Generator.add_module(str(number_of_layers) + "active", nn.Tanh())
        #_# Reduce the number of layer count by 1
        number_of_layers -= 1
        #_# Loops until number layer count = 0
        while number_of_layers > 1:
        #_# Create a hidden layer
            self.Generator.add_module(
                str(number_of_layers) + "Glayer",
                nn.Linear(self.net_dim, self.net_dim),
            )
            #_# Add an activation function
            self.Generator.add_module(str(number_of_layers) + "active", nn.Tanh())
            #_# Reduce the number of layer count by 1
            number_of_layers -= 1
        #_# Create the ouput layer
        self.Generator.add_module(
            str(number_of_layers) + "Glayer", nn.Linear(self.net_dim, self.data_dim)
        )
#-------------------------------------------------------------------------------
#_#
#__#4. Build the Critic
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This makes a Critic network with a given number of layers (number_of_layers) and
#_# a given number of nodes per layer (net_dim).
#_#It takes in a dataset that has a given number of variables (data_dim)
#_#and outputs a vector that deduces whether or not each
#_#observation is  real or synthetic.
#_#Reviewer Notes\
#_#
#_#
    def Make_Critic(self, number_of_layers):
        """
        This makes a critic network with 'number_of_layers' layers and 'net_dim' of nodes per layer.
        It takes in a vector of data that is 'data_dim' long and outputs a probability of the data being real or synthetic.
        """
        #_#Steps/
        #_#Create empty neural network object
        self.Critic = nn.Sequential()
        #_# add the input layers
        self.Critic.add_module(
            str(number_of_layers) + "Clayer", nn.Linear(self.data_dim, self.net_dim)
        )
        #_# Add an activation function
        self.Critic.add_module(str(number_of_layers) + "active", nn.Tanh())
        #_# Reduce the number of layer count by 1
        number_of_layers -= 1
        #_# Loop until number layer count = 0
        while number_of_layers > 1:
        #_# Create a hidden layer
            self.Critic.add_module(
                str(number_of_layers) + "Clayer",
                nn.Linear(self.net_dim, self.net_dim),
            )
            #_# Add an activation function
            self.Critic.add_module(str(number_of_layers) + "active", nn.Tanh())
            number_of_layers -= 1
        #_# Create the ouput layer
        self.Critic.add_module(
            str(number_of_layers) + "Clayer", nn.Linear(self.net_dim, 1)
        )
#-------------------------------------------------------------------------------
#_#
#__#5. Create synthetic data
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function creates a batch of synthetic data to use outside training
#_#Reviewer Notes\
#_#
#_#

    def create_synthetic(self, batch_size):
        """
        This creates a batch of synthetic data
        """
        #_#Steps\
        #_#Create a random matrix with dimensions batch_size * data_dim
        z = torch.randn(batch_size, self.data_dim)
        #_# Feed the random matrix into the Generator
        synthetic_data = self.Generator(z)
        #_# Output the synthetic dataset without the gradient metadata
        return synthetic_data.detach().numpy()
#-------------------------------------------------------------------------------
#_#
#__#6. Training the GAN
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function trains the GAN by alternating between training the Critic a number of times (critic_round)
#_#and training the Generator once in each epoch on
#_#a given dataset (data). If the given dataset has a 'hasmissing flag' set to TRUE,
#_#the MAWGAN code is run, otherwise the WGAN-GP code is run.
#_#This function will record the Loss of the Generator and Critic on a given schedule (record_every_n_batches)
#_#Reviewer Notes\
#_#
#_#

    def train(
        self,
        data,
        batch_size,
        epochs,
        hasmissing=False,
        record_every_n_batches=10,
        n_critic=5,
        usegpu=False
    ):
        """
        This trains the GAN by alternating between training the critic 'critic_round' times
        and training the generator once in each epoch
        """
        #_#Steps\
        #_#Save usegpu flag to the class object
        self.usegpu = usegpu
        #_#If the usegpu flag is true then move the gan to the gpu
        if self.usegpu:
            self.Critic = self.Critic.cuda()
            self.Generator = self.Generator.cuda()
        #_#Save the batch size to the class object
        self.batch_size = batch_size
        #_# Convert data format so it can be processed (from numpy.array to torch.tensor)
        data_tensor = torch.Tensor(data)
        #_# Create a matrix of ones
        one = torch.tensor(1, dtype=torch.float)
        #_# Create a matrix of minus ones
        mone = one * -1
        #_# mask the data
        mask, binary_mask = make_mask(data_tensor)
        #_# Apply the mask
        data_tensor[binary_mask] = 0
        #_# tranfer the original data to gpu if needed
        if self.usegpu:
            data_tensor = data_tensor.cuda()
            #_# tranfer the mask to gpu if needed
            mask =  mask.cuda()
        #_#Main traing loop
        for g_iter in range(epochs):
            #_# Allow the critic to be trained
            for p in self.Critic.parameters():
                p.requires_grad = True
            for d_iter in range(n_critic):
                #_# reset Critic calculate_gradient_penaltyent
                self.Critic.zero_grad()
                #_# sample dataset
                sample_data, sample_mask = self.pick_sample(data_tensor,mask)
                org_data = Variable(sample_data)
                #_#create a random matrix with dimensions batch_size * data_dim
                z = Variable(torch.randn(self.batch_size, self.data_dim))
                #_# feed the random matrix into the generator, tranfer to gpu if needed
                if self.usegpu:
                    synthetic_data = self.Generator(z.cuda())
                else:
                    synthetic_data = self.Generator(z)
                #_# Applies the mask to the generated data
                synthetic_data = synthetic_data * sample_mask
                #_# feed the original data to the Critic
                d_loss_real = self.Critic(org_data)
                #_# calculate the mean of the outputs of the Critic
                d_loss_real = d_loss_real.mean()
                #_# calculate negative gradient
                d_loss_real.backward(mone)
                #_# feed the synthetic data to the Critic
                d_loss_synthetic = self.Critic(synthetic_data)
                #_# calculate the mean of the outputs of the Critic
                d_loss_synthetic = d_loss_synthetic.mean()
                #_# calculate positive gradient
                d_loss_synthetic.backward(one)
                #_#calculate the gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(
                    org_data.data, synthetic_data.data
                )
                #_# calculate the gradient of the gradient penalty
                gradient_penalty.backward()
                #_# add all the loss
                d_loss = d_loss_synthetic - d_loss_real + gradient_penalty
                #_#adjust the weight of the Critic
                self.d_optimizer.step()
            #_# Forbid the critic to be trained
            for p in self.Critic.parameters():
                p.requires_grad = False  # to avoid computation
            self.Generator.zero_grad()
            #_#create a random matrix with dimensions batch_size * data_dim
            z = Variable(torch.randn(self.batch_size, self.data_dim))
            #_# feed the random matrix into the generator, tranfer to gpu if needed
            if self.usegpu:
                synthetic_data = self.Generator(z.cuda())
            else:
                synthetic_data = self.Generator(z)
            #_# feed the synthetic data to the Critic
            g_loss = self.Critic(synthetic_data)
            #_# calculate the mean of the outputs of the Critic
            g_loss = g_loss.mean()
            #_# calculate negative gradient
            g_loss.backward(mone)
            #_#adjust the weight of the generator
            self.g_optimizer.step()
            #_# send progress to the log file
            if g_iter % record_every_n_batches == 0:
                logging.info(
                    f"iteration: {g_iter}/{epochs}, g_loss: {g_loss:.2f}, loss_synthetic: {d_loss_synthetic:.2f}, loss_real: {d_loss_real:.2f}"
                )
        if usegpu:
            #_# move the networks back to the cpu unless they were alredy there
            self.Critic = self.Critic.cpu()
            self.Generator = self.Generator.cpu()
#-------------------------------------------------------------------------------
#_#
#__#7. Calculate the gradient penalty
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function computes a gradient penalty based on the randomly weighted mean of the real and synthetic sample
#_#Reviewer Notes\
#_#
    def calculate_gradient_penalty(self, real_data, synthetic_data):
        """
        This function computes a gradient penalty based on the randomly weighted mean of the real and synthetic sample
        """
        #_#Steps\
        #_#Create eta: a random value beteween 0 and 1
        eta = torch.FloatTensor(self.batch_size, 1).uniform_(0, 1)
        #_# Move eta to gpu if the gpu flag is True
        if self.usegpu:
            eta = eta.cuda()
        #_# Match the eta shape to the original and synthetic shape
        eta = eta.expand(self.batch_size, real_data.size(1))
        #_# Calculate eta*real_data + (1-eta)*synthetic_data term
        interpolated = eta * real_data + ((1 - eta) * synthetic_data)
        #_# Convert the interpolated variable to a format required by the library
        interpolated = Variable(interpolated, requires_grad=True)
        #_# Feed the interpolated data to the Critic
        prob_interpolated = self.Critic(interpolated)
        #_# Calculate gradients of the above output
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
        #_# Calculate gradient penalty
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        #_# outputs gradient penalty
        return grad_penalty

#-------------------------------------------------------------------------------
#_#
#__#8. Summary
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#Prints the composition of the GAN
#_#Reviewer Notes\
#_#
    def summary(self):

        """
        prints the composition of the GAN
        """
        #_# Steps\
        #_# Prints the composition of the Critic
        print(self.Critic)
        #_# Prints the composition of the Generator
        print(self.Generator)
#-------------------------------------------------------------------------------
#_#
#__#9. Pick a sample
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function picks a sample of the data the size of a batch
#_#Reviewer Notes\
#_#
    def pick_sample(self, data, mask):
        """
        This function picks a sample of the data the size of a batch
        """
        #_# Steps\
        #_# Reorder the index randomly
        perm = torch.randperm(len(data))
        #_# Crop the index to the lentgh of the batch size
        index = perm[: self.batch_size]
        #_# Output the data selected by the random index
        return data[index], mask[index]
#-------------------------------------------------------------------------------
#_#
#__#10. Save the model
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function saves the weights of the two networks that are used in the GAN on the 'filepath'.
#_#Reviewer Notes\
#_#
    def save_model(self, filepath):
        """
        This saves the weights of the two networks that are used in the GAN on the 'filepath'.
        """
        #_#Steps\
        #_# Save Generator weights
        torch.save(self.Generator.state_dict(), filepath + "_generator.pkl")
        #_# Save Critic weights
        torch.save(self.Critic.state_dict(), filepath + "_critic.pkl")
        #_# Print that the process is complete
        print("Models saved ")
#-------------------------------------------------------------------------------
#_#
#__#10. Model loading
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function loads the weights of the two networks that are used in the GAN on the 'filepath'.
#_#Reviewer Notes\
#_#
    def load_model(self, filepath):
        """
        This loads the weights of the two networks that are used in the GAN on the 'filepath'.
        """
        #_#Steps\
        #_# Load Critic weights
        self.Critic.load_state_dict(torch.load(filepath + "_critic.pkl"))
        #_# Print that loading Critic weights is complete
        print("Critic model loaded from {}-".format(filepath + "_critic.pkl"))
        #_# Load generator weights
        self.Generator.load_state_dict(torch.load(filepath + "_generator.pkl"))
        #_# Print that loading generator weights is complete
        print("Generator model loaded from {}.".format(filepath + "_generator.pkl"))
