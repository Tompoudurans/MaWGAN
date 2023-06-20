import math
import numpy as np
import Mawgan.tools as tools
import Mawgan.gans as gans
import click
import logging
import os

#___#---------------------------------------------------------------------------
#___#
#___#**Project:**         Masked Wasterstin generative adviersal network\
#___#**Script:**          __main__.py\
#___#**Author:**          Thomas Poudevigne\
#___#**Date Created:**    21 July 2021\
#___#**Reviewer:**        TBC\
#___#**Devops Feature:**  #[don't know]\
#___#**Devops Backlog:**  #[don't know]\
#___#**Devops Task:**     #[don't know]\
#___#**Devops Repo:**     ganrunner\
#___#**MARS:**            "[don't know]"
#___#
#___#
#____#Description
#____#This script is the interface between the GAN software and the user.
#____#It pre-processes data, trains it and creates synthetic datasets.
#____#
#___#---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#_#
#__#1. Inputting parameters
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_# This function reads parameters and prepares the GAN
#_#Reviewer Notes\

#_# Steps\
#_# Document all the parameters needed for the GAN
@click.command()
@click.option("--test",default=None,help="test the intallment")
@click.option(
    "--filepath",
    help=" enter the file name and location of the database and model",
)
@click.option(
    "--epochs", help="choose how long that you want to train"
)
@click.option(
    "--dataset",
    default=None,
    help="choose the dataset/table that the GAN will train on",
)
@click.option("--opti", default=None, help="choose the optimiser you want to use")
@click.option("--nodes", default=None, help="choose the number nodes per layer")
@click.option(
    "--batch", default=None, help="choose how many datapoints is process when traing in one go"
)
@click.option(
    "--layers", default=None, help="choose the number of layers of each network"
)
@click.option("--lambdas", type=float, default=None, help="learning penalty")
@click.option(
    "--sample",
    default=1,
    type=int,
    help="choose the number of generated data",
)
@click.option(
    "--rate",
    default=None,
    type=float,
    help="choose the learing rate of the model",
)

@click.option(
    "--usegpu",
    default=0,
    type=int,
    help="set to 1 to use gpu",
)

def main(
    dataset,
    filepath,
    epochs,
    opti,
    nodes,
    batch,
    layers,
    lambdas,
    sample,
    rate,
    usegpu,
    test
):
    """
    This is the code for "MaWGAN: a Generative Adversarial Network to create synthetic
    data from datasets with missing data", this script is the interface of the MaGAN
    It pre-processes data, trains gan and creates synthetic datasets.
    """
    #_# check if testing for installment
    if test != None:
    #_#if installation is sucessfull then you should be able run to this point without error
        print("installation is sucessfully")
        #_# exit the script
        return
    #_# Tell user that the computer is processing the request
    click.echo("loading...")
    #_# Extract the extension of the file path
    filename, extention = filepath.split(".")
    if extention == "csv":
        dataset = True
    #_# Create the log file
    tools.setup_log(filename + "_progress.log")
    #_# Save the parameters to the parameters list
    parameters_list = [dataset, opti, nodes, batch, layers, lambdas, rate]
    #_# Load the previous parameters and check if any is missing
    parameters, successfully_loaded = parameters_handeling(filename, parameters_list)
    #_# Convert the epochs into an integer variable
    epochs = int(epochs)
    #_# Attempt to load the dataset and pre-process the dataset
    try:
        database, details, encoded = load_data(parameters[0], filepath, extention)
    except Exception as e:
        #_# State if a file does not exist
        print("Data could not be loaded propely see logs for more info")
        #_# Record the error
        logging.exception(e)
        #_# Exit the function
        return
    #_# Create the GAN using the parameters and train it
    thegan, success = run(filename, epochs, parameters, successfully_loaded, database, bool(usegpu))
    #_# Create an empty variable so it does not produce errors
    fake = None
    #_# Check if the GAN has trained sucessfully
    if success and sample > 0:
        #_# If it has trained sucessfully make a synthetic sample data
        fake = make_samples(
            thegan,
            encoded,
            sample,
            filename,
            details,
            extention,
            bool(usegpu),
        )
    #_# Output the GAN object and the synthetic data
    return thegan, fake
#_#

#-------------------------------------------------------------------------------
#_#
#__#2. Unpack parameters
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_# This function unpacks the parameters into individual variables
#_#Reviewer Notes\

#_# Steps\
#_# Outputs the parameters into individual variables

def unpack(p):
    """
    Unpacks the parameters
    """
    return p[1], int(p[2]), int(p[3]), int(p[4])

#-------------------------------------------------------------------------------
#_#
#__#3. Set-up
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_# This function checks missing parameters and prompts to fill gaps

#_#Reviewer Notes\

#_# Steps\
#_#

def setup(parameters_list,sucess,loaded_parameters):
    """
    Creates new parameters
    """
    #_# Create an empty list for checked parameters
    parameters = []
    #_# Setup the questions
    questions = [
        "table? ",
        "opti? ",
        "nodes size? ",
        "batch size? ",
        "layers? ",
        "learning constiction? ",
        "rate? ",
    ]
    #_#Loop over each parameter
    for q in range(len(questions)):
    #_# Check if the parameters exist
        if parameters_list[q] != None:
        #_# If the parameter exists put it in the waiting list to be added to the list of checked parameters
            param = parameters_list[q]
        elif sucess:
        #_#  If the parameters was loaded it added to the list of checked parameters
            param = loaded_parameters[q]
        else:
            # If it does not exist, prompt to fill it in, parameters 1 and 2 are strings
            if q < 3:
                param = input(questions[q])
            #_# parameters 3, 4 and 5 are intergers
            elif q < 6:
                param = input_int(questions[q])
            #_# parameters 6 and 7 are floats
            else:
                param = input_float(questions[q])
        #_# Save the answers and transfer the checked parameters into the checked parameters list
        parameters.append(param)
    #_# Save the parameters list
    return parameters


#-------------------------------------------------------------------------------
#_#
#__#3. Checking the responses to questions
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_# This function checks if the response to the question is an integer

#_#Reviewer Notes\

#_# Steps\
#_#

def input_int(question):
    """
    makes sure a number is inputed
    """
    #_# Loop until the response to the question is an integer
    while True:
        try:
            #_# Ask the question
            a = input(question)
            #_# Attempt to convert the answer to an integer
            answer = int(a)
            #_# If failed, state that the answer must be an integer
        except Exception:
            print("the answer must be an integer")
            #_# Can exit programme by entering nothing
            if a == "" or None:
                raise RuntimeError
        #_# If the conversion does not fail return the converted answer
        else:
            return answer

#-------------------------------------------------------------------------------
#_#
#__#4. Checking the responses to questions
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_# This function checks if the response to the question is a float

#_#Reviewer Notes\

#_# Steps\
#_#

def input_float(question):
    """
    makes sure a number is inputed
    """
    #_# Loop until the response to the question is an float
    while True:
        try:
            #_# Ask the question
            a = input(question)
            #_# Attempt to convert the answer to an float
            answer = float(a)
            #_# If failed, state that the answer must be an float
        except Exception:
            print("the answer must be an float")
            #_# Can exit programme by entering nothing
            if a == "" or None:
                raise RuntimeError
        #_# If the conversion does not fail return the converted answer
        else:
            return answer
#-------------------------------------------------------------------------------
#_#
#__#5. Loading the data
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_# This function loads the data and pre-processes it

#_#Reviewer Notes\

#_# Steps\
#_#
#_#
def load_data(sets, filepath, extention):
    """
    Loads a dataset from an sql table
    """
#_# Check if the file format is a db file
    if extention == "db":
        #_# Load the file
        raw_data = tools.load_sql(filepath, sets)
        #_# Check if the file format is csv
    if extention == "csv":
        #_# Load the file
        raw_data = tools.pd.read_csv(filepath)
        #_# Pre-process the data
    database, details, encoded = tools.procsses_data(raw_data)
    #_# Output the database and the mapping
    return database, details, encoded

#-------------------------------------------------------------------------------
#_#
#__#6. Loading the GAN weights
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_# This function loads the weights o the GAN

#_#Reviewer Notes\

#_# Steps\
#_#
def load_gan_weight(filepath, mygan):
    """
    Loads weight from previous trained GAN
    """
#_# Attempt to load the weights of the GAN
    try:
        mygan.load_model(filepath)
        #_# If this fails, state so
    except OSError:  # as 'Unable to open file':
        print("No previous gan exsit, starting from scratch")
    finally:
        #_# Ouput the GAN class
        return mygan
#-------------------------------------------------------------------------------
#_#
#__# 7. Creating the GAN class
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_# This function creates the GAN class according to the given parameters

#_#Reviewer Notes\

#_# Steps\
#_#

def create_model(parameters, no_field):
    """
    Builds the GAN using the parameters
    """
#_# Take the learning rate (lr) from the parameter list
    lr = float(parameters[6])
    #_# Umpack the rest of the parameters
    opti, nodes_dim, batch, number_of_layers = unpack(parameters)
    #_# Create the GAN class modelusing the list of parameters
    mygan = gans.MaWGAN(
        optimiser=opti,
        number_of_variables=no_field,
        number_of_nodes=nodes_dim,
        number_of_layers=number_of_layers,
        lambdas=float(parameters[5]),
        learning_rate=lr,
    )
    #_# Print the configuration
    mygan.summary()
    #_# Output the GAN class
    return mygan

#-------------------------------------------------------------------------------
#_#
#__#8. Making samples
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_# This function creates synthetic data and saves them to a file
#_#Reviewer Notes\

#_# Steps\
#_#
def make_samples(
    mygan, database, batch, filepath, details, extention, usegpux
):
    """
    Creates a number of samples
    """
    #_# Unpack the detail list
    mean, std, info, col = details
    #_# Create an empty variable called fullset
    fullset = None
    #_# Create a batch of synthetic data
    generated_data = mygan.create_synthetic(batch)
    # sample the dataset
    data_sample = database.sample(batch)
    #_# Un-normalise the batch of synthetic data
    generated_data = tools.unnormalize(tools.pd.DataFrame(generated_data), mean, std)
    #_# Relabel the variables as the previous format could not label them
    generated_data.columns = col
    try:
        #_# calculate ls score
        if usegpux:
            ls = tools.gpu_LS(data_sample.dropna().to_numpy(),generated_data.to_numpy())
        else:
            ls = tools.LS(data_sample.dropna().to_numpy(),generated_data.to_numpy())
        #_# calculate fid score
        fid = tools.calculate_fid(data_sample.dropna(),generated_data)
        #_#print comparison
        print(" LS: ",ls,"\n FID:",fid)
    except Exception as e:
       print("comparison calculation failed")
       logging.error("comparison calculation failed due to" + str(e))
    #_# Restore the categorical variables
    fullset = tools.decoding(generated_data, info)
    #_# Save the synthetic data in the same file format as the original,
    #_# if the file format is db then save in db, if the file format is csv then save in csv
    if extention == "db":
        tools.save_sql(fullset, filepath + ".db")
    if extention == "csv":
        fullset.to_csv(filepath + "_synthetic.csv", index=False)
    #_# Output the synthetic data
    return fullset

#-------------------------------------------------------------------------------
#_#
#__#9. Saving the parameters
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_# This function saves the parameters so that the GAN can be rebuilt the same way

#_#Reviewer Notes\

#_# Steps\
#_#
def save_parameters(parameters, filepath):
    """
    Saves the parameters for the GAN
    """
#_# Create a filename for the parameters to be saved into
    fname = filepath + "_parameters.npy"
    #_# Convert the list to an array
    parameter_array = np.array(parameters)
    #_# Save the array
    np.save(fname, parameter_array)

#-------------------------------------------------------------------------------
#_#
#__#10. Loading the parameters
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_# This function loads previously saved parameters
#_#Reviewer Notes\

#_# Steps\
#_#
def load_parameters(filepath):
    """
    Loads the parameters for the GAN
    """
#_# Attempt to load the parameters
    try:
        parameter_array = np.load(filepath + "_parameters.npy", allow_pickle=True)
        #_# Otherwise print an error message
    except OSError:
        print("parameters file not found, starting from scratch")
        #_# Create a flag to state that no parameters have been loaded
        successfully_loaded = False
        parameter_array = None
    else:
        #_# Create a flag that states that the parameters have been loaded
        successfully_loaded = True
    return parameter_array, successfully_loaded

#-------------------------------------------------------------------------------
#_#
#__#11. Handeling the parameters
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_# This function loads parameters if the file exists otherwise it
#_# takes parameters that the user inputs

#_#Reviewer Notes\

#_# Steps\
#_#
def parameters_handeling(filepath, parameters_list):
    """
    Load parameters if they exist, otherwise saves new ones
    """
#_# Load parameters from the existing file
    loaded_parameters, successfully_loaded = load_parameters(filepath)
    #_# check if nothing is missing in the parameter list
    parameters = setup(parameters_list,successfully_loaded,loaded_parameters)
    #_# Save the new parameters
    save_parameters(parameters, filepath)
    #_# Print the parameters
    print(parameters)
    #_# Output that the parameters and wether they have been sucessfully loaded
    return parameters, successfully_loaded

#-------------------------------------------------------------------------------
#_#
#__#12. Running the main GAN script
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_# This function runs the main GAN script: It creates and trains a GAN
#_# from the parameters provided.
#_# It will load the weights of the GAN if they exist.

#_#Reviewer Notes\

#_# Steps\
#_#
def run(filepath, epochs, parameters, successfully_loaded, database, usegpu):
    """
    Creates and trains a GAN from the parameters provided.
    It will load the weights of the GAN if they exist.
    """
    #_# Count the number of variables
    no_field = len(database[1])
    try:
        #_# Attempt to create the model
        mygan = create_model(parameters, no_field)
        #_# Otherwise state that the building has failed, exit programme
    except Exception as e:
        print("building failed, check you parameters")
        logging.error("building failed due to" + str(e))
        return None, False
        #_# Do not load weights if parameters not loaded
    if successfully_loaded:
        #_# Load the GAN weights to the network
        mygan = load_gan_weight(filepath, mygan)
        #_# If the epoch is null then do not attempt to train
    if epochs > 0:
        #_#  Calculate a tenth of the total epochs that will be how often the loss will be recorded
        step = int(math.ceil(epochs * 0.1))
        #_# Prepare the checking of missing data
        checkI = tools.pd.DataFrame(database)
        #_#  Check if any data is missing and if so run MAWGAN instead of WGAN-GP
        checkII = checkI.isnull().sum().sum() > 0
        #_# Attempt to train the GAN
        try:
            mygan.train(database, int(parameters[3]), epochs, checkII, step, usegpu=usegpu)
            #_# State that the training has failed to check parameters if the training failed
        except Exception as e:
            logging.exception(e)
            print("training failed check you parameters")
            #_# and exit the programme
            return None, False
        else:
            #_# If training is sucessfull, save the model
            mygan.save_model(filepath)
    #_# Return the GAN class and whether or not the programme was sucessfull
    return mygan, True


if __name__ == "__main__":
    main()
