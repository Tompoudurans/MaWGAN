import pandas
#___#---------------------------------------------------------------------------
#___#
#___#**Project:**         Masked Wasterstin generative adviersal network\
#___#**Script:**          preporcessing.py\
#___#**Author:**          Thomas Poudevigne\
#___#**Date Created:**    21 July 2021\
#___#**Reviewer:**        TBC\
#___#**Devops Feature:**  #[don't know]\
#___#**Devops Backlog:**  #[don't know]\
#___#**Devops Task:**     #[don't know]\
#___#**Devops Repo:**     ganrunner\tools\
#___#**MARS:**            "[don't know]"
#___#
#___#
#____#Description
#____#This script is for pre-process the dataset before training and
#____# post-process it afterwards
#____#
#___#---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#_#
#__#1. Normalizing
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function normalises the dataset using mean and standard deviation
#_#Reviewer Notes\

def normalize(dataset, mean, std):
    """
    This function normalises the dataset by mean and standard deviation
    """
    #_#Steps\
    #_# Substract the mean from the values of the dataset
    mid = dataset - mean
    #_# Then divide by the standard deviation
    return mid / std

#-------------------------------------------------------------------------------
#_#
#__#2. Un-normalizing
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function reverts the normalised dataset to its original format,
#_#by using the inverse of the normalize function (1)
#_#Reviewer Notes\
def unnormalize(dataset, mean, std):
    """
    Reverts the normalised dataset to the original format
    """
    #_#Steps\
    #_# Mutiply the dataset by the standard deviation.
    mid = dataset * std
    #_# Then add the mean
    return mid + mean

#-------------------------------------------------------------------------------
#_#
#__#3.Normalize a dataset
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function provides the mean and standard deviation for the dataset
#_#so it can be normalised.
#_#It then sends the mean and standard deviation to the normalize function (1)
#_# It outputs the normalized dataset, the mean and the standard deviation
#_# in a numpy format so it can be used in torch library
#_#Reviewer Notes\
def get_norm(data):
    """
    Provides the mean and standard deviation for the dataset and then normalises it
    """
    #_#Steps\
    #_#Find the mean
    mean = data.mean()
    #_# Find the standard deviation
    std = data.std()
    #_# Send the above to the normalize function (1)
    data = normalize(data, mean, std)
    #_# Outputs the normalize dataset, the mean and the standard deviation
    return data.to_numpy("float"), mean.to_numpy("float"), std.to_numpy("float")

#-------------------------------------------------------------------------------
#_#
#__#5.Data processing
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function processes data ready for GAN training
#_#Reviewer Notes\

def procsses_data(database):
    """
    pre-processes the table, ready to be trained
    """
    #_#Steps/
    #_#Encode categorical data
    encoded_data, details = encoding(database)
    #_# Save the variable names
    col = encoded_data.columns
    #_# Normalize the dataset
    normalise_data, mean, std = get_norm(encoded_data)
    #_# Ouput the preprocessed dataset and its details
    return normalise_data, [mean, std, details, col], encoded_data 
#-------------------------------------------------------------------------------
#_#
#__#6.Encoding
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function transforms categorical data into numerical data, and saves mapping on a list.
#_#Reviewer Notes\
def encoding(data):
    """
    Transforms categorical data into numerical, saves mapping on a list.
    """
#_# Steps
#_# Create the detail list which will save the mappings.
#_# This saves the current number of variables to the list.
    details = [len(data.columns)]
#_# Create a counter to count the ampount of avariables in the dataset
    count_o = 0
#_# Loop through each variable
    for name in data.columns:
        #_#If a variable is categorical
        if "O" == data[name].dtype:
            #_# encode the variable
            new = pandas.get_dummies(data[name])
            #_# Add the coded variable to the dataset
            data[new.columns] = new
            #_# Drop the old variable
            data = data.drop(columns=name)
            #_# Save the mapping to the list
            details.append([name, len(new.columns)])
            #_# Add one to the counter
            count_o = +1
    #_# Print the number of categorical variables
    print("there are", count_o, "categorical data variables")
    #_# Output the new dataset and the mapping.
    return data, details
#-------------------------------------------------------------------------------
#_#
#__#6.Decoding
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function transforms numerical data into categorical data using the saved mapping (details)
#_#Reviewer Notes\

def decoding(data, details):
    """
    Transforms numerical data into categorical data using the saved mapping (details)
    """
#_# Steps
#_# Save the current number of variables
    col_len = len(data.columns)
    #_# Initialise position counter which is the old number of variables minus
    #_# the number of categorical variables
    position = details[0] - len(details) + 1
    #_# Initialise the position in the detail list
    current = 1
    #_# Loop until the position counter is equal to the number of variables
    while position < col_len:
    #_# Record the end position of the encoded group of variables.
        try:
            end = position + details[current][1]
    #_# If the position does not exist exit the loop.
        except IndexError:
            break
    #_# Grab the encoded group of variables
        set_of_cat = data.iloc[:, position:end]
    #_# Create an empty list
        restore = []
        #_# If the group of encoded variables is binary:
        if set_of_cat.shape[1] == 1:
            #_# then round the data values to the categorical equivalent
            data[details[current][0]] = set_of_cat.round()
        else:
            #_# If not, then loop through each variable in the group
            for value in range(set_of_cat.shape[0]):
            #_# to check which one is closest to the categorical equivalent
                restore.append(set_of_cat.iloc[value].idxmax())
            #_# Save to the dataset
            data[details[current][0]] = restore
        #_# Move the detail list counter by one
        current = current + 1
        #_# Move the position counter to the end of the group of variables
        position = end
    #_# Return the restored dataset
    return data
