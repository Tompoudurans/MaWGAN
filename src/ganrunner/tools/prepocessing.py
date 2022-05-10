import pandas
#___#---------------------------------------------------------------------------
#___#
#___#**Project:**         Masked Wasterstin generative adviersal network\
#___#**Script:**          preporcessing.py\
#___#**Author:**          Thomas Poudevigne\
#___#**Date Created:**    ‎21 ‎July ‎2021\
#___#**Reviewer:**        TBC\
#___#**Devops Feature:**  #[don't know]\
#___#**Devops Backlog:**  #[don't know]\
#___#**Devops Task:**     #[don't know]\
#___#**Devops Repo:**     ganrunner\tools\
#___#**MARS:**            "[don't know]"
#___#
#___#
#____#Description
#____#This script is for normalizing the dataset before training and
#____# un-normalizing it afterwards
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
    mid = df * std
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
