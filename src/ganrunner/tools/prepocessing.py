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
#____#This script is for normalizing dataset before training and unnormailzing it
#____#
#___#
#___#---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#_#
#__#1. normalizing
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#Normalises the dataset by mean and standard deviation
#_#Reviewer Notes\

def normalize(dataset, mean, std):
    """
    Normalises the dataset by mean and standard deviation
    """
    #_#Steps\
    #_# substract the mean from the values of the dataset
    mid = dataset - mean
    #_# then divied by the standard deviation.
    return mid / std

#-------------------------------------------------------------------------------
#_#
#__#2. unnormalizing
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#Reverts the normalised dataset to original format, the inverse of the normalize function (1)
#_#Reviewer Notes\
def unnormalize(dataset, mean, std):
    """
    Reverts the normalised dataset to original format
    """
    #_#Steps\
    #_# mutiply the dataset by the standard deviation.
    mid = df * std
    #_# then adds the mean
    return mid + mean

#-------------------------------------------------------------------------------
#_#
#__#3.Normalize a dataset
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#Provides the mean and standard deviation for the dataset so it can be normalised.
#_#Then send to the normalize function (1)
#_# outputs the normalize dataset, the mean and the standard deviation
#_# in a numpy format as it can be used in torch libary
#_#Reviewer Notes\
def get_norm(data):
    """
    Provides the mean and standard deviation for the dataset and then normalising it
    """
    #_#Steps\
    #_#find the mean
    mean = data.mean()
    #_# find the standard divation
    std = data.std()
    #_# send to the normalize function (1)
    data = normalize(data, mean, std)
    #_# outputs the normalize dataset, the mean and the standard deviation
    return data.to_numpy("float"), mean.to_numpy("float"), std.to_numpy("float")
