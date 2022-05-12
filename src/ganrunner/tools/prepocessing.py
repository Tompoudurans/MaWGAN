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

#-------------------------------------------------------------------------------
#_#
#__#4.Normalize a dataset
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function process data ready for gan training
#_#Reviewer Notes\

def procsses_data(database):
    """
    pre-procsses the table, ready to be trained
    """
    #_#steps/
    database, details = encoding(database)
    col = database.columns
    database, mean, std = get_norm(database)
    return database, [mean, std, details, col]

def encoding(data):
    """
    Transforms categorical data into numerical, saves maping on a list.
    """
    details = [len(data.columns)]
    count_o = 0
    for name in data.columns:
        if "O" == data[name].dtype:
            new = pandas.get_dummies(data[name])
            data[new.columns] = new
            data = data.drop(columns=name)
            details.append([name, len(new.columns)])
            count_o = +1
    print("there are", count_o, "categorical data variables")
    return data, details


def decoding(data, details):
    """
    Transforms numerical data into categorical data using the saved mapping (details)
    """
    col_len = len(data.columns)
    position = details[0] - len(details) + 1
    start = position
    current = 1
    while position < col_len:
        try:
            end = position + details[current][1]
        except IndexError:
            break
        set_of_cat = data.iloc[:, position:end]
        restore = []
        if set_of_cat.shape[1] == 1:
            data[details[current][0]] = set_of_cat.round()
        else:
            for value in range(set_of_cat.shape[0]):
                restore.append(set_of_cat.iloc[value].idxmax())
            data[details[current][0]] = restore
        current = current + 1
        position = end
    return data
