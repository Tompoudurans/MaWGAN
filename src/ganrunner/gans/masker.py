from torch import tensor

#___#---------------------------------------------------------------------------
#___#
#___#**Project:**         Masked Wasterstin generative adviersal network\
#___#**Script:**          masker.py\
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
#____#This script is part of the MaWGAN core mechanic.
#____#This script creates a mask from an 'template' dataset
#____#which indicates where the missing data is, then applies it to another dataset
#____#which is usually the generated data.
#___#
#___#---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#_#
#__#1. Make mask
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function makes a mask that indicates where the missing data is, it
#_# outputs a numeral and binary mask.
#_#Reviewer Notes\

def make_mask(data):
    """
    This function makes a mask that indicates where the missing data is
    """
    #_#Steps\
    #_#The first step is to create a binary matrix which flags True
    #_#if there is data missing in this location
    binary_mask = data.isnan()
    #_# Convert the binary mask to a numeral mask
    inverse_mask = tensor(binary_mask, dtype=int)
    #_# Inverse the 0s and 1s so 0 is where the data is missing
    mask = 1 - inverse_mask
    #_# Outputs the numeral and binary mask
    return mask, binary_mask

#-------------------------------------------------------------------------------
#_#
#__#2. Apply the mask
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function creates a mask from the template and applies it to the data.
#_#This operation can be used in the gpu if the the usegpu flag is set to True.
#_#Reviewer Notes\
#_#
#_#

def copy_format(template, data, usegpu):
    """
    This function creates a mask from the template and applies it to the data.
    """
    #_#Steps\
    #_# Create a mask see section 1
    mask, binary_mask = make_mask(template)
    #_# Applies the mask to the generated data
    #_# The usegpu flag allows this operation to be made in the gpu
    if usegpu:
        masked_data = data * mask.cuda()
    else:
        masked_data = data * mask
    #_# Replace missing values in the original dataset to 0
    template[binary_mask] = 0
    #_# Outputs both datasets
    return template, masked_data
