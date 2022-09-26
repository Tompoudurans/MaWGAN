from torch import tensor

#___#---------------------------------------------------------------------------
#___#
#___#**Project:**         Masked Wasterstin generative adviersal network\
#___#**Script:**          masker.py\
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
    inverse_mask = binary_mask.to(dtype=int)
    #_# Inverse the 0s and 1s so 0 is where the data is missing
    mask = 1 - inverse_mask
    #_# Outputs the numeral and binary mask
    return mask, binary_mask
