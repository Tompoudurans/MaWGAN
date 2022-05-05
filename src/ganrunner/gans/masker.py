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
#____#this script is part of MaWGAN core mecanic this create a mask from dataset
#____#which indicates where the missing data is, then applies it to another dataset
#____#which is most likely to be the generated data
#___#
#___#---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#_#
#__#1. make mask
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#this funtion make mask that indicates where the missing data is,
#_#Reviewer Notes\

def make_mask(data):
    """
    this funtion make mask that indicates where the missing data is
    """
    #_#Steps\
    #_#The first step is to create a binary matrix which flags True
    #_#if there is data missing in this location
    binary_mask = data.isnan()
    #_# convert the binary mask to a numeral mask
    inverse_mask = tensor(binary_mask, dtype=int)
    #_# inverse the 0s and 1s so 0 is where the data is missing
    mask = 1 - inverse_mask
    #_# outputs the numeral and binary mask
    return mask, binary_mask

#-------------------------------------------------------------------------------
#_#
#__#2. apply the mask
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#create a mask from the template and apply it to the data
#_#this operation made in the gpu if the the usegpu flag is set to True
#_#Reviewer Notes\
#_#
#_#

def copy_format(template, data, usegpu):
    """
    create a mask from the template and apply it to the data
    """
    #_#Steps\
    #_# create a mask see section 1
    mask, binary_mask = make_mask(template)
    #_# applys the mask to the generated data
    #_# the usegpu flag allows this operation made in the gpu
    if usegpu:
        masked_data = data * mask.cuda()
    else:
        masked_data = data * mask
    #_# replace missing values in the oringal dataset to 0
    template[binary_mask] = 0
    #_# outputs both datasets
    return template, masked_data
