import numpy as np
from scipy.linalg import sqrtm
#___#---------------------------------------------------------------------------
#___#
#___#**Project:**         Masked Wasterstin generative adviersal network\
#___#**Script:**          fid.py\
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
#____#This sciprt is to calculate the Frechet inception distance:
#____#
#___#---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#_#
#__#1. Calculating the Frechet inception distance:
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This function calculates the
#_#Square root of ||mu_1 – mu_2||^2 + Tr(C_1 + C_2 – 2*sqrt(C_1*C_2))
#_#Reviewer Notes\

def calculate_fid(act1, act2):
    """
    Calculates the Frechet inception distance:
    d^2 = ||mu_1 – mu_2||^2 + Tr(C_1 + C_2 – 2*sqrt(C_1*C_2))
    """
    #_# Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    #_# Calculate the sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)  # ||mu_1 – mu_2||^2
    #_# Calculate sqrt of product between covariances
    covmean = 2.0 * sqrtm(sigma1.dot(sigma2))  # 2*sqrt(C_1*C_2)
    #_# Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    #_# Calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - covmean)
    return fid
