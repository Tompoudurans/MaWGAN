# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:24:13 2019

@author: user
"""
import seaborn as sns
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as mp


def simplesplit(x, fac=10):
    """
    Randomly splits the dataset into 2 parts
    """
    size = len(x)
    z = np.split(rd.sample(range(size), size), [int(size * (1 - fac / 100))])
    return x[z[0]], x[z[1]]


def dagplot(x, y, filepath, extention=".pdf"):
    """
    plots original data vs the synthetic data then saves
    """
    fake = pd.DataFrame(x)
    real = pd.DataFrame(y)
    fake["dataset"] = ["fake"] * len(x)
    real["dataset"] = ["real"] * len(y)
    result = pd.concat([real, fake])
    sns.pairplot(result, hue="dataset")
    mp.savefig(filepath + "_compare" + extention)


def show_loss_progress(loss_discriminator, loss_generator, filepath, extention=".pdf"):
    """
    This plots and saves the progress of the Loss function over time
    """
    mp.plot(loss_discriminator)
    mp.savefig(filepath + "_loss_progress_discriminator" + extention)
    mp.plot(loss_generator)
    mp.savefig(filepath + "_loss_progress_generator" + extention)


def save_data(df, file):
    """
    Saves the synthetic data onto a csv. file
    """
    try:
        df = df.drop(columns=["dataset"])
    except KeyError:
        print("failed")
    finally:
        df.to_csv(file + "_synthetic.csv")
