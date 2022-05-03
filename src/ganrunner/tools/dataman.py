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
import logging


def dagplot(synthetic, original, filepath, extention=".pdf"):
    """
    plots original data vs the synthetic data then saves
    """
    fake = pd.DataFrame(synthetic)
    real = pd.DataFrame(original)
    fake["dataset"] = ["synthetic"] * len(synthetic)
    real["dataset"] = ["original"] * len(original)
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
    logging.info(loss_discriminator)
    logging.info(loss_generator)


def setup_log(filepath):
    """
    creates a log file
    """
    logging.basicConfig(
        filename=filepath,
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s: \n%(message)s",
    )
