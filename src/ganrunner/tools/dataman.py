# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:24:13 2019

@author: user
"""
import pandas as pd
import numpy as np
import random as rd
import logging


def simplesplit(x, fac=10):
    """
    Randomly splits the dataset into 2 parts
    """
    size = len(x)
    z = np.split(rd.sample(range(size), size), [int(size * (1 - fac / 100))])
    return x[z[0]], x[z[1]]


def setup_log(filepath):
    """
    creates a log file
    """
    logging.basicConfig(
        filename=filepath,
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s: \n%(message)s",
    )
