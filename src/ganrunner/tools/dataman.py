# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:24:13 2019

@author: user
"""
import seaborn as sns
import pd as pd
import numpy as np
import random as rd
import matplotlib.pyplot as mp
import logging
import sqlalchemy as sa
from scipy.linalg import sqrtm

def normalize(dataset, mean, std):
    """
    Normalises the dataset by mean and standard deviation
    """
    mid = dataset - mean
    new_data = mid / std
    return new_data


def unnormalize(dataset, mean, std):
    """
    Reverts the normalised dataset to original format
    """
    df = pandas.DataFrame(dataset)
    mid = df * std
    original = mid + mean
    return original


def get_norm(data):
    """
    Provides the mean and standard deviation for the dataset so it can be normalised.
    """
    mean = data.mean()
    std = data.std()
    data = normalize(data, mean, std)
    return data.to_numpy("float"), mean.to_numpy("float"), std.to_numpy("float")


def encoding(data):
    """
    Transforms categorical data into numerical, saves maping on a list.
    """
    details = [len(data.columns)]
    count_o = 0
    for name in data.columns:
        if "O" == data[name].dtype:
            new = pd.get_dummies(data[name])
            data[new.columns] = new
            data = data.drop(columns=name)
            details.append([name, len(new.columns)])
            count_o = count_o + 1
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
    data = data.drop(columns=data.columns[range(start, col_len)])
    return data

def calculate_fid(act1, act2):
    """
    Calculates the Frechet inception distance:
    d^2 = ||mu_1 – mu_2||^2 + Tr(C_1 + C_2 – 2*sqrt(C_1*C_2))
    """
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means,
    ssdiff = np.sum((mu1 - mu2) ** 2.0)  # ||mu_1 – mu_2||^2
    # calculate sqrt of product between cov
    covmean = 2.0 * sqrtm(sigma1.dot(sigma2))  # 2*sqrt(C_1*C_2)
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - covmean)
    print("FID:", fid)
    return fid

def load_sql(file, table):
    """
    Loads an SQL table
    """
    engine = sa.create_engine("sqlite:///" + file)
    connection = engine.connect()
    database = pd.read_sql(table, connection)
    try:
        return database.drop(columns="index")
    except KeyError:
        return database

def format_missing(data):
    """
    reformat missing data to the one used in the software
    """
    term = ""
    return data.replace(term,np.nan)

def procsses_sql(database):
    """
    pre-procsses the table, ready to be trained
    """
    database = format_missing(database)
    database, details = encoding(database)
    col = database.columns
    database, mean, std = get_norm(database)
    return database, [mean, std, details, col]


def save_sql(df, file, exists="append"):
    """
    Saves the generated data to a SQL table called generated_data
    """
    engine = sa.create_engine("sqlite:///" + file)
    try:
        df = df.drop(columns=["dataset"])
    except KeyError:
        pass
    df.to_sql("torch_generated_data", con=engine, if_exists=exists)  # , index=False)


def all_tables(file):
    """
    Reads all tables in a sql file
    """
    engine = sa.create_engine("sqlite:///" + file)
    connection = engine.connect()
    inspector = sa.inspect(engine)
    return inspector.get_table_names()

def simplesplit(x, fac=10):
    """
    Randomly splits the dataset into 2 parts
    """
    size = len(x)
    z = np.split(rd.sample(range(size), size), [int(size * (1 - fac / 100))])
    return x[z[0]], x[z[1]]


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
