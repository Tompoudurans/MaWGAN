
#from sklearn import datasets
import numpy as np
import ganrunner
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as mp

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

def import_penguin(file):
    """
    Imports the penguin dataset then processes the data so that it is ready to be trained.
    It sets the categorical data into numerical data
    """
    penguin = pd.read_csv(file)
    penguin = penguin.dropna()
    penguin = penguin.drop(columns=["sex", "species", "island"])
    penguin, mean, std = get_norm(penguin)
    return penguin, mean, std

def get_norm(data):
    """
    Provides the mean and standard deviation for the dataset so it can be normalised.
    """
    mean = data.mean()
    std = data.std()
    data = normalize(data, mean, std)
    return data.to_numpy("float"), mean.to_numpy("float"), std.to_numpy("float")

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
    df = pd.DataFrame(dataset)
    mid = df * std
    original = mid + mean
    return original

file = input('name')
batch = 170
vector = 170
epoch = 300
clip = 1
number_of_layers = 5
iris, mean, std = import_penguin('penguins_size.csv')
no_field = 4
mygan = ganrunner.wGANgp('RMSprop',batch,no_field,batch,clip,number_of_layers)
mygan.critic.summary()
mygan.generator.summary()
mygan.train(iris,batch,epoch,10)

real = unnormalize(iris, mean, std)
noise = np.random.normal(0, 1, (batch, batch))
generated_data = mygan.generator.predict(noise)
fake=unnormalize(generated_data, mean, std)
dagplot(fake, real, file)
