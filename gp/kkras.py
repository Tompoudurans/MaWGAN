from sklearn import datasets
#from tengan import dataGAN
from tendupden import WGANGP
#from dataman import show_loss_progress
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as mp

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

def get_norm(data):
    """
    Provides the mean and standard deviation for the dataset so it can be normalised.
    """
    mean = data.mean()
    std = data.std()
    data = normalize(data, mean, std)
    return data, mean, std

def show_loss_progress(loss_discriminator, loss_generator, filepath, extention=".pdf"):
    """
    This plots and saves the progress of the Loss function over time
    """
    mp.plot(loss_discriminator)
    mp.savefig(filepath + "_loss_progress_discriminator" + extention)
    mp.plot(loss_generator)
    mp.savefig(filepath + "_loss_progress_generator" + extention)

def dagplot(x,y,i):
    fake = pd.DataFrame(x)
    real = pd.DataFrame(y)
    fake['dataset'] = ['fake']*len(x)
    real['dataset'] = ['real']*len(y)
    result = pd.concat([real, fake])
    sns.pairplot(result,hue='dataset')
    mp.savefig(str(i) + "_compare.pdf")

batch = 150
iris = datasets.load_iris()
no_field = len(iris.data[1])
#mygan = dataGAN('adam',batch,no_field,batch)
mygan = WGANGP(input_dim = no_field
        , critic_learning_rate =0.4
        , generator_initial_dense_layer_size = 150
        , generator_learning_rate = 0.4
        , optimiser = 'adam'
        , grad_weight = 1
        , z_dim = 0
        , batch_size = batch
        , lambdas = 10
        )
#opti, noise_dim, no_field, batch, number_of_layers,pam6
norm_data, mean, standard = get_norm(iris.data)
mygan.critic.summary()
mygan.model.summary()
for i in range(50):
    mygan.train(norm_data,batch,100,'wgangp/',10,5)
    noise = np.random.normal(0, 1, (150, 150))
    generated_data = mygan.generator.predict(noise)
    unnormalize(generated_data, mean, standard)
    dagplot(generated_data,iris.data,i)
    run_folder=str(i) + "_"
    mygan.save_model(run_folder)
