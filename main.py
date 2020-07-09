from src.gans.tengan import dataGAN
from src.gans.wgan import wGAN
from src.tools.dataman import dagplot, show_loss_progress, save_data
from src.tools.fid import calculate_fid
from src.tools.prepocessing import import_penguin, unnormalize
from src.tools.core import set_core
from src.tools.sqlman import load_sql
from pandas import DataFrame
from sklearn import datasets
from math import ceil
import numpy as np
import click


@click.command()
@click.option("--mode", default="n", help="mode?(s)pyder/(n)ormal/(m)arathon)")
@click.option(
    "--filepath",
    prompt="filepath? ",
    help=" enter the file name and location of the database and model",
)
@click.option(
    "--epochs", prompt="epochs? ", help="choose how long that you want to train"
)
@click.option(
    "--dataset",
    default=None,
    help="chose the dataset/table that the GAN will to train on this can't be a single letter",
)
@click.option("--model", default=None, help="chose which model that you what to use")
@click.option("--opti", default=None, help="chose the optimiser that you want to use")
@click.option("--noise", default=None, help="chose the length of the noise vector")
@click.option(
    "--batch", default=None, help="chose how many fake data you want to make in one go"
)
@click.option(
    "--layers", default=None, help="chose the number of layers of each network"
)
@click.option("--clip", default=None, help="if using wgan chose the cliping threshold")
@click.option(
    "--core", default=0, type=int, help="select number of core that you like to run"
)
@click.option(
    "--sample",
    default=1,
    type=int,
    help="chose the number of generate data that you want: (samples*batch)",
)
def main(
    dataset,
    mode,
    filepath,
    epochs,
    model,
    opti,
    noise,
    batch,
    layers,
    clip,
    core,
    sample,
):
    """
    This code creates, loads and train a GAN from specified parameters
    This code is a copy of david foster book 'deep generative models' which
    has been modified to work with a numeric database
    """
    click.echo("loading...")
    if core != 0:
        set_core(core)
    parameters_list = [dataset, model, opti, noise, batch, layers, clip]
    parameters, successfully_loaded = parameters_handeling(filepath, parameters_list)
    epochs = int(epochs)
    database, mean, std, normalised, col = load_data(parameters[0], filepath)
    thegan = run(mode, filepath, epochs, parameters, successfully_loaded, database)
    fake = show_samples(
        thegan,
        mean,
        std,
        database,
        int(parameters[4]),
        normalised,
        sample,
        col,
        filepath,
    )
    save_data(fake, filepath)


def marathon_mode(mygan, database, batch, noise_dim, filepath, epochs):
    """
    In marathon mode the GAN is trained for 50000 epochs and substracted from the number of epochs left.
    Then the GAN model and the loss tracking is saved,
    the current loss tracking is removed from ram and a new set of training starts again
    at epoch 0. The result of the training is displayed and from there you can continue training if you wish.
    """
    while epochs > 0:
        mygan.train(database, batch, 50000, 1000)
        mygan.save_model(filepath)
        np.savetxt(filepath + str(epochs) + "_d_losses.txt", mygan.g_losses)
        np.savetxt(filepath + str(epochs) + "_g_losses.txt", mygan.g_losses)
        mygan.d_losses, mygan.g_losses = [], []
        epochs = epochs - 50000
        if epochs < 50000 and epochs > 0:
            print("almost done")
            mygan.train(database, batch, epochs, 1000)
            break
        if epochs == 0:
            noise = np.random.normal(0, 1, (noise_dim, batch))
            generated_data = mygan.generator.predict(noise)
            print(generated_data)
            dagplot(generated_data, database, filepath)
            calculate_fid(generated_data, database)
            epochs = int(input("continue?, enter n* of epochs"))


def unpack(p):
    """
    Unpacks the parameters
    """
    return p[1], p[2], int(p[3]), int(p[4]), int(p[5])


def setup(parameters_list):
    """
    Creates new parameters
    """
    parameters = []
    questions = [
        "dataset (table) ",
        "model?: (w)gan /(g)an ",
        "opti? ",
        "noise size? ",
        "batch size? ",
        "layers? ",
    ]
    for q in range(len(questions)):
        if parameters_list[q] != None:
            param = parameters_list[q]
        else:
            param = input(questions[q])
        parameters.append(param)
    if parameters[1] == "w":
        clip_threshold = float(input("clip threshold? "))
        parameters.append(clip_threshold)
    return parameters


def load_data(sets, filename):
    """
    Loads a dataset, choices are (i)ris (w)ine or (p)enguin
    """
    normalised = True
    if sets == "i":
        database = datasets.load_iris()
    elif sets == "w":
        database = datasets.load_wine()
    elif sets == "p":
        database, mean, std, col = import_penguin("data/penguins_size.csv", False)
    else:
        database, mean, std, col = load_sql(filename, sets)
    if sets == "i" or sets == "w":
        col = database.feature_names
        database = database.data
        mean = 0
        std = 1
        normalised = False
    return database, mean, std, normalised, col


def load_gan_weight(filepath, mygan):
    """
    Loads weight from previous trained GAN
    """
    try:
        mygan.load_weights(filepath)
    except OSError:  # as 'Unable to open file':
        print("file not found, starting from scratch")
    finally:
        return mygan


def create_model(parameters, no_field):
    """
    Builds the GAN using the parameters
    """
    use_model, opti, noise_dim, batch, number_of_layers = unpack(parameters)
    if use_model == "g":
        mygan = dataGAN(opti, noise_dim, no_field, batch, number_of_layers)
        mygan.discriminator.summary()
    elif use_model == "w":
        mygan = wGAN(opti, noise_dim, no_field, batch, number_of_layers, parameters[6])
        mygan.critic.summary()
    # print the stucture of the gan
    mygan.generator.summary()
    mygan.model.summary()
    return mygan, batch, noise_dim


def show_samples(mygan, mean, std, database, batch, normalised, samples, col, filepath):
    """
    Creates a number of samples
    """
    for s in range(int(samples)):
        generated_data = mygan.create_fake(batch)
        if normalised:
            generated_data = unnormalize(generated_data, mean, std)
            if s == 0:
                database = unnormalize(database, mean, std)
        else:
            generated_data = DataFrame(generated_data)
            if s == 0:
                database = DataFrame(database)
        print(generated_data)
        generated_data.columns = col
        database.columns = col
        calculate_fid(generated_data, database)
        dagplot(generated_data, database, filepath)
    return generated_data


def save_parameters(parameters, filepath):
    """
    Saves the parameters for the GAN
    """
    fname = filepath + "_parameters.npy"
    parameter_array = np.array(parameters)
    np.save(fname, parameter_array)


def load_parameters(filepath):
    """
    Loads the parameters for the GAN
    """
    try:
        parameter_array = np.load(filepath + "_parameters.npy", allow_pickle=True)
    except OSError:  # as 'Unable to open file':
        print("file not found, starting from scratch")
        successfully_loaded = False
        parameter_array = None
    else:
        successfully_loaded = True
    return parameter_array, successfully_loaded


def parameters_handeling(filepath, parameters_list):
    """
    Load parameters if they exist, otherwise saves new ones
    """
    parameters, successfully_loaded = load_parameters(filepath)
    if not successfully_loaded:
        parameters = setup(parameters_list)
        print(parameters)
        save_parameters(parameters, filepath)
    return parameters, successfully_loaded


def run(mode, filepath, epochs, parameters, successfully_loaded, database):
    """
    Creates and trains a GAN from the parameters provided.
    It will load the weights of the GAN if they exist.
    An option will be given to create samples.
    """
    # select dataset
    no_field = len(database[1])
    mygan, batch, noise_dim = create_model(parameters, no_field)
    if successfully_loaded:
        mygan = load_gan_weight(filepath, mygan)
    # marathon mode is not suitable when running less that 50000 epochs
    if epochs < 50000 and mode == "m":
        print("epochs too small, switch to normal ")
        mode = "n"
    if epochs > 0:
        step = int(ceil(epochs * 0.01))
        if mode == "m":
            marathon_mode(mygan, database, batch, noise_dim, filepath, epochs)
        else:
            # train the GAN according to the number of epochs
            mygan.train(database, batch, epochs, step)
        if mode == "s":
            # in spyder mode the GAN model is returned so it can be experimented on
            return mygan, database
        else:
            mygan.save_model(filepath)
        show_loss_progress(mygan.d_losses, mygan.g_losses, filepath)
        return mygan


if __name__ == "__main__":
    main()
