from src.main.tengan import dataGAN
from src.main.wgan import wGAN
from src.tools.dataman import dagpolt,show_loss_progress
from src.tools.fid import calculate_fid
from src.tools.prepocessing import import_penguin,unnormalize
from src.tools.core import set_core
from sklearn import datasets
from math import ceil
import numpy as np
import click


@click.command()
@click.option("--mode", default="n", help="mode?(s)pyder/(n)ormal/(m)arathon)")
@click.option("--filepath", prompt="filepath? ")
@click.option("--epochs", prompt="epochs? ")
@click.option("--dataset", default=None)
@click.option("--model", default=None)
@click.option("--opti", default=None)
@click.option("--noise", default=None)
@click.option("--batch", default=None)
@click.option("--layers", default=None)
@click.option("--clip", default=None)
@click.option("--core", default=0,type=int)


def main(dataset, mode, filepath, epochs, model, opti, noise, batch, layers, clip, core):
    """
    creates and trained gan from specified parameters it will also can load a model if it exist
    """
    click.echo("loading...")
    if core != 0:
        set_core(core)
    parameters_list = [dataset, model, opti, noise, batch, layers, clip]
    parameters, successfully_loaded = parameters_handeling(filepath, parameters_list)
    epochs = int(epochs)
    run(mode, filepath, epochs, parameters, successfully_loaded)


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
            dagpolt(generated_data, database)
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
        "set? 'w'/'i'/'p' ",
        "model?: (w)gan /(g)an ",
        "opti? ",
        "noise size? ",
        "batch size? ",
        "layers?",
    ]
    for q in range(len(questions)):
        if parameters_list[q] != None:
            param = parameters_list[q]
        else:
            param = input(questions[q])
        if q < 5:
            parameters.append(param)
        else:
            parameters.append(int(param))
    if parameters[1] == "w":
        clip_threshold = float(input("clip threshold? "))
        parameters.append(clip_threshold)
    return parameters


def load_data(sets):
    """
    Loads a dataset, choices are (i)ris (w)ine or (p)enguin
    """
    if sets == "i":
        database = datasets.load_iris()
    elif sets == "w":
        database = datasets.load_wine()
    elif sets == "p":
        database, mean, std = import_penguin("data/penguins_size.csv", False)
    else:
        return None
    if sets == "i" or sets == "w":
        database = database.data
        mean = 0
        std = 1
    return database, mean, std


def load_gan_weight(filepath, mygan):
    """
    Loads weight from previous trained GAN
    """
    try:
        mygan.load_weights(filepath)
    except OSError:  # as 'Unable to open file':
        print("Error:404 file not found, starting from scratch")
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


def show_samples(mygan, mean, std, database, batch, sets):
    """
    Creates a number of samples
    """
    samples = input("samples? ")
    for s in range(int(samples)):
        generated_data = mygan.create_fake(batch)
        if sets == "p":
            generated_data = unnormalize(generated_data, mean, std)
            if s == 0:
                database = unnormalize(database, mean, std)
        print(generated_data)
        dagpolt(generated_data, database)
        calculate_fid(generated_data, database)


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
        print("Error:404 file not found, starting from scratch")
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


def run(mode, filepath, epochs, parameters, successfully_loaded):
    """
    Creates and trains a GAN from the parameters provided.
    It will load the weights of the GAN if they exist.
    An option will be given to create samples.
    """
    # select dataset
    database, mean, std = load_data(parameters[0])
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
        show_loss_progress(mygan.d_losses, mygan.g_losses)
        show_samples(mygan, mean, std, database, batch, parameters[0])
    if mode == "s":
        return mygan


if __name__ == "__main__":
    main()
